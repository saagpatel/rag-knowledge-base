"""Unit tests for QdrantManager — all Qdrant SDK calls mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    SparseVector,
)

from rag_kb.core.config import QdrantConfig
from rag_kb.core.errors import QdrantError
from rag_kb.core.qdrant_client import QdrantManager


@pytest.fixture
def config() -> QdrantConfig:
    return QdrantConfig(
        host="127.0.0.1",
        port=6333,
        grpc_port=6334,
        collection_prefix="rag_",
        hnsw_m=16,
        hnsw_ef_construct=200,
    )


@pytest.fixture
def manager(config: QdrantConfig) -> QdrantManager:
    mgr = QdrantManager(config=config)
    mgr._client = AsyncMock()
    return mgr


def _fake_point(id: str | int, score: float = 0.9) -> MagicMock:
    p = MagicMock()
    p.id = id
    p.score = score
    p.payload = {"text": "sample", "file_type": "markdown"}
    return p


def _fake_query_response(points: list | None = None) -> MagicMock:
    resp = MagicMock()
    resp.points = points or []
    return resp


def _fake_collection(name: str) -> MagicMock:
    c = MagicMock()
    c.name = name
    return c


# --- collection prefix ---


def test_collection_name_prefixing(manager: QdrantManager) -> None:
    assert manager._prefixed("docs") == "rag_docs"


def test_no_double_prefix(manager: QdrantManager) -> None:
    assert manager._prefixed("rag_docs") == "rag_docs"


# --- create_collection ---


async def test_create_collection(manager: QdrantManager) -> None:
    manager._client.collection_exists = AsyncMock(return_value=False)
    manager._client.create_collection = AsyncMock()

    await manager.create_collection("docs")

    manager._client.collection_exists.assert_called_once_with("rag_docs")
    manager._client.create_collection.assert_called_once()

    call_kwargs = manager._client.create_collection.call_args.kwargs
    assert call_kwargs["collection_name"] == "rag_docs"

    dense_params = call_kwargs["vectors_config"]["dense"]
    assert dense_params.size == 768
    assert dense_params.distance == Distance.COSINE
    assert dense_params.hnsw_config.m == 16
    assert dense_params.hnsw_config.ef_construct == 200

    assert "sparse" in call_kwargs["sparse_vectors_config"]


async def test_create_collection_already_exists(manager: QdrantManager) -> None:
    manager._client.collection_exists = AsyncMock(return_value=True)
    manager._client.create_collection = AsyncMock()

    await manager.create_collection("docs")

    manager._client.create_collection.assert_not_called()


async def test_create_collection_custom_dim(manager: QdrantManager) -> None:
    manager._client.collection_exists = AsyncMock(return_value=False)
    manager._client.create_collection = AsyncMock()

    await manager.create_collection("docs", dense_dim=384)

    call_kwargs = manager._client.create_collection.call_args.kwargs
    assert call_kwargs["vectors_config"]["dense"].size == 384


# --- delete_collection ---


async def test_delete_collection(manager: QdrantManager) -> None:
    manager._client.collection_exists = AsyncMock(return_value=True)
    manager._client.delete_collection = AsyncMock()

    await manager.delete_collection("docs")

    manager._client.delete_collection.assert_called_once_with("rag_docs")


async def test_delete_collection_not_exists(manager: QdrantManager) -> None:
    manager._client.collection_exists = AsyncMock(return_value=False)
    manager._client.delete_collection = AsyncMock()

    await manager.delete_collection("docs")

    manager._client.delete_collection.assert_not_called()


# --- list_collections ---


async def test_list_collections(manager: QdrantManager) -> None:
    resp = MagicMock()
    resp.collections = [
        _fake_collection("rag_docs"),
        _fake_collection("rag_code"),
        _fake_collection("other_stuff"),
    ]
    manager._client.get_collections = AsyncMock(return_value=resp)

    result = await manager.list_collections()

    assert result == ["rag_docs", "rag_code"]
    assert "other_stuff" not in result


# --- upsert ---


async def test_upsert_single_batch(manager: QdrantManager) -> None:
    points = [
        PointStruct(id=i, vector={"dense": [0.1] * 768}, payload={"text": f"t{i}"})
        for i in range(50)
    ]
    manager._client.upsert = AsyncMock()

    await manager.upsert_points("docs", points, batch_size=100)

    assert manager._client.upsert.call_count == 1
    call_kwargs = manager._client.upsert.call_args.kwargs
    assert call_kwargs["collection_name"] == "rag_docs"
    assert call_kwargs["wait"] is True


async def test_upsert_multiple_batches(manager: QdrantManager) -> None:
    points = [
        PointStruct(id=i, vector={"dense": [0.1] * 768}, payload={"text": f"t{i}"})
        for i in range(250)
    ]
    manager._client.upsert = AsyncMock()

    await manager.upsert_points("docs", points, batch_size=100)

    assert manager._client.upsert.call_count == 3


# --- search_dense ---


async def test_search_dense(manager: QdrantManager) -> None:
    manager._client.query_points = AsyncMock(
        return_value=_fake_query_response([_fake_point("p1", 0.95)])
    )

    result = await manager.search_dense("docs", [0.1] * 768, top_k=5)

    assert result.search_mode == "dense"
    assert result.total == 1
    assert result.results[0].id == "p1"
    assert result.results[0].score == 0.95

    call_kwargs = manager._client.query_points.call_args.kwargs
    assert call_kwargs["using"] == "dense"
    assert call_kwargs["limit"] == 5


async def test_search_dense_with_filter(manager: QdrantManager) -> None:
    manager._client.query_points = AsyncMock(
        return_value=_fake_query_response([_fake_point("p1")])
    )

    await manager.search_dense("docs", [0.1] * 768, filters={"file_type": "markdown"})

    call_kwargs = manager._client.query_points.call_args.kwargs
    qf = call_kwargs["query_filter"]
    assert len(qf.must) == 1
    assert qf.must[0].key == "file_type"


# --- search_sparse ---


async def test_search_sparse(manager: QdrantManager) -> None:
    sparse = SparseVector(indices=[0, 5, 10], values=[1.0, 0.5, 0.3])
    manager._client.query_points = AsyncMock(
        return_value=_fake_query_response([_fake_point("p2", 0.8)])
    )

    result = await manager.search_sparse("docs", sparse, top_k=5)

    assert result.search_mode == "sparse"
    assert result.results[0].id == "p2"
    call_kwargs = manager._client.query_points.call_args.kwargs
    assert call_kwargs["using"] == "sparse"


# --- search_hybrid ---


async def test_search_hybrid_rrf(manager: QdrantManager) -> None:
    sparse = SparseVector(indices=[0, 5], values=[1.0, 0.5])
    dense = [0.1] * 768

    manager._client.query_points = AsyncMock(
        return_value=_fake_query_response([_fake_point("p3", 0.88)])
    )

    result = await manager.search_hybrid("docs", dense, sparse, top_k=5)

    assert result.search_mode == "hybrid"
    assert result.results[0].id == "p3"

    call_kwargs = manager._client.query_points.call_args.kwargs
    assert isinstance(call_kwargs["query"], FusionQuery)
    assert call_kwargs["query"].fusion == Fusion.RRF

    prefetch = call_kwargs["prefetch"]
    assert len(prefetch) == 2
    assert prefetch[0].using == "sparse"
    assert prefetch[1].using == "dense"


async def test_search_hybrid_prefetch_limit(manager: QdrantManager) -> None:
    sparse = SparseVector(indices=[0], values=[1.0])
    dense = [0.1] * 768

    # top_k=5 -> prefetch_limit=10
    manager._client.query_points = AsyncMock(return_value=_fake_query_response([]))

    await manager.search_hybrid("docs", dense, sparse, top_k=5)
    prefetch = manager._client.query_points.call_args.kwargs["prefetch"]
    assert prefetch[0].limit == 10

    # top_k=80 -> prefetch_limit=100 (capped)
    await manager.search_hybrid("docs", dense, sparse, top_k=80)
    prefetch = manager._client.query_points.call_args.kwargs["prefetch"]
    assert prefetch[0].limit == 100


# --- delete_points ---


async def test_delete_points(manager: QdrantManager) -> None:
    manager._client.delete = AsyncMock()

    await manager.delete_points("docs", ["id1", "id2"])

    call_kwargs = manager._client.delete.call_args.kwargs
    assert call_kwargs["collection_name"] == "rag_docs"
    assert call_kwargs["points_selector"].points == ["id1", "id2"]


# --- error wrapping ---


async def test_error_wrapping(manager: QdrantManager) -> None:
    original = RuntimeError("connection refused")
    manager._client.collection_exists = AsyncMock(side_effect=original)

    with pytest.raises(QdrantError) as exc_info:
        await manager.collection_exists("docs")

    assert exc_info.value.cause is original
