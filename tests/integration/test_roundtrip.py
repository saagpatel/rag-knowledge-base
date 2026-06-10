"""Integration tests — require running Ollama and Qdrant services.

Run with: uv run pytest tests/integration/ -v -m integration
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client.models import PointStruct, SparseVector

from rag_kb.core.ollama_client import OllamaClient
from rag_kb.core.qdrant_client import QdrantManager
from tests.integration.conftest import services_available

pytestmark = [pytest.mark.integration, services_available]

TEST_COLLECTION = f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def ollama():
    async with OllamaClient() as client:
        yield client


@pytest.fixture
async def qdrant():
    mgr = QdrantManager()
    yield mgr
    # Cleanup: remove test collection
    try:
        await mgr.delete_collection(TEST_COLLECTION)
    except Exception:
        pass
    await mgr.close()


async def test_health_checks(ollama: OllamaClient) -> None:
    """Verify Ollama is running and has the embedding model."""
    assert await ollama.health() is True


async def test_embed_store_search_roundtrip(
    ollama: OllamaClient, qdrant: QdrantManager
) -> None:
    """End-to-end: embed text -> store in Qdrant -> search -> verify match."""
    # 1. Embed a document
    text = "Retrieval-Augmented Generation combines search with language models."
    vector = await ollama.embed(text)
    assert len(vector) == 768

    # 2. Create collection
    await qdrant.create_collection(TEST_COLLECTION)
    assert await qdrant.collection_exists(TEST_COLLECTION)

    # 3. Upsert point with dense vector + dummy sparse vector
    point = PointStruct(
        id=1,
        vector={
            "dense": vector,
            "sparse": SparseVector(indices=[0, 1, 2], values=[1.0, 0.5, 0.3]),
        },
        payload={"text": text, "file_type": "markdown"},
    )
    await qdrant.upsert_points(TEST_COLLECTION, [point])

    # 4. Search with same vector — should get near-perfect match
    results = await qdrant.search_dense(TEST_COLLECTION, vector, top_k=1)
    assert results.total >= 1
    assert results.results[0].score > 0.99

    # 5. Search with semantically related query
    related_vec = await ollama.embed("How does RAG work with LLMs?")
    related_results = await qdrant.search_dense(TEST_COLLECTION, related_vec, top_k=1)
    assert related_results.total >= 1
    assert related_results.results[0].score > 0.5


async def test_hybrid_search_roundtrip(
    ollama: OllamaClient, qdrant: QdrantManager
) -> None:
    """Embed + upsert + hybrid search returns results."""
    text = "Python is a popular programming language for data science."
    vector = await ollama.embed(text)

    await qdrant.create_collection(TEST_COLLECTION)
    point = PointStruct(
        id=10,
        vector={
            "dense": vector,
            "sparse": SparseVector(indices=[5, 10, 15], values=[0.8, 0.6, 0.4]),
        },
        payload={"content": text, "file_type": "python"},
    )
    await qdrant.upsert_points(TEST_COLLECTION, [point])

    query_vec = await ollama.embed("data science with Python")
    sparse = SparseVector(indices=[5, 10], values=[0.9, 0.5])
    results = await qdrant.search_hybrid(
        TEST_COLLECTION, query_vec, sparse, top_k=5
    )
    assert results.total >= 1
    assert results.results[0].payload["content"] == text


async def test_collection_lifecycle(qdrant: QdrantManager) -> None:
    """Create → list → get info → delete → verify gone."""
    col_name = f"test_lifecycle_{uuid.uuid4().hex[:6]}"

    await qdrant.create_collection(col_name)
    assert await qdrant.collection_exists(col_name)

    names = await qdrant.list_collections()
    prefixed = qdrant._prefixed(col_name)
    assert prefixed in names

    info = await qdrant.get_collection_info(col_name)
    assert info["name"] == prefixed
    assert info["points_count"] == 0

    await qdrant.delete_collection(col_name)
    assert not await qdrant.collection_exists(col_name)


async def test_full_ingest_pipeline(
    ollama: OllamaClient, qdrant: QdrantManager, tmp_path
) -> None:
    """Real file → ingest_file → search returns correct content."""
    from rag_kb.core.database import init_db
    from rag_kb.ingestion.orchestrator import ingest_file
    from rag_kb.models.schema import DocumentStatus

    # Create a test file
    doc = tmp_path / "test_doc.md"
    doc.write_text("# RAG Systems\n\nRetrieval-Augmented Generation is a technique.")

    # Init temp DB
    db = await init_db(db_path=str(tmp_path / "test.db"))
    try:
        result = await ingest_file(doc, TEST_COLLECTION, db, ollama, qdrant)
        assert result.status == DocumentStatus.COMPLETED
        assert result.chunk_count > 0

        # Search for the ingested content
        query_vec = await ollama.embed("RAG technique")
        search_results = await qdrant.search_dense(TEST_COLLECTION, query_vec, top_k=1)
        assert search_results.total >= 1
        assert "RAG" in str(search_results.results[0].payload.get("content", ""))
    finally:
        await db.close()
