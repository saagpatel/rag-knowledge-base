"""Integration tests — /api/search after ingest against real services."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from tests.integration.conftest import services_available

pytestmark = [pytest.mark.integration, services_available]


@pytest_asyncio.fixture(scope="module")
async def ingested_collection(http_client, sample_doc):
    """Ingest a doc once, share the collection across search tests."""
    col = f"search_{uuid.uuid4().hex[:8]}"
    await http_client.post("/api/collections", json={"name": col})
    await http_client.post(
        "/api/ingest",
        json={"path": str(sample_doc), "collection": col},
    )
    yield col
    await http_client.delete(f"/api/collections/{col}")


async def test_search_returns_results(http_client, ingested_collection):
    """POST /api/search → total >= 1, score > 0, content non-empty."""
    resp = await http_client.post(
        "/api/search",
        json={
            "query": "vector database similarity search",
            "collection": ingested_collection,
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total"] >= 1
    assert data["results"][0]["score"] > 0
    assert len(data["results"][0]["content"]) > 0


async def test_search_dense_mode(http_client, ingested_collection):
    """POST /api/search mode=dense → total >= 1."""
    resp = await http_client.post(
        "/api/search",
        json={
            "query": "embeddings for RAG",
            "collection": ingested_collection,
            "mode": "dense",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["total"] >= 1


async def test_search_hybrid_mode(http_client, ingested_collection):
    """POST /api/search mode=hybrid → total >= 1."""
    resp = await http_client.post(
        "/api/search",
        json={
            "query": "nearest neighbor search",
            "collection": ingested_collection,
            "mode": "hybrid",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["total"] >= 1


async def test_search_empty_collection(http_client):
    """Search against empty collection → total=0, results=[]."""
    col = f"empty_{uuid.uuid4().hex[:8]}"
    await http_client.post("/api/collections", json={"name": col})

    resp = await http_client.post(
        "/api/search",
        json={"query": "anything", "collection": col},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total"] == 0
    assert data["results"] == []

    # Cleanup
    await http_client.delete(f"/api/collections/{col}")
