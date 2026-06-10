"""Integration tests — /api/ask Q&A endpoint against real services."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from tests.integration.conftest import services_available

pytestmark = [pytest.mark.integration, services_available]


def _unique_name() -> str:
    return f"ask_{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture(scope="module")
async def ask_collection(http_client, sample_doc):
    """Ingest a doc once, share the collection across ask tests."""
    col = _unique_name()
    await http_client.post("/api/collections", json={"name": col})
    await http_client.post(
        "/api/ingest",
        json={"path": str(sample_doc), "collection": col},
    )
    yield col
    await http_client.delete(f"/api/collections/{col}")


async def test_ask_returns_answer(http_client, ask_collection):
    """POST /api/ask with valid query → 200, answer + sources."""
    resp = await http_client.post(
        "/api/ask",
        json={
            "query": "What are vector databases used for?",
            "collection": ask_collection,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert len(data["answer"]) > 0
    assert data["context_chunks_used"] > 0
    assert data["latency_ms"] > 0


async def test_ask_includes_sources(http_client, ask_collection):
    """POST /api/ask → sources reference chunks from ingested documents."""
    resp = await http_client.post(
        "/api/ask",
        json={
            "query": "What features do vector databases have?",
            "collection": ask_collection,
        },
    )
    assert resp.status_code == 200
    sources = resp.json()["data"]["sources"]
    assert len(sources) > 0
    for src in sources:
        assert "file_path" in src
        assert src["score"] > 0
        assert "file_type" in src


async def test_ask_default_collection(http_client):
    """POST /api/ask with no collection → uses 'default' (may return empty answer)."""
    resp = await http_client.post(
        "/api/ask",
        json={"query": "What is RAG?"},
    )
    # Should not error — either answers or returns empty context
    assert resp.status_code == 200
    assert resp.json()["success"] is True


async def test_ask_nonexistent_collection(http_client):
    """POST /api/ask with bad collection → 404."""
    resp = await http_client.post(
        "/api/ask",
        json={
            "query": "anything",
            "collection": "nonexistent_xyz_999",
        },
    )
    assert resp.status_code == 404


async def test_ask_streaming_sse(http_client, ask_collection):
    """POST /api/ask with stream=true → SSE text/event-stream with [DONE]."""
    resp = await http_client.post(
        "/api/ask",
        json={
            "query": "Explain vector similarity search",
            "collection": ask_collection,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    # Parse SSE data lines
    text = resp.text
    lines = [l for l in text.split("\n") if l.startswith("data: ")]
    assert len(lines) >= 2  # at least one token + [DONE]
    assert lines[-1] == "data: [DONE]"


async def test_ask_with_model_override(http_client, ask_collection):
    """POST /api/ask with explicit model → response uses that model."""
    resp = await http_client.post(
        "/api/ask",
        json={
            "query": "What is similarity search?",
            "collection": ask_collection,
            "model": "mistral:7b",
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["model"] == "mistral:7b"
