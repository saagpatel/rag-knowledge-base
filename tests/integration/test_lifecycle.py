"""Integration tests — full lifecycle: create → ingest → search → ask → stats → delete."""

from __future__ import annotations

import uuid

import pytest

from tests.integration.conftest import services_available

pytestmark = [pytest.mark.integration, services_available]


def _unique_name() -> str:
    return f"lifecycle_{uuid.uuid4().hex[:8]}"


async def test_full_lifecycle(http_client, tmp_path):
    """Full workflow: create → ingest → search → ask → check stats → delete doc → delete collection."""
    col = _unique_name()

    # 1. Create collection
    resp = await http_client.post("/api/collections", json={"name": col})
    assert resp.status_code == 201

    # 2. Ingest a file
    doc = tmp_path / "lifecycle.md"
    doc.write_text(
        "# Lifecycle Test\n\n"
        "This document tests the full lifecycle of the RAG knowledge base.\n"
        "It covers creation, ingestion, search, Q&A, and deletion.\n"
    )
    resp = await http_client.post(
        "/api/ingest",
        json={"path": str(doc), "collection": col},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["processed"] == 1

    # 3. Search for content
    resp = await http_client.post(
        "/api/search",
        json={"query": "lifecycle RAG knowledge base", "collection": col},
    )
    assert resp.status_code == 200
    search_data = resp.json()["data"]
    assert search_data["total"] >= 1

    # 4. Ask a question
    resp = await http_client.post(
        "/api/ask",
        json={"query": "What does this document cover?", "collection": col},
    )
    assert resp.status_code == 200
    assert len(resp.json()["data"]["answer"]) > 0

    # 5. Check stats reflect our queries
    resp = await http_client.get("/api/stats")
    assert resp.status_code == 200
    assert resp.json()["data"]["total_queries"] >= 2

    # 6. Get document list and delete the doc
    resp = await http_client.get(f"/api/documents?collection={col}")
    assert resp.status_code == 200
    docs = resp.json()["data"]["documents"]
    assert len(docs) >= 1
    doc_id = docs[0]["id"]

    resp = await http_client.delete(f"/api/documents/{doc_id}")
    assert resp.status_code == 200

    # Verify doc is gone
    resp = await http_client.get(f"/api/documents/{doc_id}")
    assert resp.status_code == 404

    # 7. Delete collection
    resp = await http_client.delete(f"/api/collections/{col}")
    assert resp.status_code == 200

    # Verify collection is gone
    resp = await http_client.get(f"/api/collections/{col}")
    assert resp.status_code == 404


async def test_zero_leakage_after_delete(http_client, tmp_path):
    """After deleting collection, Qdrant has no points and SQLite has no docs."""
    col = _unique_name()
    await http_client.post("/api/collections", json={"name": col})

    doc = tmp_path / "leakage.md"
    doc.write_text("# Leakage Test\n\nContent that should be fully removed.\n")
    await http_client.post(
        "/api/ingest",
        json={"path": str(doc), "collection": col},
    )

    # Verify data exists
    resp = await http_client.get(f"/api/collections/{col}")
    assert resp.json()["data"]["points_count"] > 0

    resp = await http_client.get(f"/api/documents?collection={col}")
    assert resp.json()["data"]["total"] >= 1

    # Delete collection
    await http_client.delete(f"/api/collections/{col}")

    # Collection gone from Qdrant
    resp = await http_client.get(f"/api/collections/{col}")
    assert resp.status_code == 404

    # Documents endpoint should 404 for that collection too
    resp = await http_client.get(f"/api/documents?collection={col}")
    assert resp.status_code == 404


async def test_multi_collection_isolation(http_client, tmp_path):
    """Ingest to A and B — search A doesn't return B results."""
    col_a = _unique_name()
    col_b = _unique_name()
    await http_client.post("/api/collections", json={"name": col_a})
    await http_client.post("/api/collections", json={"name": col_b})

    # Ingest unique content to each
    doc_a = tmp_path / "isolation_a.md"
    doc_a.write_text("# Alpha Centauri\n\nAlpha Centauri is the nearest star system.\n")

    doc_b = tmp_path / "isolation_b.md"
    doc_b.write_text("# Mariana Trench\n\nThe Mariana Trench is the deepest ocean trench.\n")

    await http_client.post(
        "/api/ingest",
        json={"path": str(doc_a), "collection": col_a},
    )
    await http_client.post(
        "/api/ingest",
        json={"path": str(doc_b), "collection": col_b},
    )

    # Search A for B's content → should get 0 relevant results
    resp = await http_client.post(
        "/api/search",
        json={"query": "Mariana Trench deepest ocean", "collection": col_a},
    )
    assert resp.status_code == 200
    results_a = resp.json()["data"]["results"]
    # Results should not contain Mariana Trench content
    for r in results_a:
        assert "Mariana" not in r["content"], "Collection A leaked content from B"

    # Search B for A's content → should get 0 relevant results
    resp = await http_client.post(
        "/api/search",
        json={"query": "Alpha Centauri nearest star", "collection": col_b},
    )
    assert resp.status_code == 200
    results_b = resp.json()["data"]["results"]
    for r in results_b:
        assert "Alpha Centauri" not in r["content"], "Collection B leaked content from A"

    # Cleanup
    await http_client.delete(f"/api/collections/{col_a}")
    await http_client.delete(f"/api/collections/{col_b}")
