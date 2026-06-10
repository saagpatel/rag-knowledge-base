"""Integration tests — /api/ingest round-trip against real services."""

from __future__ import annotations

import uuid

import pytest

from tests.integration.conftest import services_available

pytestmark = [pytest.mark.integration, services_available]


def _unique_name() -> str:
    return f"integ_{uuid.uuid4().hex[:8]}"


async def test_ingest_single_file(http_client, sample_doc):
    """POST /api/ingest with sample Markdown → processed=1, chunk_count > 0."""
    col = _unique_name()
    await http_client.post("/api/collections", json={"name": col})

    resp = await http_client.post(
        "/api/ingest",
        json={"path": str(sample_doc), "collection": col},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["processed"] == 1
    assert data["results"][0]["chunk_count"] > 0

    # Cleanup
    await http_client.delete(f"/api/collections/{col}")


async def test_ingest_updates_collection(http_client, sample_doc):
    """After ingest, GET /api/collections/{name} → points_count > 0."""
    col = _unique_name()
    await http_client.post("/api/collections", json={"name": col})
    await http_client.post(
        "/api/ingest",
        json={"path": str(sample_doc), "collection": col},
    )

    resp = await http_client.get(f"/api/collections/{col}")
    assert resp.status_code == 200
    assert resp.json()["data"]["points_count"] > 0

    # Cleanup
    await http_client.delete(f"/api/collections/{col}")


async def test_ingest_idempotent(http_client, sample_doc):
    """Ingest same file twice → second call has skipped=1 (hash dedup)."""
    col = _unique_name()
    await http_client.post("/api/collections", json={"name": col})

    # First ingest
    resp1 = await http_client.post(
        "/api/ingest",
        json={"path": str(sample_doc), "collection": col},
    )
    assert resp1.json()["data"]["processed"] == 1

    # Second ingest — same file, same hash
    resp2 = await http_client.post(
        "/api/ingest",
        json={"path": str(sample_doc), "collection": col},
    )
    data2 = resp2.json()["data"]
    assert data2["skipped"] == 1

    # Cleanup
    await http_client.delete(f"/api/collections/{col}")


async def test_ingest_nonexistent_path(http_client):
    """POST /api/ingest with bad path → 422 with error."""
    col = _unique_name()
    await http_client.post("/api/collections", json={"name": col})

    resp = await http_client.post(
        "/api/ingest",
        json={"path": "/nonexistent/fake_file.md", "collection": col},
    )
    assert resp.status_code == 422

    # Cleanup
    await http_client.delete(f"/api/collections/{col}")


async def test_ingest_directory(http_client, tmp_path):
    """Create 3 .md files → POST /api/ingest with dir → total_files=3."""
    col = _unique_name()
    await http_client.post("/api/collections", json={"name": col})

    # Create 3 markdown files
    for i in range(3):
        f = tmp_path / f"doc_{i}.md"
        f.write_text(f"# Document {i}\n\nThis is test document number {i}.")

    resp = await http_client.post(
        "/api/ingest",
        json={"path": str(tmp_path), "collection": col},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total_files"] == 3

    # Cleanup
    await http_client.delete(f"/api/collections/{col}")
