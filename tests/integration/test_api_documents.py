"""Integration tests — /api/documents list/get/delete against real services."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from tests.integration.conftest import services_available

pytestmark = [pytest.mark.integration, services_available]


@pytest_asyncio.fixture
async def doc_collection(http_client, sample_doc):
    """Create collection, ingest a doc, yield (collection_name, doc_path)."""
    col = f"docs_{uuid.uuid4().hex[:8]}"
    await http_client.post("/api/collections", json={"name": col})
    await http_client.post(
        "/api/ingest",
        json={"path": str(sample_doc), "collection": col},
    )
    yield col
    try:
        await http_client.delete(f"/api/collections/{col}")
    except Exception:
        pass


async def test_list_documents_after_ingest(http_client, doc_collection):
    """GET /api/documents?collection=... → total=1, filename matches."""
    resp = await http_client.get(
        "/api/documents", params={"collection": doc_collection}
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total"] == 1
    assert "sample_doc" in data["documents"][0]["filename"]


async def test_get_document_by_id(http_client, doc_collection):
    """List → extract id → GET /api/documents/{id} → 200, chunk_count > 0."""
    list_resp = await http_client.get(
        "/api/documents", params={"collection": doc_collection}
    )
    doc_id = list_resp.json()["data"]["documents"][0]["id"]

    resp = await http_client.get(f"/api/documents/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["data"]["chunk_count"] > 0


async def test_delete_document(http_client, doc_collection):
    """List → delete → list again → total=0."""
    list_resp = await http_client.get(
        "/api/documents", params={"collection": doc_collection}
    )
    doc_id = list_resp.json()["data"]["documents"][0]["id"]

    del_resp = await http_client.delete(f"/api/documents/{doc_id}")
    assert del_resp.status_code == 200

    list_resp2 = await http_client.get(
        "/api/documents", params={"collection": doc_collection}
    )
    assert list_resp2.json()["data"]["total"] == 0


async def test_get_not_found(http_client):
    """GET /api/documents/fake-id → 404."""
    resp = await http_client.get("/api/documents/fake-id-does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"
