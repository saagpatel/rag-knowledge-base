"""Integration tests — /api/collections CRUD against real Qdrant."""

from __future__ import annotations

import uuid

import pytest

from tests.integration.conftest import services_available

pytestmark = [pytest.mark.integration, services_available]


def _unique_name() -> str:
    return f"integ_{uuid.uuid4().hex[:8]}"


async def test_create_collection(http_client):
    """POST /api/collections → 201, name matches."""
    name = _unique_name()
    resp = await http_client.post("/api/collections", json={"name": name})
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert name in body["data"]["name"]

    # Cleanup
    await http_client.delete(f"/api/collections/{name}")


async def test_create_appears_in_list(http_client):
    """POST create → GET /api/collections → name in list."""
    name = _unique_name()
    await http_client.post("/api/collections", json={"name": name})

    resp = await http_client.get("/api/collections")
    assert resp.status_code == 200
    names = [c["name"] for c in resp.json()["data"]]
    assert any(name in n for n in names)

    # Cleanup
    await http_client.delete(f"/api/collections/{name}")


async def test_get_collection_by_name(http_client):
    """GET /api/collections/{name} → 200, points_count=0."""
    name = _unique_name()
    await http_client.post("/api/collections", json={"name": name})

    resp = await http_client.get(f"/api/collections/{name}")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["points_count"] == 0

    # Cleanup
    await http_client.delete(f"/api/collections/{name}")


async def test_get_not_found_returns_404(http_client):
    """GET /api/collections/nonexistent → 404, COLLECTION_NOT_FOUND."""
    resp = await http_client.get("/api/collections/nonexistent_xyz_999")
    assert resp.status_code == 404
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "COLLECTION_NOT_FOUND"


async def test_delete_removes_collection(http_client):
    """POST create → DELETE → GET → 404."""
    name = _unique_name()
    await http_client.post("/api/collections", json={"name": name})

    del_resp = await http_client.delete(f"/api/collections/{name}")
    assert del_resp.status_code == 200

    get_resp = await http_client.get(f"/api/collections/{name}")
    assert get_resp.status_code == 404


# --- PUT collection settings ---


async def test_update_collection_settings(http_client):
    """PUT /api/collections/{name} with chunk_size + description → 200."""
    name = _unique_name()
    await http_client.post("/api/collections", json={"name": name})

    resp = await http_client.put(
        f"/api/collections/{name}",
        json={"description": "Updated description", "chunk_size": 256},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["description"] == "Updated description"
    assert data["chunk_size"] == 256

    # Cleanup
    await http_client.delete(f"/api/collections/{name}")


async def test_update_collection_reflects_in_get(http_client):
    """PUT → GET → settings match."""
    name = _unique_name()
    await http_client.post("/api/collections", json={"name": name})

    await http_client.put(
        f"/api/collections/{name}",
        json={"description": "Test desc", "chunk_overlap": 75},
    )

    resp = await http_client.get(f"/api/collections/{name}")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["description"] == "Test desc"
    assert data["chunk_overlap"] == 75

    # Cleanup
    await http_client.delete(f"/api/collections/{name}")


async def test_update_nonexistent_collection(http_client):
    """PUT /api/collections/nonexistent → 404."""
    resp = await http_client.put(
        "/api/collections/nonexistent_xyz_999",
        json={"description": "Should fail"},
    )
    assert resp.status_code == 404
