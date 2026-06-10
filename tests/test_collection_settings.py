"""Tests for collection settings (PUT, GET enrichment, ingest defaults)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from rag_kb.api import create_app
from rag_kb.api.schemas import UpdateCollectionRequest


# --- Helpers ---


def _make_qdrant_mock():
    m = AsyncMock()
    m.list_collections = AsyncMock(return_value=["test_col"])
    m.get_collection_info = AsyncMock(return_value={
        "name": "test_col",
        "points_count": 10,
        "vectors_count": 10,
        "status": "green",
    })
    m.create_collection = AsyncMock()
    m.delete_collection = AsyncMock()
    m.collection_exists = AsyncMock(return_value=True)
    m.close = AsyncMock()
    return m


@pytest.fixture
async def test_client(tmp_db):
    """Create an async httpx test client with real SQLite and mocked Qdrant/Ollama."""
    mock_qdrant = _make_qdrant_mock()

    app = create_app()
    app.state.db = tmp_db
    app.state.qdrant = mock_qdrant
    app.state.ollama = AsyncMock()
    app.state.ollama.health = AsyncMock(return_value=True)
    app.state.start_time = time.monotonic()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, mock_qdrant, tmp_db


# --- PUT /api/collections/{name} ---


class TestUpdateCollection:
    async def test_put_updates_settings(self, test_client):
        client, mock_qdrant, db = test_client

        # Create collection in SQLite first
        from cuid2 import cuid_wrapper

        _cuid = cuid_wrapper()
        cid = _cuid()
        await db.execute("INSERT INTO collections (id, name) VALUES (?, ?)", (cid, "test_col"))
        await db.commit()

        resp = await client.put("/api/collections/test_col", json={
            "description": "My docs",
            "chunk_size": 256,
            "chunk_overlap": 25,
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["description"] == "My docs"
        assert data["chunk_size"] == 256
        assert data["chunk_overlap"] == 25

    async def test_put_returns_enriched_info(self, test_client):
        client, mock_qdrant, db = test_client

        resp = await client.put("/api/collections/test_col", json={
            "description": "Test description",
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["name"] == "test_col"
        assert data["points_count"] == 10
        assert data["description"] == "Test description"

    async def test_put_nonexistent_collection_404(self, test_client):
        client, mock_qdrant, db = test_client
        mock_qdrant.collection_exists.return_value = False

        resp = await client.put("/api/collections/missing", json={
            "description": "nope",
        })
        assert resp.status_code == 404

    async def test_partial_update_description_only(self, test_client):
        client, mock_qdrant, db = test_client

        resp = await client.put("/api/collections/test_col", json={
            "description": "Just a description",
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["description"] == "Just a description"


# --- GET enrichment ---


class TestGetCollectionEnriched:
    async def test_get_single_includes_settings(self, test_client):
        client, mock_qdrant, db = test_client

        from cuid2 import cuid_wrapper

        _cuid = cuid_wrapper()
        cid = _cuid()
        await db.execute(
            "INSERT INTO collections (id, name, description, chunk_size) VALUES (?, ?, ?, ?)",
            (cid, "test_col", "My collection", 256),
        )
        await db.commit()

        resp = await client.get("/api/collections/test_col")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["description"] == "My collection"
        assert data["chunk_size"] == 256

    async def test_get_list_includes_settings(self, test_client):
        client, mock_qdrant, db = test_client

        from cuid2 import cuid_wrapper

        _cuid = cuid_wrapper()
        cid = _cuid()
        await db.execute(
            "INSERT INTO collections (id, name, description) VALUES (?, ?, ?)",
            (cid, "test_col", "Listed collection"),
        )
        await db.commit()

        resp = await client.get("/api/collections")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 1
        assert data[0]["description"] == "Listed collection"


# --- POST /collections with settings ---


class TestCreateCollectionWithSettings:
    async def test_post_stores_settings(self, test_client):
        client, mock_qdrant, db = test_client

        mock_qdrant.get_collection_info.return_value = {
            "name": "new_col",
            "points_count": 0,
            "vectors_count": 0,
            "status": "green",
        }

        resp = await client.post("/api/collections", json={
            "name": "new_col",
            "description": "Brand new",
            "chunk_size": 1024,
        })
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert data["description"] == "Brand new"
        assert data["chunk_size"] == 1024


# --- Validation ---


class TestUpdateCollectionValidation:
    def test_chunk_size_too_small(self):
        with pytest.raises(Exception):
            UpdateCollectionRequest(chunk_size=10)

    def test_chunk_size_too_large(self):
        with pytest.raises(Exception):
            UpdateCollectionRequest(chunk_size=5000)

    def test_chunk_overlap_negative(self):
        with pytest.raises(Exception):
            UpdateCollectionRequest(chunk_overlap=-1)

    def test_valid_request(self):
        req = UpdateCollectionRequest(description="ok", chunk_size=256)
        assert req.chunk_size == 256


# --- Ingest uses collection defaults ---


class TestIngestCollectionDefaults:
    async def test_ingest_uses_collection_chunk_size(self, test_client):
        client, mock_qdrant, db = test_client

        from cuid2 import cuid_wrapper

        _cuid = cuid_wrapper()
        cid = _cuid()
        await db.execute(
            "INSERT INTO collections (id, name, chunk_size, chunk_overlap) VALUES (?, ?, ?, ?)",
            (cid, "default", 1024, 100),
        )
        await db.commit()

        from rag_kb.ingestion.orchestrator import IngestionResult
        from rag_kb.models.schema import DocumentStatus

        fake_result = IngestionResult(
            file_path="/tmp/test.md",
            status=DocumentStatus.COMPLETED,
            chunk_count=1,
        )

        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# Test")
            tmp_file = f.name

        try:
            with patch(
                "rag_kb.api.routes.ingest_file",
                new_callable=AsyncMock,
                return_value=fake_result,
            ) as mock_ingest:
                resp = await client.post("/api/ingest", json={
                    "path": tmp_file,
                    "collection": "default",
                })
                assert resp.status_code == 200
                call_kwargs = mock_ingest.call_args.kwargs
                assert call_kwargs["chunk_size"] == 1024
                assert call_kwargs["chunk_overlap"] == 100
        finally:
            Path(tmp_file).unlink(missing_ok=True)
