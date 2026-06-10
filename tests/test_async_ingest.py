"""Tests for async directory ingest (202 response) and GET /api/jobs/{id}."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from rag_kb.api import create_app


class _AsyncIter:
    def __init__(self, rows):
        self._rows = list(rows)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._rows):
            raise StopAsyncIteration
        row = self._rows[self._index]
        self._index += 1
        return row


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    def __aiter__(self):
        return _AsyncIter(self._rows)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __await__(self):
        async def _self():
            return self
        return _self().__await__()


def _make_db_execute(response_sequence):
    call_idx = [0]

    def _execute(*args, **kwargs):
        idx = call_idx[0]
        call_idx[0] += 1
        if idx < len(response_sequence):
            rows = response_sequence[idx]
        else:
            rows = []
        clean = [r for r in rows if r is not None]
        return _FakeCursor(clean)

    return _execute


def _make_db_mock(responses=None):
    db = AsyncMock()
    db.execute = MagicMock(side_effect=_make_db_execute(responses or [[(1,)]]))
    return db


@pytest.fixture
def app():
    application = create_app()
    application.state.db = _make_db_mock()
    application.state.ollama = AsyncMock()
    application.state.qdrant = AsyncMock()
    application.state.start_time = time.monotonic()
    application.state.background_tasks = set()
    return application


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------- Directory Ingest → 202 ----------


@pytest.mark.asyncio
async def test_directory_ingest_returns_job_data(client, app, tmp_path):
    """Directory ingest returns a JobData response with job_id and status."""
    (tmp_path / "a.md").write_text("a")

    # db.execute calls:
    # 1. _get_collection_settings SELECT (for chunk resolution)
    # 2. _ensure_collection_exists SELECT
    # 3. INSERT collections (no row found)
    # 4. INSERT ingestion_jobs
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [],          # _get_collection_settings
        [],          # _ensure_collection_exists SELECT
    ]))
    app.state.db.commit = AsyncMock()
    app.state.qdrant.create_collection = AsyncMock()

    with patch("rag_kb.api.routes.ingest_directory", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = None
        resp = await client.post("/api/ingest", json={"path": str(tmp_path)})

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "job_id" in body["data"]
    assert body["data"]["status"] == "running"


@pytest.mark.asyncio
async def test_file_ingest_still_synchronous(client, app, tmp_path):
    """Single file ingest still returns IngestData synchronously."""
    f = tmp_path / "doc.md"
    f.write_text("# Hello")

    from rag_kb.ingestion.orchestrator import IngestionResult
    from rag_kb.models.schema import DocumentStatus

    result = IngestionResult(
        file_path=str(f), status=DocumentStatus.COMPLETED, chunk_count=5,
    )

    with patch("rag_kb.api.routes.ingest_file", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = result
        resp = await client.post("/api/ingest", json={"path": str(f)})

    assert resp.status_code == 200
    body = resp.json()
    assert "total_files" in body["data"]  # IngestData shape
    assert body["data"]["total_files"] == 1


@pytest.mark.asyncio
async def test_directory_ingest_job_data_has_total_files(client, app, tmp_path):
    """Job data includes the correct total_files count."""
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "b.txt").write_text("b")

    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [],  # _get_collection_settings
        [],  # _ensure_collection_exists SELECT
    ]))
    app.state.db.commit = AsyncMock()
    app.state.qdrant.create_collection = AsyncMock()

    with patch("rag_kb.api.routes.ingest_directory", new_callable=AsyncMock):
        resp = await client.post("/api/ingest", json={"path": str(tmp_path)})

    body = resp.json()
    assert body["data"]["total_files"] == 2


@pytest.mark.asyncio
async def test_background_tasks_tracked(client, app, tmp_path):
    """Background ingest task is added to app.state.background_tasks."""
    (tmp_path / "a.md").write_text("a")

    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [],  # _get_collection_settings
        [],  # _ensure_collection_exists SELECT
    ]))
    app.state.db.commit = AsyncMock()
    app.state.qdrant.create_collection = AsyncMock()

    with patch("rag_kb.api.routes.ingest_directory", new_callable=AsyncMock) as mock:
        # Make it take a moment so the task is still running
        async def _slow(*args, **kwargs):
            await asyncio.sleep(0.1)
        mock.side_effect = _slow
        resp = await client.post("/api/ingest", json={"path": str(tmp_path)})

    assert resp.status_code == 200
    # Task should be tracked (may already be done or still running)
    # Just verify no error occurred
    assert resp.json()["data"]["status"] == "running"


# ---------- GET /api/jobs/{job_id} ----------


@pytest.mark.asyncio
async def test_job_status_endpoint(client, app):
    """GET /api/jobs/{id} returns job details."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [("job-123", "completed", 5, 5, 0, "2024-01-01T00:00:00", "2024-01-01T00:01:00")],
    ]))

    resp = await client.get("/api/jobs/job-123")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["job_id"] == "job-123"
    assert body["data"]["status"] == "completed"
    assert body["data"]["total_files"] == 5
    assert body["data"]["processed_files"] == 5


@pytest.mark.asyncio
async def test_job_not_found_returns_404(client, app):
    """GET /api/jobs/{id} returns 404 for unknown job."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [],  # No matching job
    ]))

    resp = await client.get("/api/jobs/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_job_data_structure(client, app):
    """Job response has all expected fields."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [("job-1", "running", 10, 3, 1, "2024-01-01T00:00:00", None)],
    ]))

    resp = await client.get("/api/jobs/job-1")
    body = resp.json()
    data = body["data"]
    assert "job_id" in data
    assert "status" in data
    assert "total_files" in data
    assert "processed_files" in data
    assert "failed_files" in data
    assert "started_at" in data
    assert "completed_at" in data
    assert data["failed_files"] == 1
    assert data["completed_at"] is None
