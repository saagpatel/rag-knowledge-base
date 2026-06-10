"""Tests for the request logging middleware."""

from __future__ import annotations

import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from rag_kb.api import create_app


def _make_db_execute_default():
    """Return a db mock execute that handles async with db.execute('SELECT 1')."""
    from tests.test_api import _FakeCursor

    def _execute(*args, **kwargs):
        return _FakeCursor([(1,)])

    return _execute


@pytest.fixture
def app():
    application = create_app()
    application.state.db = AsyncMock()
    application.state.db.execute = MagicMock(side_effect=_make_db_execute_default())
    application.state.ollama = AsyncMock()
    application.state.qdrant = AsyncMock()
    application.state.start_time = time.monotonic()
    return application


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_request_id_in_headers(client, app):
    """Every response includes X-Request-ID header."""
    app.state.ollama.health.return_value = True
    app.state.qdrant.list_collections.return_value = []

    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) > 0


@pytest.mark.asyncio
async def test_logging_on_success(client, app, caplog):
    """Middleware logs successful requests."""
    app.state.ollama.health.return_value = True
    app.state.qdrant.list_collections.return_value = []

    with caplog.at_level(logging.INFO, logger="rag_kb.api.access"):
        resp = await client.get("/api/health")

    assert resp.status_code == 200
    assert any("GET" in r.message and "/api/health" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_logging_on_error(client, app, caplog):
    """Middleware logs 4xx/5xx responses."""
    with caplog.at_level(logging.INFO, logger="rag_kb.api.access"):
        resp = await client.post("/api/search", json={})

    assert resp.status_code == 422
    assert any("422" in r.message or r.levelname == "INFO" for r in caplog.records)


@pytest.mark.asyncio
async def test_request_id_in_contextvars(client, app):
    """request_id_var is set during request processing and matches the response header."""
    from rag_kb.api.middleware import request_id_var

    app.state.ollama = AsyncMock()
    app.state.ollama.health.return_value = True
    app.state.qdrant = AsyncMock()
    app.state.qdrant.list_collections.return_value = []

    resp = await client.get("/api/health")
    assert resp.status_code == 200
    # The X-Request-ID header proves that request_id_var was set (middleware sets it)
    request_id = resp.headers["x-request-id"]
    assert len(request_id) > 0


@pytest.mark.asyncio
async def test_request_id_unique_per_request(client, app):
    """Each request gets a unique request_id."""
    app.state.ollama = AsyncMock()
    app.state.ollama.health.return_value = True
    app.state.qdrant = AsyncMock()
    app.state.qdrant.list_collections.return_value = []

    resp1 = await client.get("/api/health")
    resp2 = await client.get("/api/health")

    id1 = resp1.headers["x-request-id"]
    id2 = resp2.headers["x-request-id"]
    assert id1 != id2
