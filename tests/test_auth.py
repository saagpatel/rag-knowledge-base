"""Tests for the API key authentication middleware."""

from __future__ import annotations

import time
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from rag_kb.api import create_app
from rag_kb.core.config import AppConfig, ServerConfig


def _make_db_execute_default():
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


def _patch_config(api_key: str = "", rate_limit_rpm: int = 0):
    """Patch get_config everywhere it's imported."""
    cfg = AppConfig(server=ServerConfig(api_key=api_key, rate_limit_rpm=rate_limit_rpm))
    stack = ExitStack()
    for module in ("rag_kb.api.auth", "rag_kb.api.rate_limit", "rag_kb.core.config"):
        stack.enter_context(patch(f"{module}.get_config", return_value=cfg))
    return stack


@pytest.mark.asyncio
async def test_auth_disabled_passes_through(client, app):
    """When api_key is empty, requests pass through without auth."""
    app.state.ollama.health.return_value = True
    app.state.qdrant.list_collections.return_value = []

    with _patch_config(api_key=""):
        resp = await client.get("/api/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_missing_auth_header_returns_401(client, app):
    """Missing Authorization header returns 401."""
    with _patch_config(api_key="test-secret-key"):
        resp = await client.get("/api/collections")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_wrong_key_returns_401(client, app):
    """Wrong API key returns 401."""
    with _patch_config(api_key="correct-key"):
        resp = await client.get(
            "/api/collections",
            headers={"Authorization": "Bearer wrong-key"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_correct_key_passes(client, app):
    """Correct API key allows access."""
    app.state.qdrant.list_collections.return_value = []

    with _patch_config(api_key="my-secret"):
        resp = await client.get(
            "/api/collections",
            headers={"Authorization": "Bearer my-secret"},
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_exempt(client, app):
    """Health endpoint is exempt from auth."""
    app.state.ollama.health.return_value = True
    app.state.qdrant.list_collections.return_value = []

    with _patch_config(api_key="secret"):
        resp = await client.get("/api/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_docs_exempt(client, app):
    """Docs endpoint is exempt from auth."""
    with _patch_config(api_key="secret"):
        resp = await client.get("/api/docs")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_openapi_exempt(client, app):
    """OpenAPI JSON endpoint is exempt from auth."""
    with _patch_config(api_key="secret"):
        resp = await client.get("/api/openapi.json")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_bearer_prefix_required(client, app):
    """Basic auth prefix is rejected — only Bearer works."""
    with _patch_config(api_key="my-key"):
        resp = await client.get(
            "/api/collections",
            headers={"Authorization": "Basic my-key"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_401_has_error_envelope(client, app):
    """401 response contains the standard error envelope."""
    with _patch_config(api_key="secret"):
        resp = await client.get("/api/collections")
    assert resp.status_code == 401
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "UNAUTHORIZED"
    assert body["error"]["statusCode"] == 401
    assert "meta" in body
    assert "request_id" in body["meta"]
    assert "timestamp" in body["meta"]


@pytest.mark.asyncio
async def test_auth_disabled_non_health_works(client, app):
    """When auth is disabled, non-health endpoints work without header."""
    app.state.qdrant.list_collections.return_value = []

    with _patch_config(api_key=""):
        resp = await client.get("/api/collections")
    assert resp.status_code == 200
