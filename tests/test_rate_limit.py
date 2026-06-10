"""Tests for the token bucket rate limiter middleware."""

from __future__ import annotations

import time
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from rag_kb.api import create_app
from rag_kb.api.rate_limit import TokenBucket
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


def _patch_config(rate_limit_rpm: int = 60, rate_limit_burst: int = 10, api_key: str = ""):
    cfg = AppConfig(server=ServerConfig(api_key=api_key, rate_limit_rpm=rate_limit_rpm, rate_limit_burst=rate_limit_burst))
    stack = ExitStack()
    for module in ("rag_kb.api.auth", "rag_kb.api.rate_limit", "rag_kb.core.config"):
        stack.enter_context(patch(f"{module}.get_config", return_value=cfg))
    return stack


class TestTokenBucket:
    def test_initial_burst(self):
        """TokenBucket starts with burst tokens available."""
        bucket = TokenBucket(rate=1.0, burst=5)
        for _ in range(5):
            assert bucket.consume() is True
        assert bucket.consume() is False

    def test_refill_over_time(self):
        """Tokens refill over time."""
        bucket = TokenBucket(rate=10.0, burst=5)
        for _ in range(5):
            bucket.consume()
        assert bucket.consume() is False

        bucket.last_refill = time.monotonic() - 1.0
        assert bucket.consume() is True

    def test_burst_cap(self):
        """Tokens cannot exceed burst limit even after long idle."""
        bucket = TokenBucket(rate=10.0, burst=3)
        bucket.last_refill = time.monotonic() - 100.0
        for _ in range(3):
            assert bucket.consume() is True
        assert bucket.consume() is False

    def test_retry_after_property(self):
        """retry_after returns time until next token."""
        bucket = TokenBucket(rate=1.0, burst=2)
        bucket.consume()
        bucket.consume()
        assert bucket.consume() is False
        assert bucket.retry_after > 0
        assert bucket.retry_after <= 1.0


@pytest.mark.asyncio
async def test_rate_limit_disabled_when_rpm_zero(client, app):
    """When rate_limit_rpm=0, rate limiting is disabled."""
    app.state.qdrant.list_collections.return_value = []

    with _patch_config(rate_limit_rpm=0):
        for _ in range(20):
            resp = await client.get("/api/collections")
            assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_exempt_from_rate_limit(client, app):
    """Health endpoint is exempt from rate limiting."""
    app.state.ollama.health.return_value = True
    app.state.qdrant.list_collections.return_value = []

    with _patch_config(rate_limit_rpm=60, rate_limit_burst=1):
        resp = await client.get("/api/collections")
        assert resp.status_code == 200
        for _ in range(5):
            resp = await client.get("/api/health")
            assert resp.status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_enforced_after_burst(client, app):
    """After burst is exhausted, requests are rejected."""
    app.state.qdrant.list_collections.return_value = []

    with _patch_config(rate_limit_rpm=60, rate_limit_burst=2):
        for _ in range(2):
            resp = await client.get("/api/collections")
            assert resp.status_code == 200

        resp = await client.get("/api/collections")
        assert resp.status_code == 429


@pytest.mark.asyncio
async def test_429_has_retry_after_header(client, app):
    """429 response includes Retry-After header."""
    app.state.qdrant.list_collections.return_value = []

    with _patch_config(rate_limit_rpm=60, rate_limit_burst=1):
        await client.get("/api/collections")
        resp = await client.get("/api/collections")
        assert resp.status_code == 429
        assert "retry-after" in resp.headers


@pytest.mark.asyncio
async def test_429_has_error_envelope(client, app):
    """429 response contains standard error envelope."""
    app.state.qdrant.list_collections.return_value = []

    with _patch_config(rate_limit_rpm=60, rate_limit_burst=1):
        await client.get("/api/collections")
        resp = await client.get("/api/collections")
        assert resp.status_code == 429
        body = resp.json()
        assert body["success"] is False
        assert body["error"]["code"] == "RATE_LIMITED"
        assert body["error"]["statusCode"] == 429
        assert "meta" in body
        assert "request_id" in body["meta"]
        assert "timestamp" in body["meta"]
