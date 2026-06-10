"""Integration tests — /api/health against real services."""

from __future__ import annotations

import pytest

from tests.integration.conftest import services_available

pytestmark = [pytest.mark.integration, services_available]


async def test_health_returns_healthy(http_client):
    """GET /api/health → 200, status=healthy, all services ok."""
    resp = await http_client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["status"] == "healthy"
    assert data["ollama"]["status"] == "ok"
    assert data["qdrant"]["status"] == "ok"
    assert data["sqlite"]["status"] == "ok"


async def test_health_envelope_shape(http_client):
    """Response has success=True, meta.request_id (UUID), meta.timestamp (ISO)."""
    resp = await http_client.get("/api/health")
    body = resp.json()
    assert body["success"] is True
    meta = body["meta"]
    assert "request_id" in meta
    assert len(meta["request_id"]) == 36  # UUID format
    assert "timestamp" in meta
    assert "T" in meta["timestamp"]  # ISO 8601


async def test_health_uptime_positive(http_client):
    """uptime_seconds > 0."""
    resp = await http_client.get("/api/health")
    data = resp.json()["data"]
    assert data["uptime_seconds"] > 0
