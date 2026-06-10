"""Integration tests — /api/stats, /api/metrics, /api/queries analytics endpoints."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from tests.integration.conftest import services_available

pytestmark = [pytest.mark.integration, services_available]


def _unique_name() -> str:
    return f"analytics_{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture(scope="module")
async def analytics_collection(http_client, sample_doc):
    """Ingest + search to generate query history for analytics tests."""
    col = _unique_name()
    await http_client.post("/api/collections", json={"name": col})
    await http_client.post(
        "/api/ingest",
        json={"path": str(sample_doc), "collection": col},
    )
    # Perform a search so queries table has data
    await http_client.post(
        "/api/search",
        json={"query": "vector database", "collection": col},
    )
    # Perform an ask so we have multiple query types
    await http_client.post(
        "/api/ask",
        json={"query": "What are vector databases?", "collection": col},
    )
    yield col
    await http_client.delete(f"/api/collections/{col}")


async def test_stats_returns_aggregate_data(http_client, analytics_collection):
    """GET /api/stats → 200, includes total_queries, latency percentiles."""
    resp = await http_client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total_queries"] >= 2  # at least our search + ask
    assert "latency_p50" in data
    assert "latency_p95" in data
    assert "latency_p99" in data
    assert "queries_by_interface" in data
    assert "queries_by_type" in data


async def test_stats_with_days_filter(http_client, analytics_collection):
    """GET /api/stats?days=7 → scoped results."""
    resp = await http_client.get("/api/stats?days=7")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["period_days"] == 7
    assert data["total_queries"] >= 0


async def test_stats_includes_top_collections(http_client, analytics_collection):
    """GET /api/stats → top_collections list."""
    resp = await http_client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "top_collections" in data
    assert isinstance(data["top_collections"], list)


async def test_metrics_returns_performance_data(http_client, analytics_collection):
    """GET /api/metrics → 200, includes latency percentiles + cache stats."""
    resp = await http_client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "total_queries" in data
    assert "latency_p50" in data
    assert "cache_hit_rate" in data
    assert "cache_size" in data
    assert "active_jobs" in data


async def test_queries_returns_history(http_client, analytics_collection):
    """GET /api/queries → 200, returns list with pagination."""
    resp = await http_client.get("/api/queries")
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert "queries" in body
    assert "total" in body
    assert body["total"] >= 2
    assert len(body["queries"]) >= 2

    # Verify query record structure
    q = body["queries"][0]
    assert "query_text" in q
    assert "query_type" in q
    assert "search_mode" in q
    assert "latency_ms" in q
    assert "interface" in q


async def test_queries_filter_by_interface(http_client, analytics_collection):
    """GET /api/queries?interface=api → filtered results."""
    resp = await http_client.get("/api/queries?interface=api")
    assert resp.status_code == 200
    body = resp.json()["data"]
    for q in body["queries"]:
        assert q["interface"] == "api"


async def test_queries_filter_by_type(http_client, analytics_collection):
    """GET /api/queries?query_type=search → only search queries."""
    resp = await http_client.get("/api/queries?query_type=search")
    assert resp.status_code == 200
    body = resp.json()["data"]
    for q in body["queries"]:
        assert q["query_type"] == "search"


async def test_query_appears_after_search(http_client):
    """Perform a search, then verify it appears in query history."""
    col = _unique_name()
    await http_client.post("/api/collections", json={"name": col})

    unique_query = f"test_tracking_{uuid.uuid4().hex[:8]}"
    await http_client.post(
        "/api/search",
        json={"query": unique_query, "collection": col},
    )

    resp = await http_client.get("/api/queries?limit=5")
    assert resp.status_code == 200
    queries = resp.json()["data"]["queries"]
    found = any(q["query_text"] == unique_query for q in queries)
    assert found, f"Query '{unique_query}' not found in recent history"

    # Cleanup
    await http_client.delete(f"/api/collections/{col}")
