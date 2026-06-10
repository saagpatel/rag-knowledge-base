"""Tests for the metrics endpoint and percentile calculations."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from rag_kb.api import create_app
from rag_kb.api.routes import _percentile
from rag_kb.core.cache import EmbeddingCache


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


# ---------- _percentile ----------


class TestPercentile:
    def test_empty_list(self) -> None:
        assert _percentile([], 50) == 0.0

    def test_single_element(self) -> None:
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 99) == 42.0

    def test_two_elements(self) -> None:
        result = _percentile([10.0, 20.0], 50)
        assert result == pytest.approx(15.0)

    def test_p50_of_sorted_list(self) -> None:
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert _percentile(data, 50) == pytest.approx(30.0)

    def test_p95_of_sorted_list(self) -> None:
        data = list(range(1, 101))
        data_float = [float(x) for x in data]
        p95 = _percentile(data_float, 95)
        assert p95 == pytest.approx(95.05, abs=0.1)

    def test_p99_of_sorted_list(self) -> None:
        data = list(range(1, 101))
        data_float = [float(x) for x in data]
        p99 = _percentile(data_float, 99)
        assert p99 == pytest.approx(99.01, abs=0.1)


# ---------- GET /api/metrics ----------


@pytest.mark.asyncio
async def test_metrics_empty_db(client, app):
    """Metrics with no queries returns zeroes."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [],     # latencies
        [(0,)], # active jobs
    ]))

    resp = await client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.json()
    data = body["data"]
    assert data["total_queries"] == 0
    assert data["latency_p50"] == 0.0
    assert data["latency_p95"] == 0.0
    assert data["latency_p99"] == 0.0
    assert data["active_jobs"] == 0


@pytest.mark.asyncio
async def test_metrics_with_queries(client, app):
    """Metrics correctly computes percentiles from query latencies."""
    latency_rows = [(10.0,), (20.0,), (30.0,), (40.0,), (50.0,)]

    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        latency_rows,  # latencies
        [(0,)],        # active jobs
    ]))

    resp = await client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.json()
    data = body["data"]
    assert data["total_queries"] == 5
    assert data["latency_p50"] == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_metrics_cache_stats(client, app):
    """Metrics includes cache hit rate and size from OllamaClient."""
    cache = EmbeddingCache(max_size=100)
    cache.put("a", [1.0, 2.0])
    cache.get("a")  # hit
    cache.get("b")  # miss
    app.state.ollama._cache = cache

    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [],     # latencies
        [(0,)], # active jobs
    ]))

    resp = await client.get("/api/metrics")
    body = resp.json()
    data = body["data"]
    assert data["cache_hit_rate"] == pytest.approx(0.5)
    assert data["cache_size"] == 1


@pytest.mark.asyncio
async def test_metrics_active_jobs(client, app):
    """Metrics reports count of running jobs."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [],     # latencies
        [(3,)], # active jobs
    ]))

    resp = await client.get("/api/metrics")
    body = resp.json()
    assert body["data"]["active_jobs"] == 3


# ---------- Enhanced /api/stats ----------


@pytest.mark.asyncio
async def test_stats_includes_percentiles(client, app):
    """Stats endpoint now includes latency percentile fields."""
    latency_rows = [(10.0,), (20.0,), (30.0,)]

    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [(3, 20.0)],       # COUNT + AVG
        latency_rows,      # Latencies for percentiles
        [("api", 3)],      # By interface
        [("search", 3)],   # By type
        [("default", 3)],  # Top collections
    ]))

    resp = await client.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    data = body["data"]
    assert "latency_p50" in data
    assert "latency_p95" in data
    assert "latency_p99" in data
    assert data["latency_p50"] == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_stats_percentiles_empty(client, app):
    """Stats with no queries has zero percentiles."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [(0, 0)],  # COUNT + AVG
        [],        # Latencies (empty)
        [],        # By interface
        [],        # By type
        [],        # Top collections
    ]))

    resp = await client.get("/api/stats")
    body = resp.json()
    assert body["data"]["latency_p50"] == 0.0
    assert body["data"]["latency_p95"] == 0.0
    assert body["data"]["latency_p99"] == 0.0
