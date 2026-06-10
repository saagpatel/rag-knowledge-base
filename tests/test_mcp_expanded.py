"""Tests for the expanded MCP tools: health, stats, query_history."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp.client import Client
from fastmcp.exceptions import ToolError


# --- Fake cursor (reused from test_mcp.py pattern) ---


class _FakeCursor:
    """Cursor that works as async context manager and supports fetchone + iteration."""

    def __init__(self, rows=None, fetchone_val=None):
        self._rows = list(rows) if rows else []
        self._fetchone_val = fetchone_val
        self._idx = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __await__(self):
        async def _self():
            return self
        return _self().__await__()

    async def fetchone(self):
        return self._fetchone_val

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._rows):
            raise StopAsyncIteration
        row = self._rows[self._idx]
        self._idx += 1
        return row


# --- Fixtures ---


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
def mock_ollama():
    m = AsyncMock()
    m.health = AsyncMock(return_value=True)
    m.close = AsyncMock()
    return m


@pytest.fixture
def mock_qdrant():
    m = AsyncMock()
    m.list_collections = AsyncMock(return_value=["rag_docs"])
    m.close = AsyncMock()
    return m


@pytest.fixture
def mcp_server(mock_db, mock_ollama, mock_qdrant):
    """Create an MCP server with mocked lifespan resources."""
    from rag_kb.mcp import mcp

    original_lifespan = mcp._lifespan

    @asynccontextmanager
    async def fake_lifespan(server):
        yield {"db": mock_db, "ollama": mock_ollama, "qdrant": mock_qdrant}

    mcp._lifespan = fake_lifespan
    yield mcp
    mcp._lifespan = original_lifespan


def _parse_list_result(result) -> list:
    if not result.content:
        return []
    return json.loads(result.content[0].text)


# --- Tool listing ---


class TestToolListingExpanded:
    @pytest.mark.asyncio
    async def test_list_tools_returns_twelve(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()

        names = {t.name for t in tools}
        assert names == {
            "search", "ask", "ingest",
            "list_collections", "create_collection", "delete_collection",
            "list_documents", "get_document", "delete_document",
            "health", "stats", "query_history",
        }


# --- Health tool ---


class TestHealthTool:
    @pytest.mark.asyncio
    async def test_health_all_ok(self, mcp_server, mock_db, mock_ollama, mock_qdrant):
        mock_db.execute = MagicMock(return_value=_FakeCursor(fetchone_val=(1,)))

        async with Client(mcp_server) as client:
            result = await client.call_tool("health", {})

        data = result.data
        assert data["status"] == "healthy"
        assert data["ollama"]["status"] == "ok"
        assert data["qdrant"]["status"] == "ok"
        assert data["sqlite"]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_ollama_down(self, mcp_server, mock_db, mock_ollama, mock_qdrant):
        mock_ollama.health = AsyncMock(side_effect=Exception("Connection refused"))
        mock_db.execute = MagicMock(return_value=_FakeCursor(fetchone_val=(1,)))

        async with Client(mcp_server) as client:
            result = await client.call_tool("health", {})

        data = result.data
        assert data["status"] == "degraded"
        assert data["ollama"]["status"] == "error"
        assert "Connection refused" in data["ollama"]["detail"]

    @pytest.mark.asyncio
    async def test_health_qdrant_down(self, mcp_server, mock_db, mock_ollama, mock_qdrant):
        mock_qdrant.list_collections = AsyncMock(side_effect=Exception("Qdrant down"))
        mock_db.execute = MagicMock(return_value=_FakeCursor(fetchone_val=(1,)))

        async with Client(mcp_server) as client:
            result = await client.call_tool("health", {})

        data = result.data
        assert data["status"] == "degraded"
        assert data["qdrant"]["status"] == "error"
        assert "Qdrant down" in data["qdrant"]["detail"]


# --- Stats tool ---


class TestStatsTool:
    @pytest.mark.asyncio
    async def test_stats_empty(self, mcp_server, mock_db):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Total queries + avg latency
                return _FakeCursor(fetchone_val=(0, 0.0))
            # by_interface or by_type: empty
            return _FakeCursor(rows=[])

        mock_db.execute = MagicMock(side_effect=side_effect)

        async with Client(mcp_server) as client:
            result = await client.call_tool("stats", {})

        data = result.data
        assert data["total_queries"] == 0
        assert data["avg_latency_ms"] == 0.0
        assert data["queries_by_interface"] == {}
        assert data["queries_by_type"] == {}

    @pytest.mark.asyncio
    async def test_stats_with_data(self, mcp_server, mock_db):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _FakeCursor(fetchone_val=(42, 55.5))
            elif call_count[0] == 2:
                return _FakeCursor(rows=[("cli", 20), ("api", 22)])
            else:
                return _FakeCursor(rows=[("search", 30), ("qa", 12)])

        mock_db.execute = MagicMock(side_effect=side_effect)

        async with Client(mcp_server) as client:
            result = await client.call_tool("stats", {"days": 7})

        data = result.data
        assert data["total_queries"] == 42
        assert data["avg_latency_ms"] == 55.5
        assert data["queries_by_interface"] == {"cli": 20, "api": 22}
        assert data["queries_by_type"] == {"search": 30, "qa": 12}
        assert data["period_days"] == 7


# --- Query history tool ---


class TestQueryHistoryTool:
    @pytest.mark.asyncio
    async def test_query_history_empty(self, mcp_server, mock_db):
        mock_db.execute = MagicMock(return_value=_FakeCursor(rows=[]))

        async with Client(mcp_server) as client:
            result = await client.call_tool("query_history", {})

        data = _parse_list_result(result)
        assert data == []

    @pytest.mark.asyncio
    async def test_query_history_with_data(self, mcp_server, mock_db):
        rows = [
            ("q1", "test query", "search", "hybrid", 5, 42.0, "cli", "2024-01-01T00:00:00"),
            ("q2", "another", "qa", "dense", 3, 55.0, "api", "2024-01-02T00:00:00"),
        ]
        mock_db.execute = MagicMock(return_value=_FakeCursor(rows=rows))

        async with Client(mcp_server) as client:
            result = await client.call_tool("query_history", {})

        data = _parse_list_result(result)
        assert len(data) == 2
        assert data[0]["query_text"] == "test query"
        assert data[1]["interface"] == "api"

    @pytest.mark.asyncio
    async def test_query_history_with_collection_filter(self, mcp_server, mock_db):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Collection lookup
                return _FakeCursor(fetchone_val=("col-id-1",))
            # Query results
            return _FakeCursor(rows=[
                ("q1", "filtered query", "search", "hybrid", 2, 30.0, "cli", "2024-01-01"),
            ])

        mock_db.execute = MagicMock(side_effect=side_effect)

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "query_history", {"collection": "my_col"}
            )

        data = _parse_list_result(result)
        assert len(data) == 1
        assert data[0]["query_text"] == "filtered query"

    @pytest.mark.asyncio
    async def test_query_history_limit(self, mcp_server, mock_db):
        rows = [
            ("q1", "query1", "search", "hybrid", 1, 10.0, "cli", "2024-01-01"),
        ]
        mock_db.execute = MagicMock(return_value=_FakeCursor(rows=rows))

        async with Client(mcp_server) as client:
            result = await client.call_tool("query_history", {"limit": 1})

        data = _parse_list_result(result)
        assert len(data) == 1
