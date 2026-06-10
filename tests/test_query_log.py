"""Tests for query logging."""

from __future__ import annotations

import json

import pytest

from rag_kb.models.schema import Interface, SearchMode
from rag_kb.models.search import RetrievalResult
from rag_kb.retrieval.engine import RetrievalResponse
from rag_kb.retrieval.query_log import log_query


def _make_response(
    query: str = "test query",
    collection: str = "docs",
    n_results: int = 3,
) -> RetrievalResponse:
    results = [
        RetrievalResult(
            id=f"p{i}", score=0.9 - i * 0.1, content=f"chunk {i}",
            file_path=f"/f{i}.md", file_type="markdown",
        )
        for i in range(n_results)
    ]
    return RetrievalResponse(
        results=results,
        query=query,
        collection=collection,
        mode=SearchMode.HYBRID,
        latency_ms=42.5,
        total=n_results,
    )


class TestLogQuery:
    @pytest.mark.asyncio
    async def test_log_inserts_row(self, tmp_db):
        response = _make_response()
        await log_query(tmp_db, response)

        async with tmp_db.execute("SELECT COUNT(*) FROM queries") as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1

    @pytest.mark.asyncio
    async def test_returns_id(self, tmp_db):
        response = _make_response()
        query_id = await log_query(tmp_db, response)
        assert isinstance(query_id, str)
        assert len(query_id) > 0

    @pytest.mark.asyncio
    async def test_fields_persisted(self, tmp_db):
        response = _make_response(query="semantic search", n_results=5)
        await log_query(tmp_db, response)

        async with tmp_db.execute(
            "SELECT query_text, search_mode, result_count, latency_ms FROM queries"
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == "semantic search"
        assert row[1] == "hybrid"
        assert row[2] == 5
        assert row[3] == pytest.approx(42.5)

    @pytest.mark.asyncio
    async def test_collection_fk_null(self, tmp_db):
        response = _make_response(collection="nonexistent")
        await log_query(tmp_db, response)

        async with tmp_db.execute("SELECT collection_id FROM queries") as cursor:
            row = await cursor.fetchone()
        assert row[0] is None

    @pytest.mark.asyncio
    async def test_metadata_json(self, tmp_db):
        response = _make_response()
        meta = {"user_id": "u123", "session": "s456"}
        await log_query(tmp_db, response, metadata=meta)

        async with tmp_db.execute("SELECT metadata FROM queries") as cursor:
            row = await cursor.fetchone()

        parsed = json.loads(row[0])
        assert parsed["user_id"] == "u123"

    @pytest.mark.asyncio
    async def test_interface_field(self, tmp_db):
        response = _make_response()
        await log_query(tmp_db, response, interface=Interface.CLI)

        async with tmp_db.execute("SELECT interface FROM queries") as cursor:
            row = await cursor.fetchone()
        assert row[0] == "cli"
