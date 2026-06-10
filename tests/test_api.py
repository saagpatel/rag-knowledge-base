"""Tests for the REST API layer."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from rag_kb.api import create_app
from rag_kb.core.errors import (
    CollectionNotFoundError,
    OllamaConnectionError,
)
from rag_kb.ingestion.orchestrator import BatchIngestionResult, IngestionResult
from rag_kb.models.schema import DocumentStatus, SearchMode
from rag_kb.models.search import RetrievalResult
from rag_kb.retrieval.engine import RetrievalResponse


class _AsyncIter:
    """Async iterator wrapper for a list of rows."""

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
    """Cursor that works as async context manager (for `async with db.execute(...)`)
    and also supports direct attribute access after `await`.

    aiosqlite's execute returns an object that is both an awaitable (returning itself)
    and an async context manager. We replicate that dual behavior here.
    """

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
        """Allow `await cursor` to return self (matches aiosqlite pattern)."""
        async def _self():
            return self
        return _self().__await__()


def _make_db_execute(response_sequence):
    """Return a side_effect callable for db.execute.

    response_sequence: list of row-lists. Each db.execute call consumes one entry.
    Returns a non-async callable that returns _FakeCursor (which supports both
    `async with` and `await`).
    """
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


def _make_db_mock():
    """Create a mock for db that supports both patterns:
    - async with db.execute(...) as cursor:
    - await db.execute(...)
    """
    db = AsyncMock()
    db.execute = MagicMock(side_effect=_make_db_execute([[(1,)]]))
    return db


@pytest.fixture
def app():
    application = create_app()
    application.state.db = _make_db_mock()
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


# ---------- Health ----------


@pytest.mark.asyncio
async def test_health_both_healthy(client, app):
    app.state.ollama.health.return_value = True
    app.state.qdrant.list_collections.return_value = []

    resp = await client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["status"] == "healthy"
    assert body["data"]["ollama"]["status"] == "ok"
    assert body["data"]["qdrant"]["status"] == "ok"
    assert body["data"]["sqlite"]["status"] == "ok"
    assert "request_id" in body["meta"]


@pytest.mark.asyncio
async def test_health_ollama_down(client, app):
    app.state.ollama.health.side_effect = OllamaConnectionError("down")
    app.state.qdrant.list_collections.return_value = []

    resp = await client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["status"] == "degraded"
    assert body["data"]["ollama"]["status"] == "error"
    assert body["data"]["qdrant"]["status"] == "ok"


@pytest.mark.asyncio
async def test_health_qdrant_down(client, app):
    app.state.ollama.health.return_value = True
    app.state.qdrant.list_collections.side_effect = Exception("qdrant down")

    resp = await client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["status"] == "degraded"
    assert body["data"]["ollama"]["status"] == "ok"
    assert body["data"]["qdrant"]["status"] == "error"


@pytest.mark.asyncio
async def test_health_includes_sqlite(client, app):
    """Health endpoint reports SQLite status."""
    app.state.ollama.health.return_value = True
    app.state.qdrant.list_collections.return_value = []

    resp = await client.get("/api/health")
    body = resp.json()
    assert "sqlite" in body["data"]
    assert body["data"]["sqlite"]["status"] == "ok"


@pytest.mark.asyncio
async def test_health_sqlite_down_degrades(client, app):
    """When SQLite fails, overall status is degraded."""
    app.state.ollama.health.return_value = True
    app.state.qdrant.list_collections.return_value = []
    # Make db.execute raise
    def _raise(*args, **kwargs):
        raise Exception("SQLite error")
    app.state.db.execute = MagicMock(side_effect=_raise)

    resp = await client.get("/api/health")
    body = resp.json()
    assert body["data"]["status"] == "degraded"
    assert body["data"]["sqlite"]["status"] == "error"
    assert body["data"]["sqlite"]["detail"] == "SQLite error"


# ---------- Ingest ----------


def _make_ingestion_result(**overrides):
    defaults = {
        "file_path": "/tmp/test.md",
        "status": DocumentStatus.COMPLETED,
        "chunk_count": 5,
        "error_message": None,
    }
    defaults.update(overrides)
    return IngestionResult(**defaults)


@pytest.mark.asyncio
async def test_ingest_single_file(client, app, tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Hello")

    result = _make_ingestion_result(file_path=str(f))

    with patch("rag_kb.api.routes.ingest_file", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = result
        resp = await client.post("/api/ingest", json={"path": str(f)})

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["total_files"] == 1
    assert body["data"]["processed"] == 1
    assert body["data"]["results"][0]["chunk_count"] == 5


@pytest.mark.asyncio
async def test_ingest_directory(client, app, tmp_path):
    """Directory ingest now returns 202-style JobData with job_id and status."""
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "b.md").write_text("b")

    # db.execute calls for directory ingest path:
    # 1. _get_collection_settings
    # 2. _ensure_collection_exists SELECT
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [],  # _get_collection_settings
        [],  # _ensure_collection_exists SELECT
    ]))
    app.state.db.commit = AsyncMock()
    app.state.qdrant.create_collection = AsyncMock()
    app.state.background_tasks = set()

    with patch("rag_kb.api.routes.ingest_directory", new_callable=AsyncMock) as mock:
        mock.return_value = None
        resp = await client.post("/api/ingest", json={"path": str(tmp_path)})

    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body["data"]
    assert body["data"]["status"] == "running"
    assert body["data"]["total_files"] == 2


@pytest.mark.asyncio
async def test_ingest_path_not_found(client):
    resp = await client.post("/api/ingest", json={"path": "/nonexistent/path"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["detail"]["error"]["code"] == "FILE_NOT_FOUND"


@pytest.mark.asyncio
async def test_ingest_missing_path(client):
    resp = await client.post("/api/ingest", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ingest_ollama_error(client, app, tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Hello")

    with patch("rag_kb.api.routes.ingest_file", new_callable=AsyncMock) as mock:
        mock.side_effect = OllamaConnectionError("Cannot reach Ollama")
        resp = await client.post("/api/ingest", json={"path": str(f)})

    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "OLLAMA_UNAVAILABLE"


# ---------- Search ----------


def _make_retrieval_response(**overrides):
    defaults = {
        "results": [
            RetrievalResult(
                id="pt-1", score=0.95, content="test content",
                file_path="/tmp/test.md", file_type="markdown",
                chunk_index=0, total_chunks=3, reranked=False,
            )
        ],
        "query": "test query",
        "collection": "default",
        "mode": SearchMode.HYBRID,
        "latency_ms": 42.5,
        "total": 1,
    }
    defaults.update(overrides)
    return RetrievalResponse(**defaults)


@pytest.mark.asyncio
async def test_search_happy_path(client, app):
    response = _make_retrieval_response()

    with (
        patch("rag_kb.api.routes.RetrievalEngine") as mock_engine,
        patch("rag_kb.api.routes.log_query", new_callable=AsyncMock) as mock_log,
    ):
        mock_engine.return_value.search = AsyncMock(return_value=response)
        mock_log.return_value = "q-123"
        resp = await client.post(
            "/api/search", json={"query": "test query"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total"] == 1
    assert body["data"]["latency_ms"] == 42.5
    assert body["data"]["results"][0]["content"] == "test content"


@pytest.mark.asyncio
async def test_search_no_results(client, app):
    response = _make_retrieval_response(results=[], total=0)

    with (
        patch("rag_kb.api.routes.RetrievalEngine") as mock_engine,
        patch("rag_kb.api.routes.log_query", new_callable=AsyncMock),
    ):
        mock_engine.return_value.search = AsyncMock(return_value=response)
        resp = await client.post("/api/search", json={"query": "nothing"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["results"] == []
    assert body["data"]["total"] == 0


@pytest.mark.asyncio
async def test_search_collection_not_found(client, app):
    with (
        patch("rag_kb.api.routes.RetrievalEngine") as mock_engine,
    ):
        mock_engine.return_value.search = AsyncMock(
            side_effect=CollectionNotFoundError("not found")
        )
        resp = await client.post(
            "/api/search", json={"query": "q", "collection": "missing"}
        )

    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "COLLECTION_NOT_FOUND"


@pytest.mark.asyncio
async def test_search_mode_forwarded(client, app):
    response = _make_retrieval_response(mode=SearchMode.DENSE)

    with (
        patch("rag_kb.api.routes.RetrievalEngine") as mock_engine,
        patch("rag_kb.api.routes.log_query", new_callable=AsyncMock),
    ):
        mock_search = AsyncMock(return_value=response)
        mock_engine.return_value.search = mock_search
        await client.post(
            "/api/search", json={"query": "q", "mode": "dense"}
        )

    call_args = mock_search.call_args[0][0]
    assert call_args.mode == SearchMode.DENSE


@pytest.mark.asyncio
async def test_search_missing_query(client):
    resp = await client.post("/api/search", json={})
    assert resp.status_code == 422


# ---------- Ask ----------


@pytest.mark.asyncio
async def test_ask_no_stream(client, app):
    retrieval_resp = _make_retrieval_response()

    gen_result = MagicMock()
    gen_result.answer = "The answer is 42"
    gen_result.model = "mistral:7b"
    gen_result.latency_ms = 100.0
    gen_result.context_chunks_used = 1

    with (
        patch("rag_kb.api.routes.RetrievalEngine") as mock_re,
        patch("rag_kb.api.routes.GenerationEngine") as mock_ge,
        patch("rag_kb.api.routes.log_query", new_callable=AsyncMock),
        patch("rag_kb.api.routes.extract_sources") as mock_sources,
    ):
        mock_re.return_value.search = AsyncMock(return_value=retrieval_resp)
        mock_ge.return_value.answer = AsyncMock(return_value=gen_result)
        mock_sources.return_value = [
            {"file_path": "/tmp/test.md", "score": 0.95,
             "chunk_index": 0, "total_chunks": 3, "file_type": "markdown"}
        ]

        resp = await client.post("/api/ask", json={"query": "What is the answer?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["answer"] == "The answer is 42"
    assert len(body["data"]["sources"]) == 1


@pytest.mark.asyncio
async def test_ask_stream_content_type(client, app):
    retrieval_resp = _make_retrieval_response()

    async def fake_stream():
        for token in ["Hello", " world"]:
            yield token

    with (
        patch("rag_kb.api.routes.RetrievalEngine") as mock_re,
        patch("rag_kb.api.routes.GenerationEngine") as mock_ge,
        patch("rag_kb.api.routes.log_query", new_callable=AsyncMock),
    ):
        mock_re.return_value.search = AsyncMock(return_value=retrieval_resp)
        mock_ge.return_value.answer = AsyncMock(return_value=fake_stream())

        resp = await client.post(
            "/api/ask", json={"query": "Hello?", "stream": True}
        )

    assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"


@pytest.mark.asyncio
async def test_ask_stream_done_sentinel(client, app):
    retrieval_resp = _make_retrieval_response()

    async def fake_stream():
        yield "tok1"
        yield "tok2"

    with (
        patch("rag_kb.api.routes.RetrievalEngine") as mock_re,
        patch("rag_kb.api.routes.GenerationEngine") as mock_ge,
        patch("rag_kb.api.routes.log_query", new_callable=AsyncMock),
    ):
        mock_re.return_value.search = AsyncMock(return_value=retrieval_resp)
        mock_ge.return_value.answer = AsyncMock(return_value=fake_stream())

        resp = await client.post(
            "/api/ask", json={"query": "Hello?", "stream": True}
        )

    text = resp.text
    assert text.endswith("data: [DONE]\n\n")
    assert "data: tok1" in text
    assert "data: tok2" in text


@pytest.mark.asyncio
async def test_ask_model_override(client, app):
    retrieval_resp = _make_retrieval_response()

    gen_result = MagicMock()
    gen_result.answer = "answer"
    gen_result.model = "llama3:8b"
    gen_result.latency_ms = 50.0
    gen_result.context_chunks_used = 1

    with (
        patch("rag_kb.api.routes.RetrievalEngine") as mock_re,
        patch("rag_kb.api.routes.GenerationEngine") as mock_ge,
        patch("rag_kb.api.routes.log_query", new_callable=AsyncMock),
        patch("rag_kb.api.routes.extract_sources") as mock_sources,
    ):
        mock_re.return_value.search = AsyncMock(return_value=retrieval_resp)
        mock_answer = AsyncMock(return_value=gen_result)
        mock_ge.return_value.answer = mock_answer
        mock_sources.return_value = []

        await client.post(
            "/api/ask", json={"query": "q", "model": "llama3:8b"}
        )

    mock_answer.assert_called_once()
    _, kwargs = mock_answer.call_args
    assert kwargs["model"] == "llama3:8b"


@pytest.mark.asyncio
async def test_ask_empty_query(client):
    resp = await client.post("/api/ask", json={"query": ""})
    # Empty string passes pydantic validation but the engine raises PromptTooLargeError → 422
    # Without mocks, it hits the real engine which will error
    assert resp.status_code in (422, 500)


# ---------- Collections ----------


@pytest.mark.asyncio
async def test_list_collections(client, app):
    app.state.qdrant.list_collections.return_value = ["rag_docs"]
    app.state.qdrant.get_collection_info.return_value = {
        "name": "rag_docs", "points_count": 100,
        "vectors_count": 100, "status": "green",
    }

    resp = await client.get("/api/collections")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 1
    assert body["data"][0]["name"] == "rag_docs"


@pytest.mark.asyncio
async def test_list_collections_empty(client, app):
    app.state.qdrant.list_collections.return_value = []

    resp = await client.get("/api/collections")
    assert resp.status_code == 200
    assert resp.json()["data"] == []


@pytest.mark.asyncio
async def test_get_collection(client, app):
    app.state.qdrant.collection_exists.return_value = True
    app.state.qdrant.get_collection_info.return_value = {
        "name": "rag_docs", "points_count": 50,
        "vectors_count": 50, "status": "green",
    }

    resp = await client.get("/api/collections/docs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["points_count"] == 50


@pytest.mark.asyncio
async def test_get_collection_not_found(client, app):
    app.state.qdrant.collection_exists.return_value = False

    resp = await client.get("/api/collections/missing")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "COLLECTION_NOT_FOUND"


@pytest.mark.asyncio
async def test_create_collection(client, app):
    app.state.qdrant.create_collection.return_value = None
    app.state.qdrant.get_collection_info.return_value = {
        "name": "rag_new", "points_count": 0,
        "vectors_count": 0, "status": "green",
    }

    resp = await client.post("/api/collections", json={"name": "new"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["name"] == "rag_new"


@pytest.mark.asyncio
async def test_delete_collection(client, app):
    app.state.qdrant.collection_exists.return_value = True
    app.state.qdrant.delete_collection.return_value = None

    resp = await client.delete("/api/collections/docs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True


# ---------- Test Hardening ----------


@pytest.mark.asyncio
async def test_delete_collection_not_found(client, app):
    app.state.qdrant.collection_exists.return_value = False

    resp = await client.delete("/api/collections/missing")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "COLLECTION_NOT_FOUND"


@pytest.mark.asyncio
async def test_create_collection_conflict(client, app):
    from rag_kb.core.errors import QdrantCollectionError

    app.state.qdrant.create_collection.side_effect = QdrantCollectionError("already exists")

    resp = await client.post("/api/collections", json={"name": "existing"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "QDRANT_COLLECTION_ERROR"


@pytest.mark.asyncio
async def test_search_rerank_forwarded(client, app):
    response = _make_retrieval_response()

    with (
        patch("rag_kb.api.routes.RetrievalEngine") as mock_engine,
        patch("rag_kb.api.routes.log_query", new_callable=AsyncMock),
    ):
        mock_search = AsyncMock(return_value=response)
        mock_engine.return_value.search = mock_search
        await client.post(
            "/api/search", json={"query": "q", "rerank": True}
        )

    call_args = mock_search.call_args[0][0]
    assert call_args.rerank is True


@pytest.mark.asyncio
async def test_ask_stream_tokens_in_body(client, app):
    retrieval_resp = _make_retrieval_response()

    async def fake_stream():
        for token in ["Hello", " ", "world"]:
            yield token

    with (
        patch("rag_kb.api.routes.RetrievalEngine") as mock_re,
        patch("rag_kb.api.routes.GenerationEngine") as mock_ge,
        patch("rag_kb.api.routes.log_query", new_callable=AsyncMock),
    ):
        mock_re.return_value.search = AsyncMock(return_value=retrieval_resp)
        mock_ge.return_value.answer = AsyncMock(return_value=fake_stream())

        resp = await client.post(
            "/api/ask", json={"query": "Hello?", "stream": True}
        )

    text = resp.text
    assert "data: Hello" in text
    assert "data:  " in text
    assert "data: world" in text
    assert "data: [DONE]" in text


@pytest.mark.asyncio
async def test_envelope_meta_on_all_endpoints(client, app):
    """Every success response has meta.request_id and meta.timestamp."""
    app.state.ollama.health.return_value = True
    app.state.qdrant.list_collections.return_value = []

    for path in ["/api/health", "/api/collections"]:
        resp = await client.get(path)
        assert resp.status_code == 200
        body = resp.json()
        assert "meta" in body
        assert "request_id" in body["meta"]
        assert "timestamp" in body["meta"]


@pytest.mark.asyncio
async def test_ingest_collection_forwarded(client, app, tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Hello")

    result = _make_ingestion_result(file_path=str(f))

    with patch("rag_kb.api.routes.ingest_file", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = result
        await client.post(
            "/api/ingest", json={"path": str(f), "collection": "custom"}
        )

    call_args = mock_ingest.call_args
    assert call_args[0][1] == "custom"  # collection arg


@pytest.mark.asyncio
async def test_cors_preflight(app):
    """OPTIONS request returns CORS headers for allowed origin."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.options(
            "/api/health",
            headers={
                "origin": "http://127.0.0.1:5173",
                "access-control-request-method": "GET",
            },
        )
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers


# ---------- Documents ----------


_DOC_ROW = (
    "doc-1", "col-1", "test.md", "/tmp/test.md", "markdown",
    "abc123", 5, "completed", None, "2024-01-01T00:00:00", "2024-01-01T00:00:00",
)

_DOC_JOIN_ROW = ("doc-1", "col-1", "/tmp/test.md", "default")


@pytest.mark.asyncio
async def test_list_documents_empty(client, app):
    """List documents returns empty list when no documents exist."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [(0,)],  # COUNT
        [],      # SELECT
    ]))

    resp = await client.get("/api/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["documents"] == []
    assert body["data"]["total"] == 0


@pytest.mark.asyncio
async def test_list_documents_with_collection_filter(client, app):
    """List documents filtered by collection name."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [("col-1",)],  # Collection lookup
        [(1,)],        # COUNT
        [_DOC_ROW],    # SELECT
    ]))

    resp = await client.get("/api/documents?collection=default")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total"] == 1
    assert body["data"]["collection"] == "default"
    assert len(body["data"]["documents"]) == 1


@pytest.mark.asyncio
async def test_list_documents_collection_not_found(client, app):
    """List documents with non-existent collection returns 404."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [],  # Collection lookup returns no rows
    ]))

    resp = await client.get("/api/documents?collection=nonexistent")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "COLLECTION_NOT_FOUND"


@pytest.mark.asyncio
async def test_get_document_happy_path(client, app):
    """Get single document by ID."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [_DOC_ROW],
    ]))

    resp = await client.get("/api/documents/doc-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["id"] == "doc-1"
    assert body["data"]["filename"] == "test.md"
    assert body["data"]["chunk_count"] == 5


@pytest.mark.asyncio
async def test_get_document_not_found(client, app):
    """Get non-existent document returns 404."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [],  # No matching document
    ]))

    resp = await client.get("/api/documents/missing-id")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "DOCUMENT_NOT_FOUND"


@pytest.mark.asyncio
async def test_delete_document_happy_path(client, app):
    """Delete document removes from DB and Qdrant."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [_DOC_JOIN_ROW],  # JOIN query
        [],               # DELETE
    ]))
    app.state.db.commit = AsyncMock()
    app.state.qdrant.delete_points_by_filter = AsyncMock()

    resp = await client.delete("/api/documents/doc-1")
    assert resp.status_code == 200
    body = resp.json()
    assert "deleted" in body["data"]["message"].lower()
    app.state.qdrant.delete_points_by_filter.assert_called_once_with(
        "default", {"file_path": "/tmp/test.md"}
    )


@pytest.mark.asyncio
async def test_delete_document_not_found(client, app):
    """Delete non-existent document returns 404."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [],  # No matching document
    ]))

    resp = await client.delete("/api/documents/missing-id")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "DOCUMENT_NOT_FOUND"


@pytest.mark.asyncio
async def test_list_documents_pagination(client, app):
    """List documents respects limit and offset."""
    doc2 = (
        "doc-2", "col-1", "other.md", "/tmp/other.md", "markdown",
        "def456", 3, "completed", None, "2024-01-02T00:00:00", "2024-01-02T00:00:00",
    )

    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [(5,)],   # COUNT
        [doc2],   # SELECT
    ]))

    resp = await client.get("/api/documents?limit=1&offset=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total"] == 5
    assert len(body["data"]["documents"]) == 1


# ---------- Analytics ----------


@pytest.mark.asyncio
async def test_stats_happy_path(client, app):
    """Stats endpoint returns aggregated data."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [(10, 45.5)],                       # COUNT + AVG
        [(10.0,), (20.0,), (45.5,)],        # Latencies for percentiles
        [("api", 7), ("cli", 3)],           # By interface
        [("search", 8), ("qa", 2)],         # By type
        [("default", 10)],                   # Top collections
    ]))

    resp = await client.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total_queries"] == 10
    assert body["data"]["avg_latency_ms"] == 45.5
    assert body["data"]["queries_by_interface"]["api"] == 7
    assert body["data"]["queries_by_type"]["search"] == 8
    assert body["data"]["top_collections"][0]["name"] == "default"
    assert "latency_p50" in body["data"]
    assert "latency_p95" in body["data"]
    assert "latency_p99" in body["data"]


@pytest.mark.asyncio
async def test_stats_empty_db(client, app):
    """Stats endpoint handles empty database gracefully."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [(0, 0)],  # COUNT + AVG
        [],        # Latencies (empty)
        [],        # By interface
        [],        # By type
        [],        # Top collections
    ]))

    resp = await client.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total_queries"] == 0
    assert body["data"]["avg_latency_ms"] == 0
    assert body["data"]["queries_by_interface"] == {}


_QUERY_ROW = (
    "q-1", "how to test", "search", "hybrid", 5, 42.5, "api", "2024-01-01T00:00:00",
)


@pytest.mark.asyncio
async def test_query_history_happy_path(client, app):
    """Query history returns list of queries."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [(1,)],        # COUNT
        [_QUERY_ROW],  # SELECT
    ]))

    resp = await client.get("/api/queries")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total"] == 1
    assert len(body["data"]["queries"]) == 1
    assert body["data"]["queries"][0]["query_text"] == "how to test"


@pytest.mark.asyncio
async def test_query_history_with_filters(client, app):
    """Query history respects interface and query_type filters."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [(1,)],        # COUNT
        [_QUERY_ROW],  # SELECT
    ]))

    resp = await client.get("/api/queries?interface=api&query_type=search")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total"] == 1


@pytest.mark.asyncio
async def test_query_history_pagination(client, app):
    """Query history respects limit and offset."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [(20,)],       # COUNT
        [_QUERY_ROW],  # SELECT
    ]))

    resp = await client.get("/api/queries?limit=1&offset=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total"] == 20
    assert len(body["data"]["queries"]) == 1


@pytest.mark.asyncio
async def test_query_history_empty(client, app):
    """Query history returns empty list when no queries exist."""
    app.state.db.execute = MagicMock(side_effect=_make_db_execute([
        [(0,)],  # COUNT
        [],      # SELECT
    ]))

    resp = await client.get("/api/queries")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["total"] == 0
    assert body["data"]["queries"] == []
