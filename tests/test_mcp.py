"""Tests for the MCP server tools."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.client import Client
from fastmcp.exceptions import ToolError

from rag_kb.core.errors import CollectionNotFoundError, OllamaConnectionError
from rag_kb.models.schema import DocumentStatus, SearchMode
from rag_kb.models.search import RetrievalResult

# --- Helpers ---


@dataclass
class FakeRetrievalResponse:
    results: list[RetrievalResult]
    query: str
    collection: str
    mode: SearchMode
    latency_ms: float
    total: int


@dataclass
class FakeGenerationResult:
    answer: str
    sources: list[dict[str, object]]
    query: str
    model: str
    latency_ms: float
    context_chunks_used: int


@dataclass
class FakeIngestionResult:
    file_path: str
    status: str
    chunk_count: int = 0
    error_message: str | None = None


@dataclass
class FakeBatchIngestionResult:
    total_files: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list = field(default_factory=list)


def _make_retrieval_results(n: int = 2) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            id=f"id-{i}",
            score=0.9 - i * 0.1,
            content=f"Content {i}",
            file_path=f"/docs/file{i}.md",
            file_type="markdown",
            chunk_index=i,
            total_chunks=n,
        )
        for i in range(n)
    ]


def _make_retrieval_response(n: int = 2) -> FakeRetrievalResponse:
    return FakeRetrievalResponse(
        results=_make_retrieval_results(n),
        query="test",
        collection="default",
        mode=SearchMode.HYBRID,
        latency_ms=42.0,
        total=n,
    )


# --- Fixtures ---


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
def mock_ollama():
    m = AsyncMock()
    m.embed = AsyncMock(return_value=[0.1] * 768)
    m.close = AsyncMock()
    config_mock = type("Config", (), {"generation_model": "mistral:7b"})()
    m._config = config_mock
    return m


@pytest.fixture
def mock_qdrant():
    m = AsyncMock()
    m.list_collections = AsyncMock(return_value=["rag_docs"])
    m.get_collection_info = AsyncMock(return_value={
        "name": "rag_docs",
        "points_count": 100,
        "vectors_count": 100,
        "status": "green",
    })
    m.create_collection = AsyncMock()
    m.delete_collection = AsyncMock()
    m.collection_exists = AsyncMock(return_value=True)
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
    """Parse a list result from call_tool (data wraps items in Root objects)."""
    if not result.content:
        return []
    return json.loads(result.content[0].text)


# --- Tool listing ---


class TestToolListing:
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


# --- Search tool ---


class TestSearchTool:
    @pytest.mark.asyncio
    async def test_search_happy_path(self, mcp_server):
        response = _make_retrieval_response(2)

        with (
            patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_cls,
            patch("rag_kb.retrieval.engine.load_bm25"),
        ):
            mock_engine = AsyncMock()
            mock_engine.search.return_value = response
            mock_cls.return_value = mock_engine

            async with Client(mcp_server) as client:
                result = await client.call_tool(
                    "search", {"query": "test query", "collection": "default"}
                )

        data = result.data
        assert data["total"] == 2
        assert len(data["results"]) == 2
        assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_search_collection_not_found(self, mcp_server):
        with patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_cls:
            mock_engine = AsyncMock()
            mock_engine.search.side_effect = CollectionNotFoundError("not found")
            mock_cls.return_value = mock_engine

            async with Client(mcp_server) as client:
                with pytest.raises(ToolError):
                    await client.call_tool(
                        "search", {"query": "q", "collection": "missing"}
                    )

    @pytest.mark.asyncio
    async def test_search_ollama_down(self, mcp_server):
        with patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_cls:
            mock_engine = AsyncMock()
            mock_engine.search.side_effect = OllamaConnectionError("down")
            mock_cls.return_value = mock_engine

            async with Client(mcp_server) as client:
                with pytest.raises(ToolError, match="Ollama unavailable"):
                    await client.call_tool("search", {"query": "q"})


# --- Ask tool ---


class TestAskTool:
    @pytest.mark.asyncio
    async def test_ask_happy_path(self, mcp_server):
        retrieval_resp = _make_retrieval_response(2)
        gen_result = FakeGenerationResult(
            answer="The answer is 42",
            sources=[],
            query="question",
            model="mistral:7b",
            latency_ms=100.0,
            context_chunks_used=2,
        )

        with (
            patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_re,
            patch("rag_kb.generation.engine.GenerationEngine") as mock_ge,
        ):
            mock_re.return_value.search = AsyncMock(return_value=retrieval_resp)
            mock_ge.return_value.answer = AsyncMock(return_value=gen_result)

            async with Client(mcp_server) as client:
                result = await client.call_tool(
                    "ask", {"query": "What is the meaning of life?"}
                )

        data = result.data
        assert data["answer"] == "The answer is 42"
        assert "sources" in data
        assert data["model"] == "mistral:7b"

    @pytest.mark.asyncio
    async def test_ask_collection_not_found(self, mcp_server):
        with patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_re:
            mock_re.return_value.search = AsyncMock(
                side_effect=CollectionNotFoundError("nope")
            )

            async with Client(mcp_server) as client:
                with pytest.raises(ToolError):
                    await client.call_tool(
                        "ask", {"query": "q", "collection": "missing"}
                    )


# --- Ingest tool ---


class TestIngestTool:
    @pytest.mark.asyncio
    async def test_ingest_file(self, mcp_server, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# Hello world")

        fake_result = FakeIngestionResult(
            file_path=str(f),
            status=DocumentStatus.COMPLETED,
            chunk_count=3,
        )

        with patch(
            "rag_kb.ingestion.orchestrator.ingest_file",
            new_callable=AsyncMock,
            return_value=fake_result,
        ):
            async with Client(mcp_server) as client:
                result = await client.call_tool(
                    "ingest", {"path": str(f)}
                )

        data = result.data
        assert data["total_files"] == 1
        assert data["processed"] == 1
        assert data["results"][0]["chunk_count"] == 3

    @pytest.mark.asyncio
    async def test_ingest_directory(self, mcp_server, tmp_path):
        (tmp_path / "a.md").write_text("# A")
        batch = FakeBatchIngestionResult(
            total_files=1, processed=1,
            results=[FakeIngestionResult(
                file_path=str(tmp_path / "a.md"),
                status=DocumentStatus.COMPLETED,
                chunk_count=2,
            )],
        )

        with patch(
            "rag_kb.ingestion.orchestrator.ingest_directory",
            new_callable=AsyncMock,
            return_value=batch,
        ):
            async with Client(mcp_server) as client:
                result = await client.call_tool(
                    "ingest", {"path": str(tmp_path)}
                )

        data = result.data
        assert data["total_files"] == 1
        assert data["processed"] == 1

    @pytest.mark.asyncio
    async def test_ingest_path_not_found(self, mcp_server):
        async with Client(mcp_server) as client:
            with pytest.raises(ToolError, match="does not exist"):
                await client.call_tool(
                    "ingest", {"path": "/nonexistent/file.md"}
                )

    @pytest.mark.asyncio
    async def test_ingest_ollama_down(self, mcp_server, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# Hello")

        with patch(
            "rag_kb.ingestion.orchestrator.ingest_file",
            new_callable=AsyncMock,
            side_effect=OllamaConnectionError("down"),
        ):
            async with Client(mcp_server) as client:
                with pytest.raises(ToolError, match="Ollama unavailable"):
                    await client.call_tool("ingest", {"path": str(f)})


# --- List collections ---


class TestListCollections:
    @pytest.mark.asyncio
    async def test_list_collections(self, mcp_server, mock_qdrant):
        async with Client(mcp_server) as client:
            result = await client.call_tool("list_collections", {})

        data = _parse_list_result(result)
        assert len(data) == 1
        assert data[0]["name"] == "rag_docs"

    @pytest.mark.asyncio
    async def test_list_collections_empty(self, mcp_server, mock_qdrant):
        mock_qdrant.list_collections.return_value = []

        async with Client(mcp_server) as client:
            result = await client.call_tool("list_collections", {})

        data = _parse_list_result(result)
        assert data == []


# --- Create collection ---


class TestCreateCollection:
    @pytest.mark.asyncio
    async def test_create_collection(self, mcp_server, mock_qdrant):
        mock_qdrant.get_collection_info.return_value = {
            "name": "rag_new",
            "points_count": 0,
            "vectors_count": 0,
            "status": "green",
        }

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "create_collection", {"name": "new"}
            )

        data = result.data
        assert data["name"] == "rag_new"
        mock_qdrant.create_collection.assert_called_once_with("new")


# --- Delete collection ---


# --- Fake cursor helper ---


class _FakeCursor:
    """Cursor that works as async context manager, awaitable, and supports fetchone + iteration."""

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


# --- List documents ---


class TestListDocuments:
    @pytest.mark.asyncio
    async def test_list_documents(self, mcp_server, mock_db):
        rows = [
            ("doc1", "readme.md", "/docs/readme.md", "md", 5, "completed", "2024-01-01"),
            ("doc2", "code.py", "/docs/code.py", "py", 3, "completed", "2024-01-02"),
        ]
        mock_db.execute = MagicMock(return_value=_FakeCursor(rows=rows))

        async with Client(mcp_server) as client:
            result = await client.call_tool("list_documents", {})

        data = _parse_list_result(result)
        assert len(data) == 2
        assert data[0]["filename"] == "readme.md"

    @pytest.mark.asyncio
    async def test_list_documents_by_collection(self, mcp_server, mock_db):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _FakeCursor(fetchone_val=("col-id-1",))
            return _FakeCursor(rows=[
                ("doc1", "readme.md", "/docs/readme.md", "md", 5, "completed", "2024-01-01"),
            ])

        mock_db.execute = MagicMock(side_effect=side_effect)

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "list_documents", {"collection": "my_col"}
            )

        data = _parse_list_result(result)
        assert len(data) == 1


# --- Get document ---


class TestGetDocument:
    @pytest.mark.asyncio
    async def test_get_document(self, mcp_server, mock_db):
        mock_db.execute = MagicMock(return_value=_FakeCursor(fetchone_val=(
            "doc1", "col1", "readme.md", "/docs/readme.md", "md",
            "abc123", 5, "completed", None, "2024-01-01", "2024-01-02",
        )))

        async with Client(mcp_server) as client:
            result = await client.call_tool("get_document", {"doc_id": "doc1"})

        data = result.data
        assert data["id"] == "doc1"
        assert data["filename"] == "readme.md"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, mcp_server, mock_db):
        mock_db.execute = MagicMock(return_value=_FakeCursor(fetchone_val=None))

        async with Client(mcp_server) as client:
            with pytest.raises(ToolError, match="not found"):
                await client.call_tool("get_document", {"doc_id": "missing"})


# --- Delete document ---


class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_delete_document(self, mcp_server, mock_db, mock_qdrant):
        mock_db.execute = MagicMock(return_value=_FakeCursor(
            fetchone_val=("doc1", "/docs/readme.md", "my_col")
        ))
        mock_db.commit = AsyncMock()
        mock_qdrant.delete_points_by_filter = AsyncMock()

        async with Client(mcp_server) as client:
            result = await client.call_tool("delete_document", {"doc_id": "doc1"})

        data = result.data
        assert "deleted" in data["message"].lower()
        mock_qdrant.delete_points_by_filter.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, mcp_server, mock_db):
        mock_db.execute = MagicMock(return_value=_FakeCursor(fetchone_val=None))

        async with Client(mcp_server) as client:
            with pytest.raises(ToolError, match="not found"):
                await client.call_tool("delete_document", {"doc_id": "missing"})


class TestDeleteCollection:
    @pytest.mark.asyncio
    async def test_delete_collection(self, mcp_server, mock_qdrant):
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "delete_collection", {"name": "docs"}
            )

        data = result.data
        assert "deleted" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_delete_collection_not_found(self, mcp_server, mock_qdrant):
        mock_qdrant.collection_exists.return_value = False

        async with Client(mcp_server) as client:
            with pytest.raises(ToolError, match="does not exist"):
                await client.call_tool(
                    "delete_collection", {"name": "missing"}
                )
