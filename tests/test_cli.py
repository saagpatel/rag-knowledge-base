"""Tests for the CLI interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from rag_kb.cli import main
from rag_kb.core.errors import CollectionNotFoundError, OllamaConnectionError
from rag_kb.models.schema import DocumentStatus, SearchMode
from rag_kb.models.search import RetrievalResult


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
    results: list[Any] = field(default_factory=list)


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


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_resources():
    """Patch _resources to yield mocked db/ollama/qdrant."""
    mock_db = AsyncMock()
    mock_ollama = AsyncMock()
    mock_qdrant = AsyncMock()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_resources(need_db=True, need_ollama=True, need_qdrant=True):
        yield {
            "db": mock_db if need_db else None,
            "ollama": mock_ollama if need_ollama else None,
            "qdrant": mock_qdrant if need_qdrant else None,
        }

    with patch("rag_kb.cli._resources", fake_resources):
        yield mock_db, mock_ollama, mock_qdrant


def _make_retrieval_results(n: int = 3) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            id=f"id-{i}",
            score=0.9 - i * 0.1,
            content=f"Content of chunk {i}",
            file_path=f"/docs/file{i}.md",
            file_type="markdown",
            chunk_index=i,
            total_chunks=5,
        )
        for i in range(n)
    ]


def _make_retrieval_response(n: int = 3) -> FakeRetrievalResponse:
    results = _make_retrieval_results(n)
    return FakeRetrievalResponse(
        results=results,
        query="test query",
        collection="default",
        mode=SearchMode.HYBRID,
        latency_ms=42.5,
        total=n,
    )


# --- Version / Help ---


class TestBasicFlags:
    def test_version_flag(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower() or "0." in result.output

    def test_help_flag(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "RAG Knowledge Base" in result.output


# --- Ingest ---


class TestIngest:
    def test_ingest_single_file(self, runner, mock_resources, tmp_path):
        mock_db, mock_ollama, mock_qdrant = mock_resources
        test_file = tmp_path / "test.md"
        test_file.write_text("# Hello")

        fake_result = FakeIngestionResult(
            file_path=str(test_file),
            status=DocumentStatus.COMPLETED,
            chunk_count=3,
        )

        with patch(
            "rag_kb.ingestion.orchestrator.ingest_file",
            new_callable=AsyncMock,
            return_value=fake_result,
        ) as mock_fn:
            result = runner.invoke(main, ["ingest", str(test_file)])
            assert result.exit_code == 0
            assert "3 chunks" in result.output
            mock_fn.assert_called_once()

    def test_ingest_directory(self, runner, mock_resources, tmp_path):
        mock_db, mock_ollama, mock_qdrant = mock_resources
        (tmp_path / "a.md").write_text("# A")

        fake_result = FakeBatchIngestionResult(
            total_files=1,
            processed=1,
            results=[
                FakeIngestionResult(
                    file_path=str(tmp_path / "a.md"),
                    status=DocumentStatus.COMPLETED,
                    chunk_count=2,
                )
            ],
        )

        with patch(
            "rag_kb.ingestion.orchestrator.ingest_directory",
            new_callable=AsyncMock,
            return_value=fake_result,
        ):
            result = runner.invoke(main, ["ingest", str(tmp_path)])
            assert result.exit_code == 0
            assert "1 processed" in result.output

    def test_ingest_nonexistent_path(self, runner):
        result = runner.invoke(main, ["ingest", "/nonexistent/path.md"])
        assert result.exit_code == 2  # Click validation error

    def test_ingest_connection_error(self, runner, mock_resources, tmp_path):
        test_file = tmp_path / "test.md"
        test_file.write_text("# Hello")

        with patch(
            "rag_kb.ingestion.orchestrator.ingest_file",
            new_callable=AsyncMock,
            side_effect=OllamaConnectionError("Cannot reach Ollama"),
        ):
            result = runner.invoke(main, ["ingest", str(test_file)])
            assert result.exit_code == 1
            assert "Ollama" in result.output or "ollama" in result.output.lower()

    def test_ingest_options_forwarded(self, runner, mock_resources, tmp_path):
        mock_db, mock_ollama, mock_qdrant = mock_resources
        test_file = tmp_path / "test.md"
        test_file.write_text("# Hello")

        fake_result = FakeIngestionResult(
            file_path=str(test_file),
            status=DocumentStatus.COMPLETED,
            chunk_count=1,
        )

        with patch(
            "rag_kb.ingestion.orchestrator.ingest_file",
            new_callable=AsyncMock,
            return_value=fake_result,
        ) as mock_fn:
            runner.invoke(
                main, ["ingest", str(test_file), "--chunk-size", "256"]
            )
            call_kwargs = mock_fn.call_args
            assert call_kwargs.kwargs.get("chunk_size") == 256


# --- Search ---


class TestSearch:
    def test_search_happy_path(self, runner, mock_resources):
        response = _make_retrieval_response(3)

        with (
            patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_engine_cls,
            patch("rag_kb.retrieval.query_log.log_query", new_callable=AsyncMock),
        ):
            mock_engine = AsyncMock()
            mock_engine.search.return_value = response
            mock_engine_cls.return_value = mock_engine

            result = runner.invoke(main, ["search", "test query"])
            assert result.exit_code == 0
            assert "file0.md" in result.output
            assert "3 results" in result.output

    def test_search_no_results(self, runner, mock_resources):
        response = _make_retrieval_response(0)

        with (
            patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_engine_cls,
            patch("rag_kb.retrieval.query_log.log_query", new_callable=AsyncMock),
        ):
            mock_engine = AsyncMock()
            mock_engine.search.return_value = response
            mock_engine_cls.return_value = mock_engine

            result = runner.invoke(main, ["search", "nothing"])
            assert result.exit_code == 0
            assert "0 results" in result.output

    def test_search_rerank_flag(self, runner, mock_resources):
        response = _make_retrieval_response(1)

        with (
            patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_engine_cls,
            patch("rag_kb.retrieval.query_log.log_query", new_callable=AsyncMock),
        ):
            mock_engine = AsyncMock()
            mock_engine.search.return_value = response
            mock_engine_cls.return_value = mock_engine

            runner.invoke(main, ["search", "test", "--rerank"])
            call_args = mock_engine.search.call_args[0][0]
            assert call_args.rerank is True

    def test_search_mode_option(self, runner, mock_resources):
        response = _make_retrieval_response(1)

        with (
            patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_engine_cls,
            patch("rag_kb.retrieval.query_log.log_query", new_callable=AsyncMock),
        ):
            mock_engine = AsyncMock()
            mock_engine.search.return_value = response
            mock_engine_cls.return_value = mock_engine

            runner.invoke(main, ["search", "test", "--mode", "sparse"])
            call_args = mock_engine.search.call_args[0][0]
            assert call_args.mode == SearchMode.SPARSE

    def test_search_collection_not_found(self, runner, mock_resources):
        with (
            patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_engine_cls,
            patch("rag_kb.retrieval.query_log.log_query", new_callable=AsyncMock),
        ):
            mock_engine = AsyncMock()
            mock_engine.search.side_effect = CollectionNotFoundError("Collection 'x' not found")
            mock_engine_cls.return_value = mock_engine

            result = runner.invoke(main, ["search", "test", "-c", "x"])
            assert result.exit_code == 1


# --- Ask ---


class TestAsk:
    def test_ask_no_stream(self, runner, mock_resources):
        response = _make_retrieval_response(2)
        gen_result = FakeGenerationResult(
            answer="The answer is 42.",
            sources=[{
                "file_path": "/docs/file0.md", "score": 0.9,
                "chunk_index": 0, "total_chunks": 5, "file_type": "markdown",
            }],
            query="test",
            model="mistral:7b",
            latency_ms=100.0,
            context_chunks_used=2,
        )

        with (
            patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_ret_cls,
            patch("rag_kb.generation.engine.GenerationEngine") as mock_gen_cls,
            patch("rag_kb.retrieval.query_log.log_query", new_callable=AsyncMock),
        ):
            mock_ret = AsyncMock()
            mock_ret.search.return_value = response
            mock_ret_cls.return_value = mock_ret

            mock_gen = AsyncMock()
            mock_gen.answer.return_value = gen_result
            mock_gen_cls.return_value = mock_gen

            result = runner.invoke(main, ["ask", "what is it?", "--no-stream"])
            assert result.exit_code == 0
            assert "42" in result.output

    def test_ask_streaming(self, runner, mock_resources):
        response = _make_retrieval_response(2)

        async def fake_stream(*args, **kwargs):
            for token in ["Hello", " ", "world"]:
                yield token

        with (
            patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_ret_cls,
            patch("rag_kb.generation.engine.GenerationEngine") as mock_gen_cls,
            patch("rag_kb.retrieval.query_log.log_query", new_callable=AsyncMock),
        ):
            mock_ret = AsyncMock()
            mock_ret.search.return_value = response
            mock_ret_cls.return_value = mock_ret

            mock_gen = AsyncMock()
            mock_gen.answer.return_value = fake_stream()
            mock_gen_cls.return_value = mock_gen

            result = runner.invoke(main, ["ask", "what is it?"])
            assert result.exit_code == 0

    def test_ask_sources_shown(self, runner, mock_resources):
        response = _make_retrieval_response(1)
        gen_result = FakeGenerationResult(
            answer="Here is the answer.",
            sources=[{
                "file_path": "/docs/file0.md", "score": 0.9,
                "chunk_index": 0, "total_chunks": 5, "file_type": "markdown",
            }],
            query="test",
            model="mistral:7b",
            latency_ms=50.0,
            context_chunks_used=1,
        )

        with (
            patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_ret_cls,
            patch("rag_kb.generation.engine.GenerationEngine") as mock_gen_cls,
            patch("rag_kb.retrieval.query_log.log_query", new_callable=AsyncMock),
        ):
            mock_ret = AsyncMock()
            mock_ret.search.return_value = response
            mock_ret_cls.return_value = mock_ret

            mock_gen = AsyncMock()
            mock_gen.answer.return_value = gen_result
            mock_gen_cls.return_value = mock_gen

            result = runner.invoke(main, ["ask", "test", "--no-stream"])
            assert result.exit_code == 0
            assert "file0.md" in result.output

    def test_ask_model_override(self, runner, mock_resources):
        response = _make_retrieval_response(1)
        gen_result = FakeGenerationResult(
            answer="OK",
            sources=[],
            query="test",
            model="llama3:8b",
            latency_ms=50.0,
            context_chunks_used=1,
        )

        with (
            patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_ret_cls,
            patch("rag_kb.generation.engine.GenerationEngine") as mock_gen_cls,
            patch("rag_kb.retrieval.query_log.log_query", new_callable=AsyncMock),
        ):
            mock_ret = AsyncMock()
            mock_ret.search.return_value = response
            mock_ret_cls.return_value = mock_ret

            mock_gen = AsyncMock()
            mock_gen.answer.return_value = gen_result
            mock_gen_cls.return_value = mock_gen

            runner.invoke(main, ["ask", "test", "--model", "llama3:8b", "--no-stream"])
            call_kwargs = mock_gen.answer.call_args
            assert call_kwargs.kwargs.get("model") == "llama3:8b"


# --- Collections ---


class TestCollections:
    def test_collections_list(self, runner, mock_resources):
        _, _, mock_qdrant = mock_resources
        mock_qdrant.list_collections.return_value = ["rag_docs", "rag_code"]
        mock_qdrant.get_collection_info.side_effect = [
            {"name": "rag_docs", "points_count": 100, "vectors_count": 100, "status": "green"},
            {"name": "rag_code", "points_count": 50, "vectors_count": 50, "status": "green"},
        ]

        result = runner.invoke(main, ["collections", "list"])
        assert result.exit_code == 0
        assert "rag_docs" in result.output
        assert "rag_code" in result.output

    def test_collections_list_empty(self, runner, mock_resources):
        _, _, mock_qdrant = mock_resources
        mock_qdrant.list_collections.return_value = []

        result = runner.invoke(main, ["collections", "list"])
        assert result.exit_code == 0
        assert "No collections" in result.output

    def test_collections_info(self, runner, mock_resources):
        _, _, mock_qdrant = mock_resources
        mock_qdrant.get_collection_info.return_value = {
            "name": "rag_docs",
            "points_count": 100,
            "vectors_count": 100,
            "status": "green",
        }

        result = runner.invoke(main, ["collections", "info", "docs"])
        assert result.exit_code == 0
        assert "rag_docs" in result.output

    def test_collections_create(self, runner, mock_resources):
        _, _, mock_qdrant = mock_resources

        result = runner.invoke(main, ["collections", "create", "mydata"])
        assert result.exit_code == 0
        assert "created" in result.output.lower()
        mock_qdrant.create_collection.assert_called_once_with("mydata")

    def test_collections_delete_confirmed(self, runner, mock_resources):
        _, _, mock_qdrant = mock_resources

        result = runner.invoke(main, ["collections", "delete", "mydata"], input="y\n")
        assert result.exit_code == 0
        assert "deleted" in result.output.lower()
        mock_qdrant.delete_collection.assert_called_once_with("mydata")

    def test_collections_delete_aborted(self, runner, mock_resources):
        _, _, mock_qdrant = mock_resources

        result = runner.invoke(main, ["collections", "delete", "mydata"], input="n\n")
        assert result.exit_code != 0  # Click aborts with non-zero
        mock_qdrant.delete_collection.assert_not_called()


# --- Documents ---


class TestDocuments:
    def test_documents_list(self, runner, mock_resources):
        mock_db, _, _ = mock_resources

        class FakeCursor:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not hasattr(self, "_rows"):
                    self._rows = iter([
                        ("doc1", "readme.md", "/docs/readme.md", "md", 5, "completed", "2024-01-01"),
                    ])
                try:
                    return next(self._rows)
                except StopIteration:
                    raise StopAsyncIteration

        mock_db.execute = MagicMock(return_value=FakeCursor())

        result = runner.invoke(main, ["documents", "list"])
        assert result.exit_code == 0
        assert "readme.md" in result.output

    def test_documents_list_empty(self, runner, mock_resources):
        mock_db, _, _ = mock_resources

        class FakeCursor:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        mock_db.execute = MagicMock(return_value=FakeCursor())

        result = runner.invoke(main, ["documents", "list"])
        assert result.exit_code == 0
        assert "No documents" in result.output

    def test_documents_info(self, runner, mock_resources):
        mock_db, _, _ = mock_resources

        class FakeCursor:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def fetchone(self):
                return (
                    "doc1", "col1", "readme.md", "/docs/readme.md", "md",
                    "abc123", 5, "completed", None, "2024-01-01", "2024-01-02",
                )

        mock_db.execute = MagicMock(return_value=FakeCursor())

        result = runner.invoke(main, ["documents", "info", "doc1"])
        assert result.exit_code == 0
        assert "readme.md" in result.output
        assert "doc1" in result.output

    def test_documents_info_not_found(self, runner, mock_resources):
        mock_db, _, _ = mock_resources

        class FakeCursor:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def fetchone(self):
                return None

        mock_db.execute = MagicMock(return_value=FakeCursor())

        result = runner.invoke(main, ["documents", "info", "missing"])
        assert result.exit_code == 0  # prints error but doesn't sys.exit
        assert "not found" in result.output.lower()

    def test_documents_delete(self, runner, mock_resources):
        mock_db, _, mock_qdrant = mock_resources

        class FakeCursor:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def __await__(self):
                async def _self():
                    return self
                return _self().__await__()

            async def fetchone(self):
                return ("doc1", "/docs/readme.md", "my_col")

        mock_db.execute = MagicMock(return_value=FakeCursor())
        mock_db.commit = AsyncMock()
        mock_qdrant.delete_points_by_filter = AsyncMock()

        result = runner.invoke(main, ["documents", "delete", "doc1", "-y"])
        assert result.exit_code == 0
        assert "deleted" in result.output.lower()


# --- Status ---


class TestStatus:
    def test_status_all_healthy(self, runner, mock_resources):
        _, mock_ollama, mock_qdrant = mock_resources
        mock_ollama.health.return_value = True
        mock_qdrant.list_collections.return_value = []

        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "Online" in result.output

    def test_status_ollama_down(self, runner, mock_resources):
        _, mock_ollama, mock_qdrant = mock_resources
        mock_ollama.health.return_value = False
        mock_qdrant.list_collections.return_value = []

        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "Offline" in result.output

    def test_status_qdrant_down(self, runner, mock_resources):
        _, mock_ollama, mock_qdrant = mock_resources
        mock_ollama.health.return_value = True
        mock_qdrant.list_collections.side_effect = Exception("Connection refused")

        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "Offline" in result.output


# ---------- Test Hardening ----------


class TestCollectionsEdgeCases:
    def test_collections_info_not_found(self, runner, mock_resources):
        _, _, mock_qdrant = mock_resources
        from rag_kb.core.errors import QdrantCollectionError

        mock_qdrant.get_collection_info.side_effect = QdrantCollectionError(
            "Collection not found"
        )

        result = runner.invoke(main, ["collections", "info", "missing"])
        assert result.exit_code == 1
        assert "Collection not found" in result.output or "error" in result.output.lower()


class TestAskEdgeCases:
    def test_ask_collection_not_found(self, runner, mock_resources):
        with (
            patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_ret_cls,
            patch("rag_kb.retrieval.query_log.log_query", new_callable=AsyncMock),
        ):
            mock_ret = AsyncMock()
            mock_ret.search.side_effect = CollectionNotFoundError("not found")
            mock_ret_cls.return_value = mock_ret

            result = runner.invoke(main, ["ask", "test", "-c", "missing", "--no-stream"])
            assert result.exit_code == 1

    def test_ask_streaming_output_content(self, runner, mock_resources):
        response = _make_retrieval_response(1)

        async def fake_stream(*args, **kwargs):
            for token in ["The", " answer"]:
                yield token

        with (
            patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_ret_cls,
            patch("rag_kb.generation.engine.GenerationEngine") as mock_gen_cls,
            patch("rag_kb.retrieval.query_log.log_query", new_callable=AsyncMock),
        ):
            mock_ret = AsyncMock()
            mock_ret.search.return_value = response
            mock_ret_cls.return_value = mock_ret

            mock_gen = AsyncMock()
            mock_gen.answer.return_value = fake_stream()
            mock_gen_cls.return_value = mock_gen

            result = runner.invoke(main, ["ask", "what is it?"])
            assert result.exit_code == 0
            # Streaming tokens should appear in output
            assert "The" in result.output
            assert "answer" in result.output


# --- Benchmark ---


class TestBenchmark:
    def test_benchmark_happy_path(self, runner, mock_resources, tmp_path):
        """Benchmark command runs and produces a report."""
        from rag_kb.cli.benchmark import BenchmarkReport, QueryResult

        testset = tmp_path / "testset.json"
        testset.write_text('{"queries": [{"query": "test", "expected_files": ["a.md"]}]}')

        fake_report = BenchmarkReport(
            results=[
                QueryResult(
                    query="test",
                    description="",
                    expected_files=["a.md"],
                    retrieved_files=["/docs/a.md"],
                    recall_at_5=1.0,
                    recall_at_10=1.0,
                    mrr=1.0,
                    latency_ms=50.0,
                )
            ],
            avg_recall_at_5=1.0,
            avg_recall_at_10=1.0,
            avg_mrr=1.0,
            avg_latency_ms=50.0,
            total_queries=1,
        )

        with patch(
            "rag_kb.cli.benchmark.run_benchmark",
            new_callable=AsyncMock,
            return_value=fake_report,
        ):
            result = runner.invoke(main, ["benchmark", str(testset)])
            assert result.exit_code == 0
            assert "Recall@5" in result.output
            assert "MRR" in result.output

    def test_benchmark_file_not_found(self, runner):
        """Benchmark with nonexistent file fails."""
        result = runner.invoke(main, ["benchmark", "/nonexistent/testset.json"])
        assert result.exit_code == 2  # Click validation error


class TestSearchInteractive:
    def test_search_interactive_quit(self, runner, mock_resources):
        """Interactive mode exits cleanly when user types 'quit'."""
        with (
            patch("rich.prompt.Prompt.ask", return_value="quit"),
        ):
            result = runner.invoke(main, ["search", "--interactive"])
            assert result.exit_code == 0
            assert "Goodbye" in result.output

    def test_search_interactive_query(self, runner, mock_resources):
        """Interactive mode processes a query then exits on 'quit'."""
        response = _make_retrieval_response(1)
        call_count = [0]

        def fake_ask(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return "test query"
            return "quit"

        with (
            patch("rich.prompt.Prompt.ask", side_effect=fake_ask),
            patch("rag_kb.retrieval.engine.RetrievalEngine") as mock_engine_cls,
            patch("rag_kb.retrieval.query_log.log_query", new_callable=AsyncMock),
        ):
            mock_engine = AsyncMock()
            mock_engine.search.return_value = response
            mock_engine_cls.return_value = mock_engine

            result = runner.invoke(main, ["search", "--interactive"])
            assert result.exit_code == 0
            assert "Goodbye" in result.output

    def test_search_requires_query_or_interactive(self, runner, mock_resources):
        """Search with no query and no -i flag shows error."""
        result = runner.invoke(main, ["search"])
        assert result.exit_code == 1
        assert "query is required" in result.output.lower()

    def test_ingest_directory_with_progress(self, runner, mock_resources, tmp_path):
        """Verify ingest uses progress callback for directory ingestion."""
        mock_db, mock_ollama, mock_qdrant = mock_resources
        (tmp_path / "a.md").write_text("# A")

        fake_result = FakeBatchIngestionResult(
            total_files=1,
            processed=1,
            results=[
                FakeIngestionResult(
                    file_path=str(tmp_path / "a.md"),
                    status=DocumentStatus.COMPLETED,
                    chunk_count=2,
                )
            ],
        )

        with patch(
            "rag_kb.ingestion.orchestrator.ingest_directory",
            new_callable=AsyncMock,
            return_value=fake_result,
        ) as mock_fn:
            result = runner.invoke(main, ["ingest", str(tmp_path)])
            assert result.exit_code == 0
            # Verify progress_callback was passed
            call_kwargs = mock_fn.call_args.kwargs
            assert "progress_callback" in call_kwargs
            assert call_kwargs["progress_callback"] is not None


class TestIngestEdgeCases:
    def test_ingest_collection_option(self, runner, mock_resources, tmp_path):
        mock_db, mock_ollama, mock_qdrant = mock_resources
        test_file = tmp_path / "test.md"
        test_file.write_text("# Hello")

        fake_result = FakeIngestionResult(
            file_path=str(test_file),
            status=DocumentStatus.COMPLETED,
            chunk_count=1,
        )

        with patch(
            "rag_kb.ingestion.orchestrator.ingest_file",
            new_callable=AsyncMock,
            return_value=fake_result,
        ) as mock_fn:
            runner.invoke(main, ["ingest", str(test_file), "-c", "custom"])
            call_args = mock_fn.call_args
            # collection_name is the second positional arg
            assert call_args[0][1] == "custom"
