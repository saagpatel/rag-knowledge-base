"""Tests for the retrieval engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.models import SparseVector

from rag_kb.core.config import SearchConfig
from rag_kb.core.errors import CollectionNotFoundError
from rag_kb.ingestion.bm25 import BM25Vectorizer
from rag_kb.models.schema import SearchMode
from rag_kb.models.search import RetrievalResult, SearchResponse, SearchResult
from rag_kb.retrieval.engine import RetrievalEngine, RetrievalRequest, RetrievalResponse


def _fake_search_response(n: int = 3) -> SearchResponse:
    """Build a fake SearchResponse with n results."""
    results = []
    for i in range(n):
        results.append(SearchResult(
            id=f"point-{i}",
            score=0.9 - i * 0.1,
            payload={
                "content": f"Chunk content {i}",
                "file_path": f"/docs/file{i}.md",
                "file_type": "markdown",
                "chunk_index": i,
                "total_chunks": n,
                "token_count": 100,
            },
        ))
    return SearchResponse(results=results, total=n, search_mode="hybrid")


@pytest.fixture
def mock_ollama():
    client = AsyncMock()
    client.embed = AsyncMock(return_value=[0.1] * 768)
    return client


@pytest.fixture
def mock_qdrant():
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=True)
    client.search_dense = AsyncMock(return_value=_fake_search_response())
    client.search_sparse = AsyncMock(return_value=_fake_search_response())
    client.search_hybrid = AsyncMock(return_value=_fake_search_response())
    return client


@pytest.fixture
def search_config():
    return SearchConfig(max_top_k=50)


@pytest.fixture
def mock_bm25():
    bm25 = BM25Vectorizer.from_texts(["hello world", "test document"])
    return bm25


@pytest.fixture
def engine(mock_ollama, mock_qdrant, search_config):
    return RetrievalEngine(mock_ollama, mock_qdrant, config=search_config)


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_hybrid_search_happy_path(self, engine, mock_qdrant, mock_bm25):
        with patch("rag_kb.retrieval.engine.load_bm25", return_value=mock_bm25):
            request = RetrievalRequest(query="test query", collection="docs")
            response = await engine.search(request)

        assert isinstance(response, RetrievalResponse)
        assert len(response.results) == 3
        assert response.query == "test query"
        assert response.collection == "docs"
        assert response.mode == SearchMode.HYBRID
        mock_qdrant.search_hybrid.assert_called_once()


class TestDenseSearch:
    @pytest.mark.asyncio
    async def test_dense_search_skips_bm25(self, engine, mock_qdrant):
        with patch("rag_kb.retrieval.engine.load_bm25") as mock_load:
            request = RetrievalRequest(
                query="test", collection="docs", mode=SearchMode.DENSE
            )
            await engine.search(request)

        mock_load.assert_not_called()
        mock_qdrant.search_dense.assert_called_once()


class TestSparseSearch:
    @pytest.mark.asyncio
    async def test_sparse_search_uses_bm25(self, engine, mock_qdrant, mock_bm25):
        with patch("rag_kb.retrieval.engine.load_bm25", return_value=mock_bm25):
            request = RetrievalRequest(
                query="test", collection="docs", mode=SearchMode.SPARSE
            )
            await engine.search(request)

        mock_qdrant.search_sparse.assert_called_once()


class TestTopKClamping:
    @pytest.mark.asyncio
    async def test_top_k_capped(self, engine, mock_qdrant, search_config):
        with patch("rag_kb.retrieval.engine.load_bm25", return_value=MagicMock(
            vectorize=MagicMock(return_value=SparseVector(indices=[], values=[]))
        )):
            request = RetrievalRequest(query="test", collection="docs", top_k=200)
            await engine.search(request)

        # Should be capped to max_top_k=50
        call_kwargs = mock_qdrant.search_hybrid.call_args
        assert call_kwargs.kwargs.get("top_k", call_kwargs[1].get("top_k")) == 50


class TestCollectionValidation:
    @pytest.mark.asyncio
    async def test_collection_not_found_raises(self, engine, mock_qdrant):
        mock_qdrant.collection_exists = AsyncMock(return_value=False)
        request = RetrievalRequest(query="test", collection="missing")
        with pytest.raises(CollectionNotFoundError, match="missing"):
            await engine.search(request)


class TestResultConversion:
    @pytest.mark.asyncio
    async def test_results_are_retrieval_result(self, engine, mock_bm25):
        with patch("rag_kb.retrieval.engine.load_bm25", return_value=mock_bm25):
            request = RetrievalRequest(query="test", collection="docs")
            response = await engine.search(request)

        for r in response.results:
            assert isinstance(r, RetrievalResult)

    @pytest.mark.asyncio
    async def test_payload_extraction(self, engine, mock_bm25):
        with patch("rag_kb.retrieval.engine.load_bm25", return_value=mock_bm25):
            request = RetrievalRequest(query="test", collection="docs")
            response = await engine.search(request)

        r = response.results[0]
        assert r.content == "Chunk content 0"
        assert r.file_path == "/docs/file0.md"
        assert r.file_type == "markdown"
        assert r.chunk_index == 0


class TestFilters:
    @pytest.mark.asyncio
    async def test_filters_forwarded(self, engine, mock_qdrant):
        request = RetrievalRequest(
            query="test", collection="docs", mode=SearchMode.DENSE,
            filters={"file_type": "markdown"},
        )
        await engine.search(request)

        call_kwargs = mock_qdrant.search_dense.call_args
        assert call_kwargs.kwargs.get("filters") == {"file_type": "markdown"}


class TestLatency:
    @pytest.mark.asyncio
    async def test_latency_recorded(self, engine, mock_bm25):
        with patch("rag_kb.retrieval.engine.load_bm25", return_value=mock_bm25):
            request = RetrievalRequest(query="test", collection="docs")
            response = await engine.search(request)

        assert response.latency_ms > 0


class TestReranking:
    @pytest.mark.asyncio
    async def test_rerank_flag(self, engine, mock_bm25):
        with patch("rag_kb.retrieval.engine.load_bm25", return_value=mock_bm25), \
             patch("rag_kb.retrieval.reranker.rerank_results") as mock_rerank:
            mock_rerank.return_value = [
                RetrievalResult(
                    id="r1", score=0.95, content="reranked",
                    file_path="/a.md", file_type="markdown",
                )
            ]
            request = RetrievalRequest(query="test", collection="docs", rerank=True)
            response = await engine.search(request)

        mock_rerank.assert_called_once()
        assert response.results[0].content == "reranked"

    @pytest.mark.asyncio
    async def test_rerank_false_skips(self, engine, mock_bm25):
        with patch("rag_kb.retrieval.engine.load_bm25", return_value=mock_bm25), \
             patch("rag_kb.retrieval.reranker.rerank_results") as mock_rerank:
            request = RetrievalRequest(query="test", collection="docs", rerank=False)
            await engine.search(request)

        mock_rerank.assert_not_called()


class TestEmptyResults:
    @pytest.mark.asyncio
    async def test_empty_results(self, engine, mock_qdrant, mock_bm25):
        mock_qdrant.search_hybrid = AsyncMock(
            return_value=SearchResponse(results=[], total=0)
        )
        with patch("rag_kb.retrieval.engine.load_bm25", return_value=mock_bm25):
            request = RetrievalRequest(query="test", collection="docs")
            response = await engine.search(request)

        assert response.results == []
        assert response.total == 0


# ---------- Test Hardening ----------


class TestResponseModeField:
    @pytest.mark.asyncio
    async def test_dense_mode_response(self, engine, mock_qdrant):
        request = RetrievalRequest(
            query="test", collection="docs", mode=SearchMode.DENSE
        )
        response = await engine.search(request)
        assert response.mode == SearchMode.DENSE

    @pytest.mark.asyncio
    async def test_sparse_mode_response(self, engine, mock_qdrant, mock_bm25):
        with patch("rag_kb.retrieval.engine.load_bm25", return_value=mock_bm25):
            request = RetrievalRequest(
                query="test", collection="docs", mode=SearchMode.SPARSE
            )
            response = await engine.search(request)
        assert response.mode == SearchMode.SPARSE


class TestFiltersHybrid:
    @pytest.mark.asyncio
    async def test_filters_forwarded_hybrid(self, engine, mock_qdrant, mock_bm25):
        with patch("rag_kb.retrieval.engine.load_bm25", return_value=mock_bm25):
            request = RetrievalRequest(
                query="test", collection="docs", mode=SearchMode.HYBRID,
                filters={"file_type": "python"},
            )
            await engine.search(request)

        call_kwargs = mock_qdrant.search_hybrid.call_args
        assert call_kwargs.kwargs.get("filters") == {"file_type": "python"}


class TestTopKClampDense:
    @pytest.mark.asyncio
    async def test_top_k_clamped_dense(self, engine, mock_qdrant, search_config):
        request = RetrievalRequest(
            query="test", collection="docs", mode=SearchMode.DENSE, top_k=200
        )
        await engine.search(request)

        call_kwargs = mock_qdrant.search_dense.call_args
        assert call_kwargs.kwargs.get("top_k") == 50  # max_top_k


class TestRerankerFailure:
    @pytest.mark.asyncio
    async def test_reranker_failure_propagates(self, engine, mock_bm25):
        with patch("rag_kb.retrieval.engine.load_bm25", return_value=mock_bm25), \
             patch("rag_kb.retrieval.reranker.rerank_results") as mock_rerank:
            mock_rerank.side_effect = RuntimeError("reranker crashed")
            request = RetrievalRequest(query="test", collection="docs", rerank=True)
            with pytest.raises(RuntimeError, match="reranker crashed"):
                await engine.search(request)
