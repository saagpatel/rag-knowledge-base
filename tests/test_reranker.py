"""Tests for the cross-encoder reranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_kb.core.errors import RetrievalError
from rag_kb.models.search import RetrievalResult
from rag_kb.retrieval.reranker import clear_model_cache, rerank_results


def _make_result(score: float, content: str = "test") -> RetrievalResult:
    return RetrievalResult(
        id="r1", score=score, content=content,
        file_path="/a.md", file_type="markdown",
    )


@pytest.fixture(autouse=True)
def _reset_model_cache():
    """Ensure model cache is clean before and after each test."""
    clear_model_cache()
    yield
    clear_model_cache()


class TestRerankResults:
    def test_rerank_sorted_by_score(self):
        results = [_make_result(0.5, "low"), _make_result(0.3, "high")]
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.2, 0.9])

        with patch("rag_kb.retrieval.reranker._get_model", return_value=mock_model):
            reranked = rerank_results("query", results)

        assert reranked[0].content == "high"
        assert reranked[1].content == "low"
        assert reranked[0].score > reranked[1].score

    def test_reranked_flag_set(self):
        results = [_make_result(0.5)]
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8])

        with patch("rag_kb.retrieval.reranker._get_model", return_value=mock_model):
            reranked = rerank_results("query", results)

        assert all(r.reranked for r in reranked)

    def test_empty_results(self):
        assert rerank_results("query", []) == []

    def test_top_k_limits_output(self):
        results = [_make_result(0.1 * i) for i in range(5)]
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

        with patch("rag_kb.retrieval.reranker._get_model", return_value=mock_model):
            reranked = rerank_results("query", results, top_k=2)

        assert len(reranked) == 2


class TestModelCaching:
    def test_model_cached(self):
        mock_cross_encoder_class = MagicMock()
        mock_instance = MagicMock()
        mock_cross_encoder_class.return_value = mock_instance
        mock_instance.predict.return_value = np.array([0.5])

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(
            CrossEncoder=mock_cross_encoder_class
        )}):
            from rag_kb.retrieval import reranker
            reranker._model = None

            # Call _get_model twice
            reranker._get_model()
            reranker._get_model()

            # CrossEncoder constructor called only once
            mock_cross_encoder_class.assert_called_once()

    def test_clear_model_cache(self):
        from rag_kb.retrieval import reranker
        reranker._model = "something"
        clear_model_cache()
        assert reranker._model is None


class TestMissingDependency:
    def test_missing_dependency_raises(self):
        import rag_kb.retrieval.reranker as reranker_mod
        reranker_mod._model = None

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises((RetrievalError, ImportError)):
                reranker_mod._get_model()
