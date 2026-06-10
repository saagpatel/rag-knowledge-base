"""Optional cross-encoder reranker — lazy import of sentence-transformers."""

from __future__ import annotations

from rag_kb.core.errors import RetrievalError
from rag_kb.models.search import RetrievalResult

_model = None
_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def _get_model():  # type: ignore[no-untyped-def]
    """Load the cross-encoder model on first call, cache globally."""
    global _model  # noqa: PLW0603
    if _model is not None:
        return _model

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RetrievalError(
            "sentence-transformers is required for reranking. "
            "Install with: uv pip install sentence-transformers",
            cause=exc,
        ) from exc

    _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank_results(
    query: str,
    results: list[RetrievalResult],
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Rerank results using the cross-encoder model.

    Returns a new list sorted by reranker score, with ``reranked=True``.
    """
    if not results:
        return []

    model = _get_model()
    pairs = [(query, r.content) for r in results]
    scores = model.predict(pairs)

    reranked = [
        r.model_copy(update={"score": float(s), "reranked": True})
        for r, s in zip(results, scores)
    ]
    reranked.sort(key=lambda r: r.score, reverse=True)

    if top_k is not None:
        reranked = reranked[:top_k]

    return reranked


def clear_model_cache() -> None:
    """Reset the cached model (for test teardown)."""
    global _model  # noqa: PLW0603
    _model = None
