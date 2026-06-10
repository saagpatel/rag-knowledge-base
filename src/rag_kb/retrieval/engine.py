"""Retrieval engine — embed query + search Qdrant + optional reranking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rag_kb.core.config import SearchConfig, get_config
from rag_kb.core.errors import CollectionNotFoundError
from rag_kb.ingestion.bm25_store import load_bm25
from rag_kb.models.schema import SearchMode
from rag_kb.models.search import RetrievalResult, SearchResult

if TYPE_CHECKING:
    from rag_kb.core.ollama_client import OllamaClient
    from rag_kb.core.qdrant_client import QdrantManager


@dataclass
class RetrievalRequest:
    """Parameters for a retrieval query."""

    query: str
    collection: str
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = 10
    filters: dict[str, object] | None = None
    rerank: bool = False
    bm25_data_dir: Path = field(default_factory=lambda: Path("data/bm25"))


@dataclass
class RetrievalResponse:
    """Result of a retrieval query."""

    results: list[RetrievalResult]
    query: str
    collection: str
    mode: SearchMode
    latency_ms: float
    total: int


def _to_retrieval_result(sr: SearchResult) -> RetrievalResult:
    """Extract typed fields from a raw SearchResult payload."""
    payload = sr.payload
    metadata = {
        k: v for k, v in payload.items()
        if k not in {"content", "file_path", "file_type", "chunk_index", "total_chunks"}
    }
    return RetrievalResult(
        id=sr.id,
        score=sr.score,
        content=str(payload.get("content", "")),
        file_path=str(payload.get("file_path", "")),
        file_type=str(payload.get("file_type", "")),
        chunk_index=int(payload.get("chunk_index", 0)),  # type: ignore[arg-type]
        total_chunks=int(payload.get("total_chunks", 0)),  # type: ignore[arg-type]
        metadata=metadata,
    )


class RetrievalEngine:
    """Embeds queries and searches Qdrant with optional reranking."""

    def __init__(
        self,
        ollama: OllamaClient,
        qdrant: QdrantManager,
        config: SearchConfig | None = None,
    ) -> None:
        self._ollama = ollama
        self._qdrant = qdrant
        self._config = config or get_config().search

    async def search(self, request: RetrievalRequest) -> RetrievalResponse:
        """Execute a search against Qdrant and return typed results."""
        start = time.perf_counter()

        # Clamp top_k
        top_k = min(request.top_k, self._config.max_top_k)

        # Verify collection exists
        if not await self._qdrant.collection_exists(request.collection):
            raise CollectionNotFoundError(
                f"Collection '{request.collection}' does not exist"
            )

        # Embed query for dense search
        dense_vector = await self._ollama.embed(request.query)

        # Build sparse vector if needed
        sparse_vector = None
        if request.mode in (SearchMode.SPARSE, SearchMode.HYBRID):
            bm25 = load_bm25(request.collection, request.bm25_data_dir)
            sparse_vector = bm25.vectorize(request.query)

        # Execute search
        filters = dict(request.filters) if request.filters else None

        if request.mode == SearchMode.DENSE:
            response = await self._qdrant.search_dense(
                request.collection, dense_vector, top_k=top_k, filters=filters,
            )
        elif request.mode == SearchMode.SPARSE:
            response = await self._qdrant.search_sparse(
                request.collection, sparse_vector, top_k=top_k, filters=filters,  # type: ignore[arg-type]
            )
        else:  # HYBRID
            response = await self._qdrant.search_hybrid(
                request.collection, dense_vector, sparse_vector,  # type: ignore[arg-type]
                top_k=top_k, filters=filters,
            )

        # Convert to typed results
        results = [_to_retrieval_result(sr) for sr in response.results]

        # Optional reranking
        if request.rerank and results:
            from .reranker import rerank_results

            results = rerank_results(request.query, results, top_k=top_k)

        latency_ms = (time.perf_counter() - start) * 1000

        return RetrievalResponse(
            results=results,
            query=request.query,
            collection=request.collection,
            mode=request.mode,
            latency_ms=latency_ms,
            total=len(results),
        )
