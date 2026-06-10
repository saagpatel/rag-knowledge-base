"""Search result models returned by retrieval clients."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single search result from Qdrant."""

    id: str | int
    score: float
    payload: dict[str, object] = Field(default_factory=dict)
    vector: list[float] | None = None


class SearchResponse(BaseModel):
    """Aggregated search response wrapping multiple results."""

    results: list[SearchResult] = Field(default_factory=list)
    total: int = 0
    search_mode: str = "dense"


class RetrievalResult(BaseModel):
    """Typed search result for consumption by generation engine and API."""

    id: str | int
    score: float
    content: str
    file_path: str
    file_type: str
    chunk_index: int = 0
    total_chunks: int = 0
    metadata: dict[str, object] = Field(default_factory=dict)
    reranked: bool = False
