"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from typing import Any, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

T = TypeVar("T")


# --- Envelope ---


class Meta(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=lambda: _utcnow_iso())


def _utcnow_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


class SuccessResponse(BaseModel, Generic[T]):  # noqa: UP046
    success: bool = True
    data: T
    meta: Meta = Field(default_factory=Meta)


class ErrorDetail(BaseModel):
    code: str
    message: str
    status_code: int = Field(alias="statusCode")
    details: Any | None = None

    model_config = {"populate_by_name": True}


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail
    meta: Meta = Field(default_factory=Meta)


# --- Health ---


class ServiceCheck(BaseModel):
    status: str
    detail: str | None = None


class HealthData(BaseModel):
    status: str
    ollama: ServiceCheck
    qdrant: ServiceCheck
    sqlite: ServiceCheck
    uptime_seconds: float
    version: str


# --- Ingest ---


class IngestRequest(BaseModel):
    path: str = Field(..., json_schema_extra={"examples": ["/path/to/docs", "/path/to/file.md"]})
    collection: str = "default"
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    patterns: list[str] | None = None
    force: bool = False


class FileIngestResult(BaseModel):
    file_path: str
    status: str
    chunk_count: int
    error_message: str | None = None


class IngestData(BaseModel):
    total_files: int
    processed: int
    failed: int
    skipped: int
    results: list[FileIngestResult]


# --- Search ---


class SearchRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"examples": ["How does RAG work?"]})
    collection: str = "default"
    mode: str = "hybrid"
    top_k: int = 10
    rerank: bool = False
    filters: dict[str, Any] | None = Field(
        None,
        json_schema_extra={"examples": [{"file_type": "markdown"}, {"file_path": "/docs/guide.md"}]},
    )


class SearchResultItem(BaseModel):
    id: str | int
    score: float
    content: str
    file_path: str
    file_type: str
    chunk_index: int
    total_chunks: int
    reranked: bool


class SearchData(BaseModel):
    results: list[SearchResultItem]
    total: int
    query: str
    collection: str
    mode: str
    latency_ms: float


# --- Ask ---


class AskRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"examples": ["Explain the retrieval pipeline"]})
    collection: str = "default"
    mode: str = "hybrid"
    top_k: int = 5
    model: str | None = None
    stream: bool = False


class SourceItem(BaseModel):
    file_path: str
    score: float
    chunk_index: int
    total_chunks: int
    file_type: str


class AskData(BaseModel):
    answer: str
    sources: list[SourceItem]
    query: str
    model: str
    latency_ms: float
    context_chunks_used: int


# --- Collections ---


class CollectionInfo(BaseModel):
    name: str
    points_count: int
    vectors_count: int
    status: str
    description: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    embedding_model: str | None = None


class UpdateCollectionRequest(BaseModel):
    description: str | None = None
    chunk_size: int | None = Field(None, ge=50, le=2048)
    chunk_overlap: int | None = Field(None, ge=0, le=200)
    embedding_model: str | None = None


class CreateCollectionRequest(BaseModel):
    name: str
    description: str | None = None
    chunk_size: int | None = Field(None, ge=50, le=2048)
    chunk_overlap: int | None = Field(None, ge=0, le=200)
    embedding_model: str | None = None


# --- Documents ---


class DocumentInfo(BaseModel):
    id: str
    collection_id: str
    filename: str
    file_path: str
    file_type: str
    file_hash: str
    chunk_count: int
    status: str
    error_message: str | None = None
    created_at: str
    updated_at: str


class DocumentListData(BaseModel):
    documents: list[DocumentInfo]
    total: int
    collection: str | None = None


# --- Analytics ---


class QueryRecord(BaseModel):
    id: str
    query_text: str
    query_type: str
    search_mode: str
    result_count: int
    latency_ms: float
    interface: str
    created_at: str


class QueryListData(BaseModel):
    queries: list[QueryRecord]
    total: int


class StatsData(BaseModel):
    total_queries: int
    avg_latency_ms: float
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    queries_by_interface: dict[str, int]
    queries_by_type: dict[str, int]
    top_collections: list[dict[str, Any]]
    period_days: int


# --- Jobs ---


class JobData(BaseModel):
    job_id: str
    status: str
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    started_at: str | None = None
    completed_at: str | None = None


# --- Metrics ---


class MetricsData(BaseModel):
    total_queries: int
    latency_p50: float
    latency_p95: float
    latency_p99: float
    cache_hit_rate: float
    cache_size: int
    active_jobs: int
