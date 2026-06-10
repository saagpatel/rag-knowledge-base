"""Pydantic models and enums for the RAG knowledge base."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryType(StrEnum):
    SEARCH = "search"
    QA = "qa"


class SearchMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class Interface(StrEnum):
    CLI = "cli"
    API = "api"
    WEB = "web"
    MCP = "mcp"


class Collection(BaseModel):
    id: str
    name: str
    description: str = ""
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "nomic-embed-text"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Document(BaseModel):
    id: str
    collection_id: str
    filename: str
    file_path: str
    file_type: str
    file_hash: str
    file_size: int
    chunk_count: int = 0
    status: DocumentStatus = DocumentStatus.PENDING
    error_message: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class IngestionJob(BaseModel):
    id: str
    collection_id: str
    status: JobStatus = JobStatus.PENDING
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class Query(BaseModel):
    id: str
    collection_id: str | None = None
    query_text: str
    query_type: QueryType = QueryType.SEARCH
    search_mode: SearchMode = SearchMode.HYBRID
    result_count: int = 0
    latency_ms: float = 0.0
    interface: Interface = Interface.API
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
