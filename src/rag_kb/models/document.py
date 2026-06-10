"""Pipeline data models for document loading and chunking."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    """Output of loaders, input to chunkers."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    file_path: str
    file_type: str
    file_hash: str
    file_size: int


class Chunk(BaseModel):
    """Output of chunkers, input to embedding pipeline."""

    content: str
    chunk_index: int
    total_chunks: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    file_path: str
    file_type: str
    file_hash: str
    token_count: int = 0
