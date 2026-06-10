"""Tests for Pydantic models."""

from __future__ import annotations

import pytest

from rag_kb.models.schema import (
    Collection,
    Document,
    DocumentStatus,
    IngestionJob,
    Query,
    SearchMode,
)


def test_collection_model():
    """Valid collection data serializes and deserializes."""
    c = Collection(id="col-1", name="test")
    assert c.id == "col-1"
    assert c.name == "test"
    data = c.model_dump()
    c2 = Collection(**data)
    assert c2.id == c.id


def test_collection_defaults():
    """Collection has correct defaults for chunk_size and chunk_overlap."""
    c = Collection(id="col-1", name="test")
    assert c.chunk_size == 512
    assert c.chunk_overlap == 50
    assert c.embedding_model == "nomic-embed-text"
    assert c.description == ""


def test_document_model():
    """All document fields validate correctly."""
    d = Document(
        id="doc-1",
        collection_id="col-1",
        filename="test.md",
        file_path="/path/test.md",
        file_type="markdown",
        file_hash="abc123",
        file_size=1024,
    )
    assert d.status == DocumentStatus.PENDING
    assert d.chunk_count == 0
    assert d.error_message is None
    assert d.metadata == {}


def test_document_status_enum():
    """Only valid statuses are accepted."""
    d = Document(
        id="doc-1",
        collection_id="col-1",
        filename="test.md",
        file_path="/path/test.md",
        file_type="markdown",
        file_hash="abc123",
        file_size=1024,
        status=DocumentStatus.COMPLETED,
    )
    assert d.status == "completed"

    with pytest.raises(ValueError):
        Document(
            id="doc-1",
            collection_id="col-1",
            filename="test.md",
            file_path="/path/test.md",
            file_type="markdown",
            file_hash="abc123",
            file_size=1024,
            status="invalid_status",
        )


def test_ingestion_job_model():
    """IngestionJob has correct defaults."""
    job = IngestionJob(id="job-1", collection_id="col-1")
    assert job.status == "pending"
    assert job.total_files == 0
    assert job.started_at is None


def test_query_model():
    """Query model validates search modes."""
    q = Query(id="q-1", query_text="test query")
    assert q.search_mode == SearchMode.HYBRID
    assert q.collection_id is None
    assert q.result_count == 0
