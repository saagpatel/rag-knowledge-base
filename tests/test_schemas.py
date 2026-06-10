"""Tests for Pydantic schemas in api/schemas.py."""

from datetime import datetime
from uuid import UUID

from rag_kb.api.schemas import (
    DocumentInfo,
    ErrorDetail,
    ErrorResponse,
    Meta,
    StatsData,
    SuccessResponse,
)


def test_meta_generates_uuid_request_id():
    meta = Meta()
    UUID(meta.request_id)  # raises if not valid UUID


def test_meta_generates_iso_timestamp():
    meta = Meta()
    datetime.fromisoformat(meta.timestamp)  # raises if not valid ISO


def test_success_response_wraps_data():
    resp = SuccessResponse(data={"x": 1})
    assert resp.success is True
    assert resp.data == {"x": 1}
    assert resp.meta.request_id  # non-empty


def test_error_response_has_error():
    detail = ErrorDetail(code="TEST_ERROR", message="Something went wrong", statusCode=400)
    resp = ErrorResponse(error=detail)
    assert resp.success is False
    assert resp.error.code == "TEST_ERROR"
    assert resp.error.message == "Something went wrong"


def test_document_info_roundtrip():
    doc = DocumentInfo(
        id="doc-1",
        collection_id="default",
        filename="readme.md",
        file_path="/tmp/readme.md",
        file_type="markdown",
        file_hash="abc123",
        chunk_count=5,
        status="completed",
        error_message=None,
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
    )
    dumped = doc.model_dump()
    assert dumped["id"] == "doc-1"
    assert dumped["filename"] == "readme.md"
    assert dumped["chunk_count"] == 5
    assert dumped["error_message"] is None


def test_stats_data_accepts_empty_dicts():
    stats = StatsData(
        total_queries=0,
        avg_latency_ms=0.0,
        queries_by_interface={},
        queries_by_type={},
        top_collections=[],
        period_days=7,
    )
    assert stats.total_queries == 0
    assert stats.queries_by_interface == {}
