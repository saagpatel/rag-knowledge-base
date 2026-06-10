"""Tests for the embedding pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rag_kb.ingestion.bm25 import BM25Vectorizer
from rag_kb.ingestion.pipeline import (
    build_payload,
    build_points,
    embed_chunks,
    generate_point_id,
)
from rag_kb.models.document import Chunk


def _make_chunk(
    content: str = "test content",
    chunk_index: int = 0,
    total_chunks: int = 1,
    file_path: str = "/tmp/test.md",
    file_type: str = "markdown",
    file_hash: str = "abc123",
    token_count: int = 10,
    metadata: dict | None = None,
) -> Chunk:
    return Chunk(
        content=content,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        file_path=file_path,
        file_type=file_type,
        file_hash=file_hash,
        token_count=token_count,
        metadata=metadata or {},
    )


def _fake_vector(dim: int = 768) -> list[float]:
    return [0.1] * dim


class TestGeneratePointId:
    def test_deterministic(self):
        id1 = generate_point_id("hash1", 0)
        id2 = generate_point_id("hash1", 0)
        assert id1 == id2

    def test_different_inputs(self):
        id1 = generate_point_id("hash1", 0)
        id2 = generate_point_id("hash1", 1)
        id3 = generate_point_id("hash2", 0)
        assert id1 != id2
        assert id1 != id3
        assert id2 != id3


class TestBuildPayload:
    def test_core_fields(self):
        chunk = _make_chunk()
        payload = build_payload(chunk)
        assert payload["content"] == "test content"
        assert payload["file_path"] == "/tmp/test.md"
        assert payload["file_type"] == "markdown"
        assert payload["file_hash"] == "abc123"
        assert payload["chunk_index"] == 0
        assert payload["total_chunks"] == 1
        assert payload["token_count"] == 10

    def test_metadata_merge(self):
        chunk = _make_chunk(metadata={"heading": "Introduction", "page_number": 1})
        payload = build_payload(chunk)
        assert payload["heading"] == "Introduction"
        assert payload["page_number"] == 1

    def test_no_overwrite_core_fields(self):
        chunk = _make_chunk(metadata={"content": "EVIL", "file_path": "EVIL"})
        payload = build_payload(chunk)
        assert payload["content"] == "test content"
        assert payload["file_path"] == "/tmp/test.md"


class TestBuildPoints:
    def test_correct_count(self):
        chunks = [_make_chunk(chunk_index=i) for i in range(3)]
        dense = [_fake_vector() for _ in range(3)]
        bm25 = BM25Vectorizer.from_texts(["test content"] * 3)
        sparse = bm25.vectorize_batch(["test content"] * 3)
        points = build_points(chunks, dense, sparse)
        assert len(points) == 3

    def test_mismatch_raises(self):
        chunks = [_make_chunk()]
        dense = [_fake_vector(), _fake_vector()]
        sparse = [BM25Vectorizer().vectorize("test")]
        with pytest.raises(ValueError, match="Length mismatch"):
            build_points(chunks, dense, sparse)

    def test_vector_structure(self):
        chunk = _make_chunk()
        dense = [_fake_vector()]
        bm25 = BM25Vectorizer.from_texts(["test content"])
        sparse = bm25.vectorize_batch(["test content"])
        points = build_points([chunk], dense, sparse)
        point = points[0]
        assert "dense" in point.vector
        assert "sparse" in point.vector
        assert len(point.vector["dense"]) == 768


class TestEmbedChunks:
    @pytest.fixture
    def mock_ollama(self):
        mock = AsyncMock()
        mock.embed_batch = AsyncMock(return_value=[_fake_vector()])
        return mock

    async def test_empty_input(self, mock_ollama):
        result = await embed_chunks([], mock_ollama)
        assert result == []
        mock_ollama.embed_batch.assert_not_called()

    async def test_calls_ollama(self, mock_ollama):
        chunks = [_make_chunk(content="hello world")]
        mock_ollama.embed_batch.return_value = [_fake_vector()]
        await embed_chunks(chunks, mock_ollama)
        mock_ollama.embed_batch.assert_called_once()
        call_texts = mock_ollama.embed_batch.call_args[0][0]
        assert call_texts == ["hello world"]

    async def test_auto_builds_bm25(self, mock_ollama):
        chunks = [_make_chunk(content="hello world")]
        mock_ollama.embed_batch.return_value = [_fake_vector()]
        points = await embed_chunks(chunks, mock_ollama, bm25=None)
        assert len(points) == 1

    async def test_uses_provided_bm25(self, mock_ollama):
        bm25 = BM25Vectorizer.from_texts(["hello world", "python code"])
        chunks = [_make_chunk(content="hello world")]
        mock_ollama.embed_batch.return_value = [_fake_vector()]
        points = await embed_chunks(chunks, mock_ollama, bm25=bm25)
        assert len(points) == 1
        # Sparse vector should use terms from the provided BM25 vocabulary
        sparse = points[0].vector["sparse"]
        assert hasattr(sparse, "indices")
