"""Tests for the ingestion orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from rag_kb.ingestion.orchestrator import ingest_directory, ingest_file
from rag_kb.models.schema import DocumentStatus, JobStatus


def _fake_vector(dim: int = 768) -> list[float]:
    return [0.1] * dim


@pytest.fixture
def mock_ollama():
    mock = AsyncMock()
    mock.embed_batch = AsyncMock(return_value=[_fake_vector()])
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_qdrant():
    mock = AsyncMock()
    mock.create_collection = AsyncMock()
    mock.upsert_points = AsyncMock()
    return mock


@pytest.fixture
def sample_files(tmp_dir: Path):
    """Create small test files."""
    md = tmp_dir / "readme.md"
    md.write_text("# Hello\n\nThis is a test document with enough content.")

    txt = tmp_dir / "notes.txt"
    txt.write_text("Some plain text notes for testing purposes.")

    py = tmp_dir / "example.py"
    py.write_text("def hello():\n    return 'world'\n")

    return {"md": md, "txt": txt, "py": py}


def _adjust_mock_for_chunks(mock_ollama, chunk_count: int):
    """Set embed_batch to return the right number of vectors."""
    mock_ollama.embed_batch.return_value = [_fake_vector() for _ in range(chunk_count)]


class TestIngestFile:
    async def test_happy_path(self, tmp_db, sample_files, mock_ollama, mock_qdrant):
        _adjust_mock_for_chunks(mock_ollama, 1)
        result = await ingest_file(
            sample_files["md"], "test_col", tmp_db, mock_ollama, mock_qdrant
        )
        assert result.status == DocumentStatus.COMPLETED
        assert result.chunk_count > 0
        mock_qdrant.upsert_points.assert_called_once()

        # Check SQLite record
        async with tmp_db.execute(
            "SELECT status, chunk_count FROM documents WHERE file_path = ?",
            (str(sample_files["md"].resolve()),),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == DocumentStatus.COMPLETED

    async def test_creates_collection(self, tmp_db, sample_files, mock_ollama, mock_qdrant):
        _adjust_mock_for_chunks(mock_ollama, 1)
        await ingest_file(sample_files["txt"], "new_col", tmp_db, mock_ollama, mock_qdrant)
        mock_qdrant.create_collection.assert_called()

        async with tmp_db.execute(
            "SELECT name FROM collections WHERE name = ?", ("new_col",)
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None

    async def test_skip_unchanged(self, tmp_db, sample_files, mock_ollama, mock_qdrant):
        _adjust_mock_for_chunks(mock_ollama, 1)
        # First ingest
        r1 = await ingest_file(sample_files["md"], "test_col", tmp_db, mock_ollama, mock_qdrant)
        assert r1.chunk_count > 0

        # Second ingest — same file, same hash
        r2 = await ingest_file(sample_files["md"], "test_col", tmp_db, mock_ollama, mock_qdrant)
        assert r2.status == DocumentStatus.COMPLETED
        assert r2.chunk_count == 0  # Skipped

    async def test_reprocess_changed(self, tmp_db, sample_files, mock_ollama, mock_qdrant):
        _adjust_mock_for_chunks(mock_ollama, 1)
        # First ingest
        await ingest_file(sample_files["txt"], "test_col", tmp_db, mock_ollama, mock_qdrant)

        # Modify file
        sample_files["txt"].write_text("Updated content that is different now.")
        _adjust_mock_for_chunks(mock_ollama, 1)
        r2 = await ingest_file(sample_files["txt"], "test_col", tmp_db, mock_ollama, mock_qdrant)
        assert r2.status == DocumentStatus.COMPLETED
        assert r2.chunk_count > 0

    async def test_failure_marks_failed(self, tmp_db, mock_ollama, mock_qdrant, tmp_dir):
        bad_file = tmp_dir / "missing.md"
        result = await ingest_file(
            bad_file, "test_col", tmp_db, mock_ollama, mock_qdrant
        )
        assert result.status == DocumentStatus.FAILED
        assert result.error_message is not None

    async def test_document_fields(self, tmp_db, sample_files, mock_ollama, mock_qdrant):
        _adjust_mock_for_chunks(mock_ollama, 1)
        await ingest_file(sample_files["md"], "test_col", tmp_db, mock_ollama, mock_qdrant)

        async with tmp_db.execute(
            "SELECT filename, file_type, file_hash FROM documents WHERE file_path = ?",
            (str(sample_files["md"].resolve()),),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "readme.md"  # filename
        assert row[1] == "md"  # file_type
        assert len(row[2]) > 0  # file_hash is non-empty


class TestIngestDirectory:
    async def test_all_files(self, tmp_db, sample_files, mock_ollama, mock_qdrant, tmp_dir):
        _adjust_mock_for_chunks(mock_ollama, 1)
        result = await ingest_directory(
            tmp_dir, "test_col", tmp_db, mock_ollama, mock_qdrant
        )
        assert result.total_files == 3
        assert result.processed == 3
        assert result.failed == 0

    async def test_partial_failure(self, tmp_db, sample_files, mock_ollama, mock_qdrant, tmp_dir):
        call_count = 0

        async def embed_side_effect(texts, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Ollama exploded")
            return [_fake_vector() for _ in texts]

        mock_ollama.embed_batch = AsyncMock(side_effect=embed_side_effect)
        result = await ingest_directory(
            tmp_dir, "test_col", tmp_db, mock_ollama, mock_qdrant
        )
        assert result.failed == 1
        assert result.processed == 2

    async def test_empty_directory(self, tmp_db, mock_ollama, mock_qdrant, tmp_dir):
        empty = tmp_dir / "empty"
        empty.mkdir()
        result = await ingest_directory(
            empty, "test_col", tmp_db, mock_ollama, mock_qdrant
        )
        assert result.total_files == 0
        assert result.processed == 0

    async def test_pattern_filter(self, tmp_db, sample_files, mock_ollama, mock_qdrant, tmp_dir):
        _adjust_mock_for_chunks(mock_ollama, 1)
        result = await ingest_directory(
            tmp_dir, "test_col", tmp_db, mock_ollama, mock_qdrant,
            patterns=["*.md"],
        )
        assert result.total_files == 1
        assert result.processed == 1

    async def test_job_tracking(self, tmp_db, sample_files, mock_ollama, mock_qdrant, tmp_dir):
        _adjust_mock_for_chunks(mock_ollama, 1)
        await ingest_directory(
            tmp_dir, "test_col", tmp_db, mock_ollama, mock_qdrant
        )

        async with tmp_db.execute("SELECT status, total_files FROM ingestion_jobs") as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == JobStatus.COMPLETED
        assert row[1] == 3

    async def test_skips_unchanged(self, tmp_db, sample_files, mock_ollama, mock_qdrant, tmp_dir):
        _adjust_mock_for_chunks(mock_ollama, 1)
        # First pass
        await ingest_directory(tmp_dir, "test_col", tmp_db, mock_ollama, mock_qdrant)

        # Second pass — all unchanged
        r2 = await ingest_directory(tmp_dir, "test_col", tmp_db, mock_ollama, mock_qdrant)
        assert r2.skipped == 3
        assert r2.processed == 0


# ---------- Test Hardening ----------


class TestIngestFilePython:
    async def test_python_file_produces_chunks(
        self, tmp_db, mock_ollama, mock_qdrant, tmp_dir
    ):
        py_file = tmp_dir / "example.py"
        py_file.write_text("def hello():\n    return 'world'\n\ndef foo():\n    return 'bar'\n")
        # Dynamic mock: return one vector per input text
        mock_ollama.embed_batch = AsyncMock(
            side_effect=lambda texts, **kw: [_fake_vector() for _ in texts]
        )

        result = await ingest_file(
            py_file, "test_col", tmp_db, mock_ollama, mock_qdrant
        )
        assert result.status == DocumentStatus.COMPLETED
        assert result.chunk_count > 0


class TestIngestDirectorySubdirs:
    async def test_nested_subdirectories(
        self, tmp_db, mock_ollama, mock_qdrant, tmp_dir
    ):
        sub = tmp_dir / "deep" / "nested"
        sub.mkdir(parents=True)
        (sub / "doc.md").write_text("# Nested document")
        _adjust_mock_for_chunks(mock_ollama, 1)

        result = await ingest_directory(
            tmp_dir, "test_col", tmp_db, mock_ollama, mock_qdrant
        )
        assert result.total_files >= 1
        paths = [r.file_path for r in result.results]
        assert any("nested" in p for p in paths)


class TestCustomChunkSize:
    async def test_custom_chunk_size_forwarded(
        self, tmp_db, mock_ollama, mock_qdrant, tmp_dir
    ):
        md_file = tmp_dir / "doc.md"
        md_file.write_text("# Hello\n\nSome content.")
        _adjust_mock_for_chunks(mock_ollama, 1)

        result = await ingest_file(
            md_file, "test_col", tmp_db, mock_ollama, mock_qdrant,
            chunk_size=128,
        )
        # Just verify it completes — the smaller chunk_size doesn't error
        assert result.status == DocumentStatus.COMPLETED


class TestForceReIngest:
    async def test_force_skips_hash_check(self, tmp_db, sample_files, mock_ollama, mock_qdrant):
        """Force re-ingest should process even if hash is unchanged."""
        _adjust_mock_for_chunks(mock_ollama, 1)
        mock_qdrant.delete_points_by_filter = AsyncMock()
        # First ingest
        r1 = await ingest_file(sample_files["md"], "test_col", tmp_db, mock_ollama, mock_qdrant)
        assert r1.chunk_count > 0

        # Second ingest with force — should NOT skip
        r2 = await ingest_file(
            sample_files["md"], "test_col", tmp_db, mock_ollama, mock_qdrant,
            force=True,
        )
        assert r2.status == DocumentStatus.COMPLETED
        assert r2.chunk_count > 0  # Not skipped

    async def test_force_deletes_old_chunks(self, tmp_db, sample_files, mock_ollama, mock_qdrant):
        """Force re-ingest should call delete_points_by_filter before re-inserting."""
        _adjust_mock_for_chunks(mock_ollama, 1)
        mock_qdrant.delete_points_by_filter = AsyncMock()
        # First ingest
        await ingest_file(sample_files["md"], "test_col", tmp_db, mock_ollama, mock_qdrant)

        # Second ingest with force
        await ingest_file(
            sample_files["md"], "test_col", tmp_db, mock_ollama, mock_qdrant,
            force=True,
        )
        mock_qdrant.delete_points_by_filter.assert_called_once()
        call_args = mock_qdrant.delete_points_by_filter.call_args
        assert call_args[0][0] == "test_col"
        assert "file_path" in call_args[0][1]

    async def test_non_force_skips_unchanged(self, tmp_db, sample_files, mock_ollama, mock_qdrant):
        """Without force, unchanged files are skipped (existing behavior)."""
        _adjust_mock_for_chunks(mock_ollama, 1)
        # First ingest
        await ingest_file(sample_files["md"], "test_col", tmp_db, mock_ollama, mock_qdrant)

        # Second ingest without force — should skip
        r2 = await ingest_file(
            sample_files["md"], "test_col", tmp_db, mock_ollama, mock_qdrant,
            force=False,
        )
        assert r2.chunk_count == 0  # Skipped

    async def test_force_in_directory_ingest(
        self, tmp_db, sample_files, mock_ollama, mock_qdrant, tmp_dir
    ):
        """Force flag is passed through to ingest_directory."""
        _adjust_mock_for_chunks(mock_ollama, 1)
        mock_qdrant.delete_points_by_filter = AsyncMock()
        # First pass
        await ingest_directory(tmp_dir, "test_col", tmp_db, mock_ollama, mock_qdrant)

        # Second pass with force — should reprocess
        result = await ingest_directory(
            tmp_dir, "test_col", tmp_db, mock_ollama, mock_qdrant,
            force=True,
        )
        assert result.processed == 3
        assert result.skipped == 0


class TestUpsertPayloadStructure:
    async def test_upsert_call_has_expected_fields(
        self, tmp_db, mock_ollama, mock_qdrant, tmp_dir
    ):
        md_file = tmp_dir / "payload_test.md"
        md_file.write_text("# Payload Test\n\nContent for testing payload.")
        _adjust_mock_for_chunks(mock_ollama, 1)

        await ingest_file(
            md_file, "test_col", tmp_db, mock_ollama, mock_qdrant
        )

        # Verify upsert was called with PointStruct objects containing payloads
        assert mock_qdrant.upsert_points.called
        points = mock_qdrant.upsert_points.call_args[0][1]  # second positional arg
        assert len(points) > 0
        payload = points[0].payload
        assert "content" in payload
        assert "file_path" in payload
        assert "file_type" in payload
