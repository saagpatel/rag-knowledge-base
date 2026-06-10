"""Tests for BM25 vocabulary persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_kb.core.errors import BM25StoreError
from rag_kb.ingestion.bm25 import BM25Vectorizer
from rag_kb.ingestion.bm25_store import bm25_exists, load_bm25, save_bm25


@pytest.fixture
def sample_vectorizer() -> BM25Vectorizer:
    """Build a BM25 vectorizer from sample texts."""
    return BM25Vectorizer.from_texts([
        "Python is a great programming language",
        "Machine learning uses Python extensively",
        "Vector databases store embeddings efficiently",
    ])


class TestSaveBM25:
    def test_save_creates_file(self, tmp_dir: Path, sample_vectorizer: BM25Vectorizer):
        path = save_bm25(sample_vectorizer, "test_collection", data_dir=tmp_dir)
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_creates_parent_dirs(self, tmp_dir: Path, sample_vectorizer: BM25Vectorizer):
        nested = tmp_dir / "nested" / "bm25"
        save_bm25(sample_vectorizer, "docs", data_dir=nested)
        assert (nested / "docs.json").exists()


class TestLoadBM25:
    def test_load_roundtrip(self, tmp_dir: Path, sample_vectorizer: BM25Vectorizer):
        save_bm25(sample_vectorizer, "roundtrip", data_dir=tmp_dir)
        loaded = load_bm25("roundtrip", data_dir=tmp_dir)

        assert loaded.vocab == sample_vectorizer.vocab
        assert loaded.idf == sample_vectorizer.idf
        assert loaded.avg_doc_len == pytest.approx(sample_vectorizer.avg_doc_len)
        assert loaded.k1 == sample_vectorizer.k1
        assert loaded.b == sample_vectorizer.b

    def test_load_missing_raises(self, tmp_dir: Path):
        with pytest.raises(BM25StoreError, match="not found"):
            load_bm25("nonexistent", data_dir=tmp_dir)

    def test_load_corrupt_raises(self, tmp_dir: Path):
        corrupt_file = tmp_dir / "corrupt.json"
        corrupt_file.write_text("not valid json {{{")
        with pytest.raises(BM25StoreError, match="Corrupt"):
            load_bm25("corrupt", data_dir=tmp_dir)


class TestBM25Exists:
    def test_exists_true(self, tmp_dir: Path, sample_vectorizer: BM25Vectorizer):
        save_bm25(sample_vectorizer, "exists_test", data_dir=tmp_dir)
        assert bm25_exists("exists_test", data_dir=tmp_dir) is True

    def test_exists_false(self, tmp_dir: Path):
        assert bm25_exists("missing", data_dir=tmp_dir) is False


class TestSanitization:
    def test_collection_name_sanitized(self, tmp_dir: Path, sample_vectorizer: BM25Vectorizer):
        save_bm25(sample_vectorizer, "my/collection/../evil", data_dir=tmp_dir)
        # Should not escape the data_dir — file is inside tmp_dir
        files = list(tmp_dir.glob("*.json"))
        assert len(files) == 1
        # No path traversal
        assert tmp_dir in files[0].parents or files[0].parent == tmp_dir


class TestRoundtripVectorization:
    def test_vectorizer_works_after_roundtrip(
        self, tmp_dir: Path, sample_vectorizer: BM25Vectorizer
    ):
        save_bm25(sample_vectorizer, "functional", data_dir=tmp_dir)
        loaded = load_bm25("functional", data_dir=tmp_dir)

        original = sample_vectorizer.vectorize("Python programming language")
        restored = loaded.vectorize("Python programming language")

        assert original.indices == restored.indices
        assert original.values == pytest.approx(restored.values)
