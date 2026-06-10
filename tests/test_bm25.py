"""Tests for BM25 sparse vector generator."""

from __future__ import annotations

from rag_kb.ingestion.bm25 import BM25Vectorizer, tokenize


class TestTokenize:
    def test_basic_splitting(self):
        tokens = tokenize("Hello World testing")
        assert "hello" in tokens
        assert "world" in tokens
        assert "testing" in tokens

    def test_special_chars_and_underscores(self):
        tokens = tokenize("my_variable! foo-bar baz.qux")
        assert "my_variable" in tokens
        assert "foo" in tokens
        assert "bar" in tokens
        assert "baz" in tokens
        assert "qux" in tokens

    def test_single_char_tokens_filtered(self):
        tokens = tokenize("a b c hello")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "hello" in tokens

    def test_stop_words_filtered(self):
        tokens = tokenize("the quick brown fox and the lazy dog")
        assert "the" not in tokens
        assert "and" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens


class TestFromTexts:
    def test_builds_vocabulary(self):
        v = BM25Vectorizer.from_texts(["hello world", "world python", "python rocks"])
        assert len(v.vocab) > 0
        assert all(isinstance(idx, int) for idx in v.vocab.values())
        assert all(v.idf[term] >= 0 for term in v.vocab)

    def test_max_vocab_size(self):
        # Build a corpus with many unique terms
        texts = [f"unique_term_{i} common" for i in range(100)]
        v = BM25Vectorizer.from_texts(texts, max_vocab_size=10)
        assert len(v.vocab) <= 10

    def test_empty_corpus(self):
        v = BM25Vectorizer.from_texts([])
        assert v.vocab == {}
        assert v.idf == {}
        assert v.avg_doc_len == 0.0

    def test_single_document(self):
        v = BM25Vectorizer.from_texts(["hello world python"])
        assert len(v.vocab) > 0
        assert v.avg_doc_len > 0

    def test_idf_rare_vs_common(self):
        texts = [
            "common term here",
            "common term there",
            "common rare_word special",
        ]
        v = BM25Vectorizer.from_texts(texts)
        # "common" appears in all 3 docs, "rare_word" in 1
        if "common" in v.idf and "rare_word" in v.idf:
            assert v.idf["rare_word"] > v.idf["common"]


class TestVectorize:
    def test_known_terms(self):
        v = BM25Vectorizer.from_texts(["hello world", "python code"])
        sv = v.vectorize("hello world")
        assert len(sv.indices) > 0
        assert len(sv.values) > 0
        assert len(sv.indices) == len(sv.values)
        assert all(val > 0 for val in sv.values)

    def test_unknown_terms(self):
        v = BM25Vectorizer.from_texts(["hello world"])
        sv = v.vectorize("zzzzz xxxxx")
        assert len(sv.indices) == 0
        assert len(sv.values) == 0

    def test_empty_vocab(self):
        v = BM25Vectorizer()
        sv = v.vectorize("hello world")
        assert len(sv.indices) == 0
        assert len(sv.values) == 0

    def test_batch_length(self):
        v = BM25Vectorizer.from_texts(["hello", "world", "test"])
        batch = v.vectorize_batch(["hello", "world", "test", "extra"])
        assert len(batch) == 4

    def test_deterministic(self):
        v = BM25Vectorizer.from_texts(["hello world", "python code"])
        sv1 = v.vectorize("hello world python")
        sv2 = v.vectorize("hello world python")
        assert sv1.indices == sv2.indices
        assert sv1.values == sv2.values
