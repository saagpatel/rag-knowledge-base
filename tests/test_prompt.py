"""Tests for RAG prompt builder."""

from __future__ import annotations

import pytest

from rag_kb.core.config import GenerationConfig
from rag_kb.core.errors import PromptTooLargeError
from rag_kb.generation.prompt import build_messages, extract_sources
from rag_kb.models.search import RetrievalResult


def _make_result(index: int = 0, total: int = 5) -> RetrievalResult:
    """Create a sample RetrievalResult."""
    return RetrievalResult(
        id=f"point-{index}",
        score=0.9 - index * 0.05,
        content=f"Content of chunk {index}",
        file_path=f"/docs/file{index}.md",
        file_type="markdown",
        chunk_index=index,
        total_chunks=total,
    )


class TestBuildMessages:
    def test_returns_system_and_user_messages(self):
        results = [_make_result(0)]
        messages = build_messages("What is RAG?", results)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_context_numbered_sources(self):
        results = [_make_result(0), _make_result(1)]
        messages = build_messages("query", results)
        user_msg = messages[1]["content"]
        assert "[Source 1:" in user_msg
        assert "[Source 2:" in user_msg

    def test_max_context_truncation(self):
        results = [_make_result(i, 15) for i in range(15)]
        config = GenerationConfig(max_context_chunks=5)
        messages = build_messages("query", results, config=config)
        user_msg = messages[1]["content"]
        assert "[Source 5:" in user_msg
        assert "[Source 6:" not in user_msg

    def test_empty_results_fallback(self):
        messages = build_messages("query", [])
        user_msg = messages[1]["content"]
        assert "No relevant context found" in user_msg

    def test_query_in_user_message(self):
        messages = build_messages("What is semantic search?", [_make_result(0)])
        user_msg = messages[1]["content"]
        assert "Question: What is semantic search?" in user_msg

    def test_empty_query_raises(self):
        with pytest.raises(PromptTooLargeError, match="empty"):
            build_messages("", [_make_result(0)])

    def test_whitespace_query_raises(self):
        with pytest.raises(PromptTooLargeError, match="empty"):
            build_messages("   ", [_make_result(0)])


class TestExtractSources:
    def test_extract_sources_fields(self):
        results = [_make_result(0), _make_result(1)]
        sources = extract_sources(results)
        assert len(sources) == 2
        assert sources[0]["file_path"] == "/docs/file0.md"
        assert sources[0]["score"] == pytest.approx(0.9)
        assert sources[0]["chunk_index"] == 0

    def test_extract_sources_empty(self):
        assert extract_sources([]) == []
