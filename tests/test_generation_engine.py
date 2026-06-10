"""Tests for the generation engine."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rag_kb.core.config import GenerationConfig
from rag_kb.generation.engine import GenerationEngine, GenerationResult
from rag_kb.models.search import RetrievalResult


def _make_result(index: int = 0, total: int = 5) -> RetrievalResult:
    return RetrievalResult(
        id=f"point-{index}",
        score=0.9 - index * 0.05,
        content=f"Content of chunk {index}",
        file_path=f"/docs/file{index}.md",
        file_type="markdown",
        chunk_index=index,
        total_chunks=total,
    )


@pytest.fixture
def mock_ollama():
    client = AsyncMock()
    client.chat = AsyncMock(return_value="This is the answer.")
    # Mock _config for model name access
    config_mock = type("Config", (), {"generation_model": "mistral:7b"})()
    client._config = config_mock
    return client


@pytest.fixture
def gen_config():
    return GenerationConfig(max_context_chunks=5)


@pytest.fixture
def engine(mock_ollama, gen_config):
    return GenerationEngine(mock_ollama, config=gen_config)


class TestNonStreamingAnswer:
    @pytest.mark.asyncio
    async def test_answer_non_streaming(self, engine):
        results = [_make_result(0), _make_result(1)]
        result = await engine.answer("What is RAG?", results)

        assert isinstance(result, GenerationResult)
        assert result.answer == "This is the answer."
        assert result.query == "What is RAG?"
        assert len(result.sources) == 2
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_answer_calls_chat(self, engine, mock_ollama):
        results = [_make_result(0)]
        await engine.answer("query", results)

        mock_ollama.chat.assert_called_once()
        call_args = mock_ollama.chat.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_custom_model_forwarded(self, engine, mock_ollama):
        await engine.answer("query", [_make_result(0)], model="llama3:8b")

        call_kwargs = mock_ollama.chat.call_args
        assert call_kwargs.kwargs.get("model") == "llama3:8b"

    @pytest.mark.asyncio
    async def test_context_chunks_used(self, engine):
        results = [_make_result(i) for i in range(3)]
        result = await engine.answer("query", results)
        assert result.context_chunks_used == 3

    @pytest.mark.asyncio
    async def test_max_context_limits_sources(self, engine, gen_config):
        results = [_make_result(i, 15) for i in range(15)]
        result = await engine.answer("query", results)
        assert result.context_chunks_used == gen_config.max_context_chunks
        assert len(result.sources) == gen_config.max_context_chunks

    @pytest.mark.asyncio
    async def test_empty_results_works(self, engine):
        result = await engine.answer("query", [])
        assert isinstance(result, GenerationResult)
        assert result.context_chunks_used == 0


class TestStreamingAnswer:
    @pytest.mark.asyncio
    async def test_streaming_returns_generator(self, engine, mock_ollama):
        async def _fake_stream():
            yield "Hello"
            yield " world"

        mock_ollama.chat = AsyncMock(return_value=_fake_stream())
        gen = await engine.answer("query", [_make_result(0)], stream=True)

        # Should be an async generator
        tokens = []
        async for token in gen:
            tokens.append(token)
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_streaming_yields_tokens(self, engine, mock_ollama):
        async def _fake_stream():
            for t in ["The", " answer", " is", " 42"]:
                yield t

        mock_ollama.chat = AsyncMock(return_value=_fake_stream())
        gen = await engine.answer("query", [_make_result(0)], stream=True)

        tokens = [t async for t in gen]
        assert len(tokens) == 4
        assert "".join(tokens) == "The answer is 42"


# ---------- Test Hardening ----------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_answer_empty_results_produces_answer(self, engine):
        """answer() with empty context list still produces an answer."""
        result = await engine.answer("What is this?", [])
        assert isinstance(result, GenerationResult)
        assert result.answer == "This is the answer."
        assert result.context_chunks_used == 0

    @pytest.mark.asyncio
    async def test_answer_ollama_connection_error(self, engine, mock_ollama):
        """OllamaConnectionError during generation propagates."""
        from rag_kb.core.errors import OllamaConnectionError

        mock_ollama.chat = AsyncMock(
            side_effect=OllamaConnectionError("Cannot reach Ollama")
        )
        with pytest.raises(OllamaConnectionError):
            await engine.answer("query", [_make_result(0)])
