"""Generation engine — prompt construction + Ollama chat."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_kb.core.config import GenerationConfig, get_config
from rag_kb.models.search import RetrievalResult

from .prompt import build_messages, extract_sources

if TYPE_CHECKING:
    from rag_kb.core.ollama_client import OllamaClient


@dataclass
class GenerationResult:
    """Result of a non-streaming generation."""

    answer: str
    sources: list[dict[str, object]]
    query: str
    model: str
    latency_ms: float
    context_chunks_used: int


class GenerationEngine:
    """Builds RAG prompts and calls Ollama for answers."""

    def __init__(
        self,
        ollama: OllamaClient,
        config: GenerationConfig | None = None,
    ) -> None:
        self._ollama = ollama
        self._config = config or get_config().generation

    async def answer(
        self,
        query: str,
        results: list[RetrievalResult],
        model: str | None = None,
        stream: bool = False,
    ) -> GenerationResult | AsyncGenerator[str, None]:
        """Generate an answer from retrieval results.

        If ``stream=True``, returns an async generator yielding tokens.
        If ``stream=False``, returns a ``GenerationResult``.
        """
        selected = results[: self._config.max_context_chunks]
        messages = build_messages(query, selected, self._config)
        sources = extract_sources(selected)

        if stream:
            return await self._ollama.chat(messages, model=model, stream=True)  # type: ignore[return-value]

        start = time.perf_counter()
        answer_text = await self._ollama.chat(messages, model=model)
        latency_ms = (time.perf_counter() - start) * 1000

        return GenerationResult(
            answer=str(answer_text),
            sources=sources,
            query=query,
            model=model or self._ollama._config.generation_model,
            latency_ms=latency_ms,
            context_chunks_used=len(selected),
        )
