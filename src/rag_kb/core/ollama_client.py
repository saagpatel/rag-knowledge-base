"""Async Ollama HTTP client — embeddings, generation, health checks with retry."""

from __future__ import annotations

import random
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from rag_kb.core.cache import EmbeddingCache
from rag_kb.core.config import OllamaConfig, get_config
from rag_kb.core.errors import (
    OllamaConnectionError,
    OllamaError,
    OllamaTimeoutError,
)


class OllamaClient:
    """Async wrapper around Ollama's HTTP API.

    Usage::

        async with OllamaClient() as client:
            vec = await client.embed("hello world")
            answer = await client.generate("Explain RAG in one sentence.")
    """

    def __init__(
        self,
        config: OllamaConfig | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self._config = config or get_config().ollama
        self._client = httpx.AsyncClient(
            base_url=self._config.host,
            timeout=httpx.Timeout(self._config.timeout, connect=10.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        # Embedding cache setup
        if cache is not None:
            self._cache: EmbeddingCache | None = cache
        else:
            try:
                cache_config = get_config().cache
                if cache_config.enabled:
                    self._cache = EmbeddingCache(max_size=cache_config.embedding_cache_size)
                else:
                    self._cache = None
            except Exception:
                self._cache = None

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    # --- Embeddings ---

    async def embed(self, text: str) -> list[float]:
        """Embed a single text, returning its vector."""
        if self._cache is not None:
            cached = self._cache.get(text)
            if cached is not None:
                return cached

        data = await self._request_with_retry(
            "POST",
            "/api/embed",
            json={"model": self._config.embedding_model, "input": [text]},
        )
        result: list[float] = data["embeddings"][0]

        if self._cache is not None:
            self._cache.put(text, result)

        return result

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Embed multiple texts in batches."""
        if not texts:
            return []

        embeddings: list[list[float]] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        use_progress = show_progress and total_batches > 1

        iterator: Any
        if use_progress:
            from tqdm import tqdm

            iterator = tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch")
        else:
            iterator = range(0, len(texts), batch_size)

        for start in iterator:
            batch = texts[start : start + batch_size]
            data = await self._request_with_retry(
                "POST",
                "/api/embed",
                json={"model": self._config.embedding_model, "input": batch},
            )
            embeddings.extend(data["embeddings"])

        return embeddings

    # --- Generation ---

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """Generate a completion. Returns full string or async token generator."""
        model = model or self._config.generation_model
        body = {"model": model, "prompt": prompt, "stream": stream}

        if stream:
            return self._stream_response("/api/generate", body, token_key="response")

        data = await self._request_with_retry("POST", "/api/generate", json=body)
        return data["response"]  # type: ignore[no-any-return]

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """Chat completion. Returns full string or async token generator."""
        model = model or self._config.generation_model
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}

        if stream:
            return self._stream_response(
                "/api/chat", body, token_key="message", nested_key="content"
            )

        data = await self._request_with_retry("POST", "/api/chat", json=body)
        return data["message"]["content"]  # type: ignore[no-any-return]

    # --- Health ---

    async def health(self) -> bool:
        """Check Ollama liveness and embedding model availability."""
        try:
            await self._client.get("/")
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            target = self._config.embedding_model.removesuffix(":latest")
            for m in models:
                name = m.get("name", "").removesuffix(":latest")
                if name == target:
                    return True
            return False
        except (httpx.HTTPError, Exception):
            return False

    # --- Internal helpers ---

    async def _request_with_retry(
        self, method: str, path: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Execute an HTTP request with exponential backoff retry."""
        last_exc: Exception | None = None
        max_retries = self._config.max_retries

        for attempt in range(1, max_retries + 1):
            try:
                resp = await self._client.request(method, path, **kwargs)
                if resp.status_code >= 500:
                    last_exc = OllamaError(
                        f"Server error {resp.status_code}: {resp.text}"
                    )
                    if attempt < max_retries:
                        await self._backoff(attempt)
                        continue
                    break
                if resp.status_code >= 400:
                    raise OllamaError(
                        f"Client error {resp.status_code}: {resp.text}"
                    )
                return resp.json()  # type: ignore[no-any-return]

            except httpx.ConnectError as exc:
                last_exc = exc
                if attempt < max_retries:
                    await self._backoff(attempt)
                    continue

            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < max_retries:
                    await self._backoff(attempt)
                    continue

            except OllamaError:
                raise

            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < max_retries:
                    await self._backoff(attempt)
                    continue

        # All retries exhausted
        if isinstance(last_exc, httpx.TimeoutException):
            raise OllamaTimeoutError(
                f"Ollama request timed out after {max_retries} attempts", cause=last_exc
            )
        if isinstance(last_exc, httpx.ConnectError):
            raise OllamaConnectionError(
                f"Cannot reach Ollama after {max_retries} attempts", cause=last_exc
            )
        raise OllamaError(
            f"Ollama request failed after {max_retries} attempts: {last_exc}",
            cause=last_exc,
        )

    @staticmethod
    async def _backoff(attempt: int) -> None:
        """Exponential backoff with +-25% jitter."""
        import asyncio

        base = 2 ** (attempt - 1)
        jitter = base * random.uniform(-0.25, 0.25)
        await asyncio.sleep(base + jitter)

    async def _stream_response(
        self,
        path: str,
        body: dict[str, Any],
        token_key: str,
        nested_key: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream NDJSON responses, yielding tokens."""
        async with self._client.stream("POST", path, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                import json

                chunk = json.loads(line)
                if chunk.get("done"):
                    break
                token = chunk.get(token_key, "")
                if nested_key and isinstance(token, dict):
                    token = token.get(nested_key, "")
                if token:
                    yield token
