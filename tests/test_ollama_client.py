"""Unit tests for OllamaClient — all HTTP calls mocked."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from rag_kb.core.config import OllamaConfig
from rag_kb.core.errors import OllamaConnectionError, OllamaError, OllamaTimeoutError
from rag_kb.core.ollama_client import OllamaClient


@pytest.fixture
def config() -> OllamaConfig:
    return OllamaConfig(
        host="http://test:11434",
        embedding_model="nomic-embed-text",
        generation_model="mistral:7b",
        timeout=30,
        max_retries=3,
    )


@pytest.fixture
def client(config: OllamaConfig) -> OllamaClient:
    return OllamaClient(config=config)


def _mock_response(data: dict | None = None, status: int = 200, text: str = "") -> httpx.Response:
    """Build a fake httpx.Response."""
    kwargs: dict = {"status_code": status, "request": httpx.Request("POST", "http://test")}
    if data is not None:
        kwargs["json"] = data
    else:
        kwargs["text"] = text
    return httpx.Response(**kwargs)


def _fake_vector(dim: int = 768) -> list[float]:
    return [0.1] * dim


# --- embed ---


async def test_embed_single(client: OllamaClient) -> None:
    vec = _fake_vector()
    client._client = AsyncMock(spec=httpx.AsyncClient)
    client._client.request = AsyncMock(return_value=_mock_response({"embeddings": [vec]}))

    result = await client.embed("hello world")

    assert result == vec
    assert len(result) == 768
    call_kwargs = client._client.request.call_args
    body = call_kwargs.kwargs["json"]
    assert body["model"] == "nomic-embed-text"
    assert body["input"] == ["hello world"]


# --- embed_batch ---


async def test_embed_batch_single_batch(client: OllamaClient) -> None:
    vecs = [_fake_vector() for _ in range(10)]
    client._client = AsyncMock(spec=httpx.AsyncClient)
    client._client.request = AsyncMock(return_value=_mock_response({"embeddings": vecs}))

    result = await client.embed_batch([f"text_{i}" for i in range(10)], batch_size=32)

    assert len(result) == 10
    assert client._client.request.call_count == 1


async def test_embed_batch_multiple_batches(client: OllamaClient) -> None:
    client._client = AsyncMock(spec=httpx.AsyncClient)

    batch1 = [_fake_vector() for _ in range(32)]
    batch2 = [_fake_vector() for _ in range(32)]
    batch3 = [_fake_vector() for _ in range(6)]

    client._client.request = AsyncMock(
        side_effect=[
            _mock_response({"embeddings": batch1}),
            _mock_response({"embeddings": batch2}),
            _mock_response({"embeddings": batch3}),
        ]
    )

    result = await client.embed_batch(
        [f"text_{i}" for i in range(70)], batch_size=32, show_progress=False
    )

    assert len(result) == 70
    assert client._client.request.call_count == 3


async def test_embed_batch_empty(client: OllamaClient) -> None:
    client._client = AsyncMock(spec=httpx.AsyncClient)
    result = await client.embed_batch([])
    assert result == []
    client._client.request.assert_not_called()


# --- generate ---


async def test_generate_non_streaming(client: OllamaClient) -> None:
    client._client = AsyncMock(spec=httpx.AsyncClient)
    client._client.request = AsyncMock(
        return_value=_mock_response({"response": "RAG retrieves documents for context."})
    )

    result = await client.generate("What is RAG?")

    assert result == "RAG retrieves documents for context."
    body = client._client.request.call_args.kwargs["json"]
    assert body["model"] == "mistral:7b"
    assert body["stream"] is False


async def test_generate_streaming(client: OllamaClient) -> None:
    lines = [
        json.dumps({"response": "Hello", "done": False}),
        json.dumps({"response": " world", "done": False}),
        json.dumps({"response": "", "done": True}),
    ]

    mock_stream = AsyncMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=False)
    mock_stream.raise_for_status = MagicMock()

    async def fake_aiter_lines():
        for line in lines:
            yield line

    mock_stream.aiter_lines = fake_aiter_lines
    client._client = AsyncMock(spec=httpx.AsyncClient)
    client._client.stream = MagicMock(return_value=mock_stream)

    gen = await client.generate("Hi", stream=True)
    tokens = [t async for t in gen]

    assert tokens == ["Hello", " world"]


# --- chat ---


async def test_chat_non_streaming(client: OllamaClient) -> None:
    client._client = AsyncMock(spec=httpx.AsyncClient)
    client._client.request = AsyncMock(
        return_value=_mock_response({"message": {"content": "I'm an AI assistant."}})
    )

    messages = [{"role": "user", "content": "Who are you?"}]
    result = await client.chat(messages)

    assert result == "I'm an AI assistant."


async def test_chat_custom_model(client: OllamaClient) -> None:
    client._client = AsyncMock(spec=httpx.AsyncClient)
    client._client.request = AsyncMock(
        return_value=_mock_response({"message": {"content": "Hi"}})
    )

    await client.chat([{"role": "user", "content": "Hi"}], model="llama3:8b")

    body = client._client.request.call_args.kwargs["json"]
    assert body["model"] == "llama3:8b"


# --- health ---


async def test_health_success(client: OllamaClient) -> None:
    client._client = AsyncMock(spec=httpx.AsyncClient)
    client._client.get = AsyncMock(
        side_effect=[
            _mock_response({}),  # GET /
            _mock_response(  # GET /api/tags
                {"models": [{"name": "nomic-embed-text:latest"}]}
            ),
        ]
    )

    assert await client.health() is True


async def test_health_model_not_found(client: OllamaClient) -> None:
    client._client = AsyncMock(spec=httpx.AsyncClient)
    client._client.get = AsyncMock(
        side_effect=[
            _mock_response({}),
            _mock_response({"models": [{"name": "llama3:8b"}]}),
        ]
    )

    assert await client.health() is False


async def test_health_connection_error(client: OllamaClient) -> None:
    client._client = AsyncMock(spec=httpx.AsyncClient)
    client._client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    assert await client.health() is False


# --- retry logic ---


async def test_retry_then_success(client: OllamaClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag_kb.core.ollama_client.OllamaClient._backoff", AsyncMock())

    client._client = AsyncMock(spec=httpx.AsyncClient)
    client._client.request = AsyncMock(
        side_effect=[
            httpx.ConnectError("refused"),
            _mock_response({"embeddings": [[0.1]]}),
        ]
    )

    result = await client.embed("retry test")

    assert result == [0.1]
    assert client._client.request.call_count == 2


async def test_retry_exhausted_connection(
    client: OllamaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rag_kb.core.ollama_client.OllamaClient._backoff", AsyncMock())

    client._client = AsyncMock(spec=httpx.AsyncClient)
    client._client.request = AsyncMock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(OllamaConnectionError, match="Cannot reach Ollama"):
        await client.embed("fail")

    assert client._client.request.call_count == 3


async def test_retry_exhausted_timeout(
    client: OllamaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rag_kb.core.ollama_client.OllamaClient._backoff", AsyncMock())

    client._client = AsyncMock(spec=httpx.AsyncClient)
    client._client.request = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(OllamaTimeoutError, match="timed out"):
        await client.embed("fail")

    assert client._client.request.call_count == 3


async def test_no_retry_on_4xx(client: OllamaClient) -> None:
    client._client = AsyncMock(spec=httpx.AsyncClient)
    client._client.request = AsyncMock(
        return_value=_mock_response(None, status=404, text="not found")
    )

    with pytest.raises(OllamaError, match="Client error 404"):
        await client.embed("fail")

    assert client._client.request.call_count == 1


# --- context manager ---


async def test_context_manager(config: OllamaConfig) -> None:
    async with OllamaClient(config=config) as c:
        assert c._client is not None
    # After exit, client should be closed (aclose called internally by httpx)
