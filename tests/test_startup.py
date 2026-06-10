"""Tests for startup validation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from rag_kb.core.config import AppConfig, ServerConfig, SqliteConfig
from rag_kb.core.startup import validate_config


def _patch_config(strict: bool = False, sqlite_path: str = "data/rag_kb.db"):
    cfg = AppConfig(
        server=ServerConfig(strict_startup=strict),
        sqlite=SqliteConfig(path=sqlite_path),
    )
    return patch("rag_kb.core.startup.get_config", return_value=cfg)


@pytest.mark.asyncio
async def test_all_healthy_no_warnings():
    """When all services are healthy, no warnings are returned."""
    ollama = AsyncMock()
    ollama.health.return_value = True
    qdrant = AsyncMock()
    qdrant.list_collections.return_value = []

    with _patch_config():
        warnings = await validate_config(ollama, qdrant, strict=False)
    assert warnings == []


@pytest.mark.asyncio
async def test_ollama_unhealthy_lenient_returns_warning():
    """When Ollama is unhealthy in lenient mode, a warning is returned."""
    ollama = AsyncMock()
    ollama.health.return_value = False
    qdrant = AsyncMock()
    qdrant.list_collections.return_value = []

    with _patch_config():
        warnings = await validate_config(ollama, qdrant, strict=False)
    assert len(warnings) == 1
    assert "Ollama" in warnings[0]


@pytest.mark.asyncio
async def test_ollama_unhealthy_strict_raises():
    """When Ollama is unhealthy in strict mode, RuntimeError is raised."""
    ollama = AsyncMock()
    ollama.health.return_value = False
    qdrant = AsyncMock()
    qdrant.list_collections.return_value = []

    with _patch_config():
        with pytest.raises(RuntimeError, match="Ollama"):
            await validate_config(ollama, qdrant, strict=True)


@pytest.mark.asyncio
async def test_qdrant_unreachable_lenient_returns_warning():
    """When Qdrant is unreachable in lenient mode, a warning is returned."""
    ollama = AsyncMock()
    ollama.health.return_value = True
    qdrant = AsyncMock()
    qdrant.list_collections.side_effect = ConnectionError("refused")

    with _patch_config():
        warnings = await validate_config(ollama, qdrant, strict=False)
    assert len(warnings) == 1
    assert "Qdrant" in warnings[0]


@pytest.mark.asyncio
async def test_qdrant_unreachable_strict_raises():
    """When Qdrant is unreachable in strict mode, RuntimeError is raised."""
    ollama = AsyncMock()
    ollama.health.return_value = True
    qdrant = AsyncMock()
    qdrant.list_collections.side_effect = ConnectionError("refused")

    with _patch_config():
        with pytest.raises(RuntimeError, match="Qdrant"):
            await validate_config(ollama, qdrant, strict=True)
