"""Startup validation — check Ollama, Qdrant, SQLite before serving."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rag_kb.core.config import get_config

if TYPE_CHECKING:
    from rag_kb.core.ollama_client import OllamaClient
    from rag_kb.core.qdrant_client import QdrantManager

logger = logging.getLogger(__name__)


async def validate_config(
    ollama: OllamaClient,
    qdrant: QdrantManager,
    strict: bool | None = None,
) -> list[str]:
    config = get_config()
    if strict is None:
        strict = config.server.strict_startup

    warnings: list[str] = []

    # SQLite path
    sqlite_path = Path(config.sqlite.path)
    parent = sqlite_path.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            msg = f"SQLite directory not writable: {parent} ({exc})"
            if strict:
                raise RuntimeError(msg)
            warnings.append(msg)
            logger.warning(msg)

    # Ollama
    try:
        if await ollama.health():
            logger.info("Ollama is reachable")
        else:
            msg = "Ollama health check returned unhealthy"
            if strict:
                raise RuntimeError(msg)
            warnings.append(msg)
            logger.warning(msg)
    except RuntimeError:
        raise
    except Exception as exc:
        msg = f"Ollama unreachable: {exc}"
        if strict:
            raise RuntimeError(msg)
        warnings.append(msg)
        logger.warning(msg)

    # Qdrant
    try:
        await qdrant.list_collections()
        logger.info("Qdrant is reachable")
    except Exception as exc:
        msg = f"Qdrant unreachable: {exc}"
        if strict:
            raise RuntimeError(msg)
        warnings.append(msg)
        logger.warning(msg)

    return warnings
