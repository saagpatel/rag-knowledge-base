"""FastAPI dependency helpers — extract singletons from app.state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    import aiosqlite

    from rag_kb.core.ollama_client import OllamaClient
    from rag_kb.core.qdrant_client import QdrantManager


def get_db(request: Request) -> aiosqlite.Connection:
    return request.app.state.db  # type: ignore[no-any-return]


def get_ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama  # type: ignore[no-any-return]


def get_qdrant(request: Request) -> QdrantManager:
    return request.app.state.qdrant  # type: ignore[no-any-return]


def get_start_time(request: Request) -> float:
    return request.app.state.start_time  # type: ignore[no-any-return]
