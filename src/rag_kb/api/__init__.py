"""FastAPI application factory with lifespan management."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag_kb.core.config import get_config
from rag_kb.core.database import init_db
from rag_kb.core.ollama_client import OllamaClient
from rag_kb.core.qdrant_client import QdrantManager
from rag_kb.core.startup import validate_config

from .auth import AuthMiddleware
from .errors import register_exception_handlers
from .middleware import RequestLoggingMiddleware
from .rate_limit import RateLimitMiddleware
from .routes import router

logger = logging.getLogger(__name__)

WEB_DIST = Path(__file__).resolve().parent.parent.parent.parent / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Startup: open DB, Ollama, Qdrant. Shutdown: close all."""
    app.state.start_time = time.monotonic()
    app.state.background_tasks: set = set()
    app.state.db = await init_db()
    ollama = OllamaClient()
    app.state.ollama = await ollama.__aenter__()
    qdrant = QdrantManager()
    app.state.qdrant = await qdrant.__aenter__()

    await validate_config(app.state.ollama, app.state.qdrant)

    yield

    await app.state.qdrant.close()
    await app.state.ollama.close()
    await app.state.db.close()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    application = FastAPI(
        title="RAG Knowledge Base API",
        version="0.1.0",
        description="Local-only RAG system with semantic search, Q&A, and document management.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "Health", "description": "Service health checks"},
            {"name": "Ingest", "description": "Document ingestion"},
            {"name": "Search", "description": "Semantic search"},
            {"name": "Ask", "description": "AI-powered Q&A"},
            {"name": "Collections", "description": "Collection management"},
            {"name": "Documents", "description": "Document management"},
            {"name": "Analytics", "description": "Query analytics and metrics"},
            {"name": "Jobs", "description": "Background job tracking"},
        ],
    )

    application.add_middleware(RequestLoggingMiddleware)
    application.add_middleware(AuthMiddleware)
    application.add_middleware(RateLimitMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(application)
    application.include_router(router)

    # Serve the React SPA from web/dist/ when it exists (production build).
    if WEB_DIST.is_dir():
        application.mount(
            "/assets",
            StaticFiles(directory=WEB_DIST / "assets"),
            name="static-assets",
        )

        @application.get("/{path:path}", include_in_schema=False)
        async def spa_fallback(request: Request, path: str) -> FileResponse:
            """Serve index.html for all non-API routes (SPA client-side routing)."""
            # Try to serve a static file first (e.g. fonts, favicon)
            static_file = WEB_DIST / path
            if static_file.is_file() and ".." not in path:
                return FileResponse(static_file)
            return FileResponse(WEB_DIST / "index.html")

    return application


app = create_app()
