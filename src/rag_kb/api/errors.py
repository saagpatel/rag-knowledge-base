"""Exception handlers — map domain errors to HTTP envelope responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag_kb.core.errors import (
    CollectionNotFoundError,
    DocumentNotFoundError,
    GenerationError,
    IngestionError,
    LoaderError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    PromptTooLargeError,
    QdrantCollectionError,
    QdrantConnectionError,
    RAGError,
    RetrievalError,
)

from .schemas import ErrorDetail, ErrorResponse

# Domain error → (HTTP status, machine-readable code)
_ERROR_MAP: list[tuple[type[RAGError], int, str]] = [
    (OllamaConnectionError, 503, "OLLAMA_UNAVAILABLE"),
    (OllamaModelNotFoundError, 503, "OLLAMA_MODEL_NOT_FOUND"),
    (OllamaTimeoutError, 504, "OLLAMA_TIMEOUT"),
    (QdrantConnectionError, 503, "QDRANT_UNAVAILABLE"),
    (QdrantCollectionError, 500, "QDRANT_COLLECTION_ERROR"),
    (DocumentNotFoundError, 404, "DOCUMENT_NOT_FOUND"),
    (CollectionNotFoundError, 404, "COLLECTION_NOT_FOUND"),
    (LoaderError, 422, "LOADER_ERROR"),
    (PromptTooLargeError, 422, "PROMPT_TOO_LARGE"),
    (IngestionError, 500, "INGESTION_ERROR"),
    (RetrievalError, 500, "RETRIEVAL_ERROR"),
    (GenerationError, 500, "GENERATION_ERROR"),
]


def _resolve_rag_error(exc: RAGError) -> tuple[int, str]:
    """Walk the mapping (most-specific first) to find status + code."""
    for cls, status, code in _ERROR_MAP:
        if isinstance(exc, cls):
            return status, code
    return 500, "INTERNAL_ERROR"


def register_exception_handlers(app: FastAPI) -> None:
    """Register handlers for domain errors and validation errors."""

    @app.exception_handler(RAGError)
    async def _rag_error_handler(_request: Request, exc: RAGError) -> JSONResponse:
        status, code = _resolve_rag_error(exc)
        body = ErrorResponse(
            error=ErrorDetail(code=code, message=str(exc), statusCode=status),
        )
        return JSONResponse(status_code=status, content=body.model_dump(by_alias=True))

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        body = ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Request validation failed",
                statusCode=422,
                details=exc.errors(),
            ),
        )
        return JSONResponse(status_code=422, content=body.model_dump(by_alias=True))
