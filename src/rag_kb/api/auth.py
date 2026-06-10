"""API key authentication middleware — Bearer token validation."""

from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from rag_kb.core.config import get_config

logger = logging.getLogger(__name__)

_EXEMPT_PATHS = {"/api/health", "/api/docs", "/api/openapi.json"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        config = get_config()
        api_key = config.server.api_key

        if not api_key:
            return await call_next(request)

        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return _unauthorized("Missing or invalid Authorization header")

        if auth_header[7:] != api_key:
            logger.warning("Auth failure from %s", request.client.host if request.client else "unknown")
            return _unauthorized("Invalid API key")

        return await call_next(request)


def _unauthorized(message: str) -> JSONResponse:
    from datetime import UTC, datetime
    from uuid import uuid4

    return JSONResponse(
        status_code=401,
        content={
            "success": False,
            "error": {"code": "UNAUTHORIZED", "message": message, "statusCode": 401},
            "meta": {"request_id": str(uuid4()), "timestamp": datetime.now(UTC).isoformat()},
        },
    )
