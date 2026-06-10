"""Request logging middleware — logs method, path, status, latency, request_id."""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar

from cuid2 import cuid_wrapper
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_cuid = cuid_wrapper()
logger = logging.getLogger("rag_kb.api.access")

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status code, latency, and request ID."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = _cuid()
        token = request_id_var.set(request_id)
        start = time.perf_counter()

        try:
            try:
                response = await call_next(request)
            except Exception:
                latency_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    "Request failed",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": 500,
                        "latency_ms": round(latency_ms, 2),
                        "request_id": request_id,
                    },
                )
                raise

            latency_ms = (time.perf_counter() - start) * 1000
            response.headers["X-Request-ID"] = request_id

            logger.info(
                "%s %s %d %.1fms",
                request.method,
                request.url.path,
                response.status_code,
                latency_ms,
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": round(latency_ms, 2),
                    "request_id": request_id,
                },
            )

            return response
        finally:
            request_id_var.reset(token)
