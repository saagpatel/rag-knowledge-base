"""In-memory token bucket rate limiter middleware."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from rag_kb.core.config import get_config

logger = logging.getLogger(__name__)

_EXEMPT_PATHS = {"/api/health"}


class TokenBucket:
    def __init__(self, rate: float, burst: int) -> None:
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_refill = time.monotonic()

    def consume(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    @property
    def retry_after(self) -> float:
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.rate


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self._buckets: dict[str, TokenBucket] = defaultdict(self._make_bucket)

    def _make_bucket(self) -> TokenBucket:
        config = get_config()
        rate = config.server.rate_limit_rpm / 60.0
        return TokenBucket(rate, config.server.rate_limit_burst)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        config = get_config()
        if config.server.rate_limit_rpm == 0:
            return await call_next(request)
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        bucket = self._buckets[client_ip]
        if not bucket.consume():
            retry_after = int(bucket.retry_after) + 1
            logger.warning("Rate limited: %s", client_ip)
            return _rate_limited(retry_after)

        return await call_next(request)


def _rate_limited(retry_after: int) -> JSONResponse:
    from datetime import UTC, datetime
    from uuid import uuid4

    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(retry_after)},
        content={
            "success": False,
            "error": {"code": "RATE_LIMITED", "message": "Too many requests", "statusCode": 429},
            "meta": {"request_id": str(uuid4()), "timestamp": datetime.now(UTC).isoformat()},
        },
    )
