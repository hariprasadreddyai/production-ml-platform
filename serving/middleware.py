"""Middleware: request logging and per-client rate limiting."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %d in %.1fms",
            request.method, request.url.path, response.status_code, latency_ms,
        )
        response.headers["X-Process-Time-Ms"] = f"{latency_ms:.1f}"
        return response


class TokenBucket:
    """Simple in-memory token bucket for rate limiting."""

    def __init__(self, rate_per_minute: int, burst: int):
        self.rate = rate_per_minute / 60.0
        self.burst = burst
        self.tokens = float(burst)
        self.updated_at = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.updated_at
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.updated_at = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 1000, burst: int = 50):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(requests_per_minute, burst)
        )

    async def dispatch(self, request: Request, call_next):
        client = request.client.host if request.client else "unknown"
        if not self.buckets[client].allow():
            return JSONResponse(
                {"detail": "Rate limit exceeded. Try again shortly."},
                status_code=429,
            )
        return await call_next(request)
