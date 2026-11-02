"""
Rate limiting middleware.

Requests are counted in fixed windows per *verified* client identity:

- a valid ``X-API-Key`` gets its own bucket with that key's ``rate_limit``,
- a valid Bearer JWT shares the bucket of the API key it was issued for,
- anything else (no credentials, an unknown key, a bad token) is counted per
  client IP with ``NLP_API__ANONYMOUS_RATE_LIMIT``, so made-up keys cannot buy
  extra budget or a fresh bucket.

Counters live in Redis when ``NLP_REDIS_URL`` is reachable (shared across
replicas) and otherwise in a bounded in-process store. Over-limit requests get a
429 response with ``Retry-After`` and ``X-RateLimit-*`` headers in the standard
error envelope.
"""

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Dict, Iterable, Optional, Tuple

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from utils.logger import get_logger
from utils.metrics import record_rate_limit_hit

logger = get_logger(__name__)

# Liveness, readiness, metrics and the API docs are never rate limited.
EXEMPT_PATHS = frozenset(
    {"/health", "/ready", "/metrics", "/", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
)
MAX_MEMORY_BUCKETS = 10_000
REDIS_TIMEOUT_SECONDS = 0.25
REDIS_RETRY_SECONDS = 30.0


@dataclass(frozen=True)
class ClientIdentity:
    """Who a request is billed to."""

    bucket: str
    limit: int
    kind: str  # "api_key", "jwt" or "anonymous"


def _api_settings():
    from config.settings import get_settings

    return get_settings().api


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def identify_client(request: Request) -> ClientIdentity:
    """Resolve the rate-limit bucket from verified credentials only (no auth metrics recorded)."""
    from .auth import jwt_subject, lookup_api_key

    api = _api_settings()
    kind = "api_key"
    api_key = request.headers.get("X-API-Key")
    user = lookup_api_key(api_key) if api_key else None

    if user is None:
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            subject = jwt_subject(token.strip())
            user = lookup_api_key(subject) if subject else None
            kind = "jwt"

    if user is not None:
        digest = sha256(user.api_key.encode("utf-8")).hexdigest()[:24]
        return ClientIdentity(f"key:{digest}", user.rate_limit or api.default_rate_limit, kind)
    return ClientIdentity(f"ip:{_client_ip(request)}", api.anonymous_rate_limit, "anonymous")


class MemoryWindowStore:
    """Fixed-window counters in process memory, bounded to `max_buckets` entries."""

    def __init__(self, max_buckets: int = MAX_MEMORY_BUCKETS):
        self.max_buckets = max_buckets
        self._lock = threading.Lock()
        self._counts: "OrderedDict[str, Tuple[int, int]]" = OrderedDict()

    def hit(self, bucket: str, window: int) -> int:
        with self._lock:
            previous = self._counts.pop(bucket, None)
            count = previous[1] + 1 if previous and previous[0] == window else 1
            self._counts[bucket] = (window, count)
            if len(self._counts) > self.max_buckets:
                self._prune(window)
            return count

    def peek(self, bucket: str, window: int) -> int:
        with self._lock:
            entry = self._counts.get(bucket)
            return entry[1] if entry and entry[0] == window else 0

    def _prune(self, window: int) -> None:
        for stale in [key for key, (bucket_window, _) in self._counts.items() if bucket_window != window]:
            del self._counts[stale]
        while len(self._counts) > self.max_buckets:
            self._counts.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._counts.clear()

    def __len__(self) -> int:
        return len(self._counts)


class RedisWindowStore:
    """
    Fixed-window counters in Redis.

    Uses short socket timeouts and, after a failure, stops trying for
    REDIS_RETRY_SECONDS so an unreachable Redis costs at most one short
    timeout per retry period instead of a connection attempt per request.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._client = None
        self._url: Optional[str] = None
        self._retry_at = 0.0

    @staticmethod
    def _configured_url() -> Optional[str]:
        from config.settings import get_settings

        return get_settings().redis_url or None

    def _mark_failed(self, error: Exception) -> None:
        if self._client is not None or self._retry_at == 0.0:
            logger.warning(f"Redis rate limiting unavailable, using in-memory counters: {error}")
        self._client = None
        self._retry_at = time.monotonic() + REDIS_RETRY_SECONDS

    def client(self):
        """A connected client, or None when Redis is unset, failing, or in its retry back-off."""
        url = self._configured_url()
        if not url:
            return None
        with self._lock:
            if self._client is not None and self._url == url:
                return self._client
            if time.monotonic() < self._retry_at:
                return None
            try:
                import redis

                candidate = redis.Redis.from_url(
                    url,
                    socket_connect_timeout=REDIS_TIMEOUT_SECONDS,
                    socket_timeout=REDIS_TIMEOUT_SECONDS,
                    decode_responses=True,
                )
                candidate.ping()
            except Exception as error:
                self._mark_failed(error)
                return None
            self._client, self._url, self._retry_at = candidate, url, 0.0
            return candidate

    def hit(self, bucket: str, window: int, window_seconds: int) -> Optional[int]:
        client = self.client()
        if client is None:
            return None
        key = f"nlp-rate-limit:{sha256(bucket.encode('utf-8')).hexdigest()}:{window}"
        try:
            pipe = client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window_seconds + 5)
            count, _ = pipe.execute()
            return int(count)
        except Exception as error:
            with self._lock:
                self._mark_failed(error)
            return None

    def health(self) -> Dict[str, bool]:
        configured = bool(self._configured_url())
        client = self.client()
        if client is None:
            return {"configured": configured, "available": False}
        try:
            client.ping()
            return {"configured": True, "available": True}
        except Exception as error:
            with self._lock:
                self._mark_failed(error)
            return {"configured": True, "available": False}

    def reset(self) -> None:
        with self._lock:
            self._client, self._url, self._retry_at = None, None, 0.0


_memory_store = MemoryWindowStore()
_redis_store = RedisWindowStore()


def get_redis_health() -> Dict[str, bool]:
    """Redis rate-limit backend status for readiness checks."""
    return _redis_store.health()


def get_redis_client():
    """The Redis client used for rate limiting, or None (kept for callers of the old API)."""
    return _redis_store.client()


def clear_rate_limit_storage() -> None:
    """Reset in-memory counters and the Redis connection state (useful for tests)."""
    _memory_store.clear()
    _redis_store.reset()


def _window(window_seconds: int, now: Optional[float] = None) -> Tuple[int, int, float]:
    now = time.time() if now is None else now
    window = int(now // window_seconds)
    return window, (window + 1) * window_seconds, now


def get_rate_limit_status(request: Request) -> dict:
    """Current limit, remaining requests and reset time for the caller (in-memory view)."""
    identity = identify_client(request)
    window_seconds = _api_settings().rate_limit_window_seconds
    window, reset_at, _ = _window(window_seconds)
    used = _memory_store.peek(identity.bucket, window)
    return {"limit": identity.limit, "remaining": max(0, identity.limit - used), "reset_at": reset_at}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-identity fixed-window rate limiting.

    Register it *before* RequestLoggingMiddleware and CORSMiddleware so those run
    outside it: 429 responses then carry a request ID, are logged and counted,
    and get CORS headers like every other response.
    """

    def __init__(self, app, window_seconds: int = 60, exempt_paths: Iterable[str] = EXEMPT_PATHS):
        super().__init__(app)
        self.window_seconds = window_seconds
        self.exempt_paths = frozenset(exempt_paths)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.exempt_paths or request.method == "OPTIONS":
            return await call_next(request)

        identity = identify_client(request)
        window, reset_at, now = _window(self.window_seconds)
        count = _redis_store.hit(identity.bucket, window, self.window_seconds)
        backend = "redis"
        if count is None:
            count = _memory_store.hit(identity.bucket, window)
            backend = "memory"

        headers = {
            "X-RateLimit-Limit": str(identity.limit),
            "X-RateLimit-Remaining": str(max(0, identity.limit - count)),
            "X-RateLimit-Reset": str(reset_at),
            "X-RateLimit-Backend": backend,
        }

        if count > identity.limit:
            retry_after = max(1, math.ceil(reset_at - now))
            record_rate_limit_hit(identity.kind)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded: {identity.limit} requests per {self.window_seconds} seconds.",
                    "details": {
                        "limit": identity.limit,
                        "window_seconds": self.window_seconds,
                        "retry_after_seconds": retry_after,
                    },
                    "request_id": getattr(request.state, "request_id", None),
                },
                headers={**headers, "Retry-After": str(retry_after)},
            )

        response = await call_next(request)
        for name, value in headers.items():
            response.headers[name] = value
        return response
