"""
Rate limiting middleware using SlowAPI

Provides per-user rate limiting with configurable limits based on API key tiers.
"""

import time
from typing import Callable, Optional

from fastapi import HTTPException, Request, Response, status
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

# Storage for rate limit tracking (in production, use Redis)
_rate_limit_storage: dict[str, dict] = {}


def get_api_key_from_request(request: Request) -> str:
    """
    Extract rate limit key from request

    Uses API key if authenticated, otherwise falls back to IP address.
    This allows different rate limits per user tier.
    """
    # Try to get API key from header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key}"

    # Try to get from Bearer token (if already decoded)
    if hasattr(request.state, "user") and request.state.user:
        return f"apikey:{request.state.user.api_key}"

    # Fall back to IP address
    return f"ip:{get_remote_address(request)}"


def get_rate_limit_for_key(key: str) -> int:
    """
    Get rate limit for a given key

    Returns requests per minute based on user tier.
    """
    if key.startswith("apikey:"):
        api_key = key.replace("apikey:", "")
        try:
            from .auth import verify_api_key

            user = verify_api_key(api_key)
            if user:
                return user.rate_limit
        except Exception:
            pass
        return 100  # Default for valid API keys

    # Default rate limit for unauthenticated requests (by IP)
    return 20


# Create limiter instance
limiter = Limiter(
    key_func=get_api_key_from_request,
    default_limits=["100/minute"],
    storage_uri="memory://",
)


def get_rate_limiter() -> Limiter:
    """Get the global rate limiter instance"""
    return limiter


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Custom handler for rate limit exceeded errors

    Returns a proper JSON response with rate limit information.
    """
    from fastapi.responses import JSONResponse

    # Parse the rate limit info from the exception
    retry_after = int(exc.detail.split("retry after ")[1].split(" ")[0]) if "retry after" in exc.detail else 60

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please slow down.",
            "detail": str(exc.detail),
            "retry_after_seconds": retry_after,
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": request.headers.get("X-RateLimit-Limit", "100"),
            "X-RateLimit-Remaining": "0",
        },
    )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Custom rate limiting middleware with per-user limits

    This middleware provides more granular control than SlowAPI decorators,
    allowing dynamic rate limits based on user tier.
    """

    def __init__(self, app, default_limit: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.default_limit = default_limit
        self.window_seconds = window_seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health endpoints
        if request.url.path in ["/health", "/ready", "/metrics"]:
            return await call_next(request)

        # Get rate limit key
        key = get_api_key_from_request(request)

        # Get rate limit for this key
        limit = get_rate_limit_for_key(key)

        # Check and update rate limit
        current_time = time.time()
        window_start = current_time - self.window_seconds

        # Get or create tracking entry
        if key not in _rate_limit_storage:
            _rate_limit_storage[key] = {"requests": [], "limit": limit}

        entry = _rate_limit_storage[key]

        # Clean old requests outside the window
        entry["requests"] = [t for t in entry["requests"] if t > window_start]

        # Check if limit exceeded
        if len(entry["requests"]) >= limit:
            oldest_request = min(entry["requests"]) if entry["requests"] else current_time
            retry_after = int(oldest_request + self.window_seconds - current_time) + 1

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Maximum {limit} requests per minute.",
                    "limit": limit,
                    "retry_after_seconds": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(oldest_request + self.window_seconds)),
                },
            )

        # Add current request to tracking
        entry["requests"].append(current_time)
        remaining = limit - len(entry["requests"])

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window_seconds))

        return response


def dynamic_rate_limit(limit_func: Optional[Callable[[Request], str]] = None):
    """
    Decorator for dynamic rate limiting on specific endpoints

    Args:
        limit_func: Function that returns rate limit string (e.g., "10/minute")

    Example:
        @app.get("/expensive")
        @dynamic_rate_limit(lambda r: f"{get_user_limit(r)}/minute")
        async def expensive_endpoint():
            ...
    """

    def get_limit(request: Request) -> str:
        if limit_func:
            return limit_func(request)

        # Default dynamic limit based on user tier
        key = get_api_key_from_request(request)
        limit = get_rate_limit_for_key(key)
        return f"{limit}/minute"

    return limiter.limit(get_limit)


def clear_rate_limit_storage():
    """Clear rate limit storage (useful for testing)"""
    global _rate_limit_storage
    _rate_limit_storage = {}


def get_rate_limit_status(request: Request) -> dict:
    """
    Get current rate limit status for a request

    Returns:
        Dictionary with limit, remaining, and reset time
    """
    key = get_api_key_from_request(request)
    limit = get_rate_limit_for_key(key)
    current_time = time.time()
    window_seconds = 60

    if key not in _rate_limit_storage:
        return {
            "limit": limit,
            "remaining": limit,
            "reset_at": int(current_time + window_seconds),
        }

    entry = _rate_limit_storage[key]
    window_start = current_time - window_seconds
    recent_requests = [t for t in entry["requests"] if t > window_start]

    return {
        "limit": limit,
        "remaining": max(0, limit - len(recent_requests)),
        "reset_at": int(current_time + window_seconds),
    }
