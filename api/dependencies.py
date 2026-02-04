"""
Shared dependencies for API endpoints

Provides dependency injection for common operations like model loading,
authentication, and request validation.
"""

import asyncio
from typing import Annotated, Optional

from fastapi import Depends, Request

from api.middleware.auth import User, get_current_user, get_optional_user
from api.middleware.rate_limit import get_rate_limit_status


async def get_model_cache():
    """
    Dependency to get the model cache instance

    Returns:
        ModelCache singleton instance
    """
    from utils.model_cache import get_model_cache as _get_cache

    return _get_cache()


async def get_model_registry():
    """
    Dependency to get the model registry instance

    Returns:
        ModelRegistry singleton instance
    """
    from config.settings import get_model_registry as _get_registry

    return _get_registry()


async def get_settings():
    """
    Dependency to get application settings

    Returns:
        Settings singleton instance
    """
    from config.settings import get_settings as _get_settings

    return _get_settings()


async def load_model_async(category: str, model_key: str):
    """
    Load a model asynchronously (non-blocking)

    Args:
        category: Model category
        model_key: Model key

    Returns:
        Loaded model pipeline
    """
    from utils.model_cache import load_model

    return await asyncio.to_thread(load_model, category, model_key)


class RateLimitInfo:
    """Rate limit information for the current request"""

    def __init__(self, limit: int, remaining: int, reset_at: int):
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at


async def get_rate_limit_info(request: Request) -> RateLimitInfo:
    """
    Dependency to get rate limit status for current request

    Returns:
        RateLimitInfo with current limit status
    """
    status = get_rate_limit_status(request)
    return RateLimitInfo(
        limit=status["limit"],
        remaining=status["remaining"],
        reset_at=status["reset_at"],
    )


# Type aliases for cleaner dependency injection
CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[Optional[User], Depends(get_optional_user)]
RateLimit = Annotated[RateLimitInfo, Depends(get_rate_limit_info)]
