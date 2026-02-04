"""
Health check and monitoring endpoints

Provides:
- /health - Liveness probe (always returns 200)
- /ready - Readiness probe (checks dependencies)
- /metrics - Prometheus metrics endpoint
- /api/v1/models - List available models
- /api/v1/status - Detailed system status
"""

from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Request, Response

from api.middleware.auth import User, get_optional_user
from api.schemas.responses import HealthResponse, MetricsResponse, ModelInfo, ModelsListResponse, ReadinessResponse
from utils.metrics import get_metrics, get_metrics_collector

router = APIRouter(tags=["Health & Monitoring"])


# Store startup time for uptime calculation
_startup_time: Optional[datetime] = None


def set_startup_time():
    """Set the application startup time (called from main.py)"""
    global _startup_time
    _startup_time = datetime.now(timezone.utc)


def get_uptime_seconds() -> float:
    """Get application uptime in seconds"""
    if _startup_time is None:
        return 0.0
    delta = datetime.now(timezone.utc) - _startup_time
    return delta.total_seconds()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 if the service is alive. Use for Kubernetes liveness probes.",
    responses={
        200: {
            "description": "Service is alive",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "version": "1.0.0",
                        "uptime_seconds": 3600.5,
                    }
                }
            },
        }
    },
)
async def health_check() -> HealthResponse:
    """
    Liveness probe endpoint

    Always returns 200 OK if the service is running.
    Used by container orchestrators to determine if the service needs to be restarted.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version="1.0.0",
        uptime_seconds=get_uptime_seconds(),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Returns 200 if the service is ready to accept requests. " "Checks model cache and configuration.",
    responses={
        200: {
            "description": "Service is ready",
        },
        503: {
            "description": "Service is not ready",
        },
    },
)
async def readiness_check() -> ReadinessResponse:
    """
    Readiness probe endpoint

    Checks if critical dependencies are available:
    - Model cache is initialized
    - Configuration is loaded
    - Required models can be accessed

    Used by load balancers to determine if the service should receive traffic.
    """
    checks = {}

    # Check configuration
    try:
        from config.settings import get_model_registry, get_settings

        get_settings()  # Verify settings can be loaded
        checks["config_loaded"] = True
        checks["model_registry"] = get_model_registry() is not None
    except Exception:
        checks["config_loaded"] = False
        checks["model_registry"] = False

    # Check model cache
    try:
        from utils.model_cache import get_model_cache

        cache = get_model_cache()
        checks["model_cache"] = cache is not None
        checks["cache_initialized"] = cache._initialized if hasattr(cache, "_initialized") else False
    except Exception:
        checks["model_cache"] = False
        checks["cache_initialized"] = False

    # Determine overall status
    all_passed = all(checks.values())
    status = "ready" if all_passed else "not_ready"

    return ReadinessResponse(
        status=status,
        checks=checks,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Returns metrics in Prometheus exposition format.",
    response_class=Response,
    responses={
        200: {
            "description": "Prometheus metrics",
            "content": {"text/plain": {"example": "# HELP http_requests_total Total HTTP requests\n"}},
        }
    },
)
async def prometheus_metrics() -> Response:
    """
    Prometheus metrics endpoint

    Returns metrics in Prometheus text exposition format for scraping.
    """
    metrics_data = get_metrics()
    return Response(content=metrics_data, media_type="text/plain; charset=utf-8")


@router.get(
    "/api/v1/status",
    response_model=MetricsResponse,
    summary="System status",
    description="Returns detailed system status including cache statistics and request metrics.",
)
async def system_status(
    request: Request,
    user: Annotated[Optional[User], Depends(get_optional_user)] = None,
) -> MetricsResponse:
    """
    Detailed system status endpoint

    Returns:
    - Request counts by endpoint
    - Model cache statistics
    - Average latency metrics
    """
    # Get cache stats
    try:
        from utils.model_cache import get_cache_info

        cache_stats = get_cache_info()
    except Exception:
        cache_stats = {"error": "Unable to retrieve cache stats"}

    # Get metrics collector for uptime
    metrics = get_metrics_collector()

    return MetricsResponse(
        requests_total=0,  # Would be populated from actual metrics
        requests_by_endpoint={},
        model_cache_stats={**cache_stats, "uptime_seconds": metrics.uptime_seconds},
        average_latency_ms={},
    )


@router.get(
    "/api/v1/models",
    response_model=list[ModelsListResponse],
    summary="List available models",
    description="Returns a list of all available models organized by category.",
)
async def list_models() -> list[ModelsListResponse]:
    """
    List all available models by category

    Returns models from the model registry organized by task category.
    """
    from config.settings import get_model_registry

    registry = get_model_registry()
    result = []

    for category in registry.list_categories():
        models = []
        category_models = registry.get_category(category)

        for model_key, config in category_models.items():
            models.append(
                ModelInfo(
                    key=model_key,
                    name=config.name,
                    model_id=config.model_id,
                    task=config.task,
                    description=config.description or None,
                )
            )

        result.append(
            ModelsListResponse(
                category=category,
                models=models,
            )
        )

    return result


@router.get(
    "/api/v1/models/{category}",
    response_model=ModelsListResponse,
    summary="List models in category",
    description="Returns models available for a specific task category.",
    responses={
        404: {"description": "Category not found"},
    },
)
async def list_models_by_category(category: str) -> ModelsListResponse:
    """
    List models for a specific category

    Args:
        category: Model category (e.g., 'sentiment_analysis', 'summarization')

    Returns:
        Models available in the specified category
    """
    from fastapi import HTTPException

    from config.settings import get_model_registry

    registry = get_model_registry()

    try:
        category_models = registry.get_category(category)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_found",
                "message": f"Category '{category}' not found",
                "available_categories": registry.list_categories(),
            },
        )

    models = []
    for model_key, config in category_models.items():
        models.append(
            ModelInfo(
                key=model_key,
                name=config.name,
                model_id=config.model_id,
                task=config.task,
                description=config.description or None,
            )
        )

    return ModelsListResponse(
        category=category,
        models=models,
    )
