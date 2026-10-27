"""
FastAPI Application - Transformers NLP Suite

Production-minded REST API for customer-intelligence NLP inference with:
- JWT and API key authentication
- Rate limiting with usage tracking
- Prometheus metrics
- Comprehensive OpenAPI documentation
"""

from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.middleware.auth import api_key_header
from api.middleware.logging import RequestLoggingMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.routers import customer_intelligence, health, ner, qa, sentiment, similarity, summarization
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager

    Handles startup and shutdown events:
    - Startup: Initialize model cache, set startup time
    - Shutdown: Clean up resources
    """
    # Startup
    logger.info("Starting Transformers NLP Suite API...")

    # Set startup time for health endpoint
    from api.routers.health import set_startup_time

    set_startup_time()

    # Initialize metrics collector
    from utils.metrics import get_metrics_collector

    get_metrics_collector()

    # Pre-warm model cache if configured
    try:
        from config.settings import get_settings

        settings = get_settings()
        logger.info(f"API running with device: {settings.device}")
    except Exception as e:
        logger.warning(f"Could not load settings: {e}")

    logger.info("Transformers NLP Suite API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Transformers NLP Suite API...")

    # Clear model cache to free memory
    try:
        from utils.model_cache import clear_model_cache

        clear_model_cache()
    except Exception:
        pass

    logger.info("Transformers NLP Suite API shut down complete")


HTTP_ERROR_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    429: "rate_limit_exceeded",
    503: "service_unavailable",
}


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Transformers NLP Suite",
        description="""
## Overview

Transformers NLP Suite: an API-first service that turns customer reviews, support
notes and market text into structured NLP signals (sentiment, entities, similarity,
summaries and extractive answers), plus a combined customer-intelligence workflow.

### Authentication

Inference endpoints require one of:
- **API key**: `X-API-Key` header
- **JWT**: `Authorization: Bearer <token>`, obtained from `POST /api/v1/auth/token`

Use the **Authorize** button to set either credential for "Try it out".
Health, readiness, metrics and model-listing endpoints are public.

### Rate limiting

Requests are counted per verified client in fixed windows (60 s by default):
each API key has its own `rate_limit`, a JWT shares the budget of the key it was
issued for, and requests without valid credentials share a per-IP anonymous limit
(20 by default). Responses include `X-RateLimit-Limit`, `X-RateLimit-Remaining`,
`X-RateLimit-Reset` (window end, Unix time) and `X-RateLimit-Backend`; over-limit
requests get `429` with `Retry-After`.

### Input limits

Text longer than a model's context window is truncated to the model's token limit
(for example 512 tokens for the sentiment models, 1,024 for BART summarization).

### Errors

Every error uses the same envelope:
```json
{
  "error": "error_code",
  "message": "Human-readable description",
  "details": {},
  "request_id": "a1b2c3d4"
}
```
`details` is present when there is more to report (validation errors list each field).
        """,
        version="1.0.0",
        contact={
            "name": "Transformers NLP Suite on GitHub",
            "url": "https://github.com/MysterionRise/transformers-nlp-suite",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Optional OpenTelemetry instrumentation. The packages are intentionally
    # optional so local portfolio setup stays lightweight unless enabled.
    try:
        from config.settings import get_settings

        settings = get_settings()
        if settings.otel_enabled:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
            logger.info("OpenTelemetry FastAPI instrumentation enabled")
    except Exception as e:
        logger.warning(f"OpenTelemetry instrumentation not enabled: {e}")

    # Middleware order: Starlette runs the middleware added LAST outermost.
    # RateLimit is added first so it runs inside RequestLogging and CORS: its 429
    # responses then carry X-Request-ID, are logged and counted, and get CORS headers.
    try:
        from config.settings import get_settings

        settings = get_settings()
        cors_origins = settings.api.cors_origins
        rate_limit_window_seconds = settings.api.rate_limit_window_seconds
    except Exception:
        cors_origins = ["http://localhost:3000", "http://localhost:8000"]
        rate_limit_window_seconds = 60

    app.add_middleware(RateLimitMiddleware, window_seconds=rate_limit_window_seconds)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-Processing-Time-Ms",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "Retry-After",
        ],
    )

    # Include routers
    app.include_router(health.router)
    app.include_router(sentiment.router)
    app.include_router(summarization.router)
    app.include_router(ner.router)
    app.include_router(similarity.router)
    app.include_router(qa.router)
    app.include_router(customer_intelligence.router)

    # Custom exception handlers. Registered for Starlette's HTTPException so routing
    # errors (404, 405) use the same envelope as errors raised by the routers.
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Custom HTTP exception handler with consistent error format"""
        request_id = getattr(request.state, "request_id", None)

        if isinstance(exc.detail, dict):
            content = dict(exc.detail)
            content["request_id"] = request_id
        else:
            content = {
                "error": HTTP_ERROR_CODES.get(exc.status_code, "http_error"),
                "message": str(exc.detail),
                "request_id": request_id,
            }

        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """422 responses in the documented error envelope instead of FastAPI's default shape"""
        details = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ()) if part != "body"),
                "message": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": details,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions"""
        request_id = getattr(request.state, "request_id", None)
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "request_id": request_id,
            },
        )

    return app


def custom_openapi() -> Dict[str, Any]:
    """
    Customize OpenAPI schema

    Adds security schemes and additional documentation.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Security schemes come from the auth dependencies (ApiKeyAuth, BearerAuth) and
    # are attached only to the operations that use them, so public endpoints such as
    # /health show no lock and Swagger's "Authorize" credentials are sent by "Try it out".
    schemes = openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})
    schemes.setdefault("ApiKeyAuth", {"type": "apiKey", "in": "header", "name": "X-API-Key"})
    schemes.setdefault("BearerAuth", {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"})

    # Add tags metadata
    openapi_schema["tags"] = [
        {
            "name": "Health & Monitoring",
            "description": "Health checks and system monitoring endpoints",
        },
        {
            "name": "Sentiment Analysis",
            "description": "Analyze text sentiment using transformer models",
        },
        {
            "name": "Summarization",
            "description": "Generate text summaries using seq2seq models",
        },
        {
            "name": "Named Entity Recognition",
            "description": "Extract named entities from text",
        },
        {
            "name": "Semantic Similarity",
            "description": "Compute text similarity using embeddings",
        },
        {
            "name": "Question Answering",
            "description": "Extract answers from context passages",
        },
        {
            "name": "Customer Intelligence",
            "description": "Analyze customer feedback with combined NLP signals",
        },
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Create the application instance
app = create_app()
app.openapi = custom_openapi


# Root endpoint redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation"""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")


# Token endpoint for JWT authentication
@app.post(
    "/api/v1/auth/token",
    tags=["Authentication"],
    summary="Get JWT token",
    description="Exchange API key for a JWT token. The token can be used for "
    "Bearer authentication and has a configurable expiration time.",
    responses={
        200: {
            "description": "JWT token generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                        "token_type": "bearer",
                        "expires_in": 86400,
                    }
                }
            },
        },
        401: {"description": "Invalid API key"},
    },
)
async def get_token(
    request: Request,
    api_key: Annotated[Optional[str], Security(api_key_header)] = None,
):
    """
    Generate a JWT token from an API key

    Send your API key in the X-API-Key header to receive a JWT token.
    """
    from api.middleware.auth import create_access_token, get_api_settings, verify_api_key

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "unauthorized",
                "message": "X-API-Key header is required",
            },
        )

    api_settings = get_api_settings()
    if not api_settings.jwt_secret:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "jwt_not_configured",
                "message": "JWT authentication is not configured on this server (NLP_API__JWT_SECRET is empty)",
            },
        )

    user = verify_api_key(api_key)
    if not user:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "unauthorized",
                "message": "Invalid API key",
            },
        )

    token = create_access_token(user)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": int(api_settings.jwt_expiration_hours * 3600),
    }
