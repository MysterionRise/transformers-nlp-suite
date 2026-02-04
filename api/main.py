"""
FastAPI Application - Enterprise NLP API

Production-ready REST API for NLP inference with:
- JWT and API key authentication
- Rate limiting with usage tracking
- Prometheus metrics
- Comprehensive OpenAPI documentation
"""

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from api.middleware.logging import RequestLoggingMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.routers import health, ner, qa, sentiment, similarity, summarization
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
    logger.info("Starting Enterprise NLP API...")

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

    logger.info("Enterprise NLP API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Enterprise NLP API...")

    # Clear model cache to free memory
    try:
        from utils.model_cache import clear_model_cache

        clear_model_cache()
    except Exception:
        pass

    logger.info("Enterprise NLP API shut down complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Enterprise NLP API",
        description="""
## Overview

Production-ready REST API for Natural Language Processing tasks.

### Features

- **Sentiment Analysis**: Analyze text sentiment (positive/negative/neutral)
- **Text Summarization**: Generate concise summaries
- **Named Entity Recognition**: Extract people, organizations, locations, dates
- **Semantic Similarity**: Compare text similarity using embeddings
- **Question Answering**: Extract answers from context

### Authentication

All endpoints require authentication via one of:
- **API Key**: Include `X-API-Key` header
- **JWT Token**: Include `Authorization: Bearer <token>` header

### Rate Limiting

Requests are rate-limited based on your API key tier:
- **Admin**: 1000 requests/minute
- **User**: 100 requests/minute
- **Demo**: 20 requests/minute

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Unix timestamp when window resets

### Error Handling

All errors follow a consistent format:
```json
{
  "error": "error_type",
  "message": "Human-readable description",
  "details": {},
  "request_id": "req_xxx"
}
```
        """,
        version="1.0.0",
        contact={
            "name": "API Support",
            "email": "support@example.com",
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

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-Processing-Time-Ms",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ],
    )

    # Add custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, default_limit=100, window_seconds=60)

    # Include routers
    app.include_router(health.router)
    app.include_router(sentiment.router)
    app.include_router(summarization.router)
    app.include_router(ner.router)
    app.include_router(similarity.router)
    app.include_router(qa.router)

    # Custom exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Custom HTTP exception handler with consistent error format"""
        request_id = getattr(request.state, "request_id", None)

        # Handle dict detail (our custom format)
        if isinstance(exc.detail, dict):
            content = exc.detail
            content["request_id"] = request_id
        else:
            content = {
                "error": "http_error",
                "message": str(exc.detail),
                "request_id": request_id,
            }

        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            headers=exc.headers,
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

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication",
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token for authentication",
        },
    }

    # Apply security to all endpoints
    openapi_schema["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]

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
async def get_token(request: Request):
    """
    Generate a JWT token from an API key

    Send your API key in the X-API-Key header to receive a JWT token.
    """
    from api.middleware.auth import create_access_token, verify_api_key

    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "unauthorized",
                "message": "X-API-Key header is required",
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
        "expires_in": 86400,  # 24 hours
    }
