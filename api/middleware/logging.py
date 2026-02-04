"""
Request/Response logging middleware

Provides structured logging for all API requests with timing information.
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from utils.logger import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all HTTP requests and responses

    Logs:
    - Request method, path, and headers
    - Response status code
    - Processing time
    - Request ID for correlation
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Extract relevant request info
        method = request.method
        path = request.url.path
        query = str(request.query_params) if request.query_params else ""
        client_ip = request.client.host if request.client else "unknown"

        # Get user info if authenticated
        user_info = "anonymous"
        if hasattr(request.state, "user") and request.state.user:
            user_info = f"{request.state.user.name} ({request.state.user.role})"

        # Log request start
        logger.info(
            f"[{request_id}] {method} {path} - Client: {client_ip}",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "query": query,
                "client_ip": client_ip,
            },
        )

        # Process request and measure time
        start_time = time.time()
        try:
            response = await call_next(request)
            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time-Ms"] = f"{processing_time:.2f}"

            # Log response
            log_level = "info" if response.status_code < 400 else "warning"
            getattr(logger, log_level)(
                f"[{request_id}] {method} {path} - {response.status_code} ({processing_time:.2f}ms)",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "processing_time_ms": processing_time,
                    "user": user_info,
                },
            )

            return response

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(
                f"[{request_id}] {method} {path} - Error: {str(e)} ({processing_time:.2f}ms)",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "error": str(e),
                    "processing_time_ms": processing_time,
                },
                exc_info=True,
            )
            raise


def log_request_body(request: Request, body: dict):
    """
    Log request body (called from endpoints for detailed logging)

    Args:
        request: FastAPI request object
        body: Parsed request body
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Truncate long text fields for logging
    logged_body = {}
    for key, value in body.items():
        if isinstance(value, str) and len(value) > 100:
            logged_body[key] = f"{value[:100]}... ({len(value)} chars)"
        else:
            logged_body[key] = value

    logger.debug(
        f"[{request_id}] Request body: {logged_body}",
        extra={
            "request_id": request_id,
            "body": logged_body,
        },
    )


def log_response_body(request: Request, body: dict):
    """
    Log response body (called from endpoints for detailed logging)

    Args:
        request: FastAPI request object
        body: Response body to log
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.debug(
        f"[{request_id}] Response body keys: {list(body.keys())}",
        extra={
            "request_id": request_id,
            "response_keys": list(body.keys()),
        },
    )
