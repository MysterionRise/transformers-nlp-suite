"""
Enterprise NLP API - Production REST API Service

Provides a FastAPI-based REST API for NLP inference with:
- JWT and API key authentication
- Rate limiting with usage tracking
- Prometheus metrics
- Comprehensive OpenAPI documentation
"""

from .main import app, create_app

__all__ = ["app", "create_app"]
