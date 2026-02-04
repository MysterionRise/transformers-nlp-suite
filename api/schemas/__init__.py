"""
Pydantic schemas for API requests and responses
"""

from .requests import BatchTextRequest, NERRequest, QARequest, SentimentRequest, SimilarityRequest, SummarizationRequest
from .responses import (
    APIError,
    EntityResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfo,
    ModelsListResponse,
    NERResponse,
    QAResponse,
    SentimentResponse,
    SimilarityResponse,
    SummarizationResponse,
)

__all__ = [
    # Requests
    "SentimentRequest",
    "SummarizationRequest",
    "NERRequest",
    "SimilarityRequest",
    "QARequest",
    "BatchTextRequest",
    # Responses
    "SentimentResponse",
    "SummarizationResponse",
    "NERResponse",
    "EntityResponse",
    "SimilarityResponse",
    "QAResponse",
    "HealthResponse",
    "MetricsResponse",
    "ModelsListResponse",
    "ModelInfo",
    "APIError",
]
