"""
Pydantic response models for API endpoints

All response models include comprehensive documentation and examples
for beautiful OpenAPI/Swagger UI documentation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    """Individual sentiment analysis result"""

    label: str = Field(
        ...,
        description="Sentiment label (e.g., 'positive', 'negative', 'neutral')",
        json_schema_extra={"example": "positive"},
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1",
        json_schema_extra={"example": 0.9876},
    )


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis"""

    text: str = Field(
        ...,
        description="Original input text",
    )
    sentiment: SentimentResult = Field(
        ...,
        description="Primary sentiment prediction",
    )
    all_scores: Optional[List[SentimentResult]] = Field(
        default=None,
        description="Scores for all sentiment classes (if available)",
    )
    model: str = Field(
        ...,
        description="Model used for inference",
        json_schema_extra={"example": "cardiffnlp/twitter-roberta-base-sentiment-latest"},
    )
    processing_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds",
        json_schema_extra={"example": 45.2},
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "I love this product!",
                "sentiment": {"label": "positive", "score": 0.9876},
                "all_scores": [
                    {"label": "positive", "score": 0.9876},
                    {"label": "neutral", "score": 0.0098},
                    {"label": "negative", "score": 0.0026},
                ],
                "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "processing_time_ms": 45.2,
            }
        }
    }


class SummarizationResponse(BaseModel):
    """Response model for text summarization"""

    original_text: str = Field(
        ...,
        description="Original input text",
    )
    summary: str = Field(
        ...,
        description="Generated summary",
        json_schema_extra={"example": "AI has transformed industries through automation and improved decision-making."},
    )
    original_length: int = Field(
        ...,
        description="Character count of original text",
        json_schema_extra={"example": 450},
    )
    summary_length: int = Field(
        ...,
        description="Character count of summary",
        json_schema_extra={"example": 85},
    )
    compression_ratio: float = Field(
        ...,
        description="Ratio of summary length to original length",
        json_schema_extra={"example": 0.189},
    )
    model: str = Field(
        ...,
        description="Model used for inference",
        json_schema_extra={"example": "facebook/bart-large-cnn"},
    )
    processing_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds",
        json_schema_extra={"example": 1250.5},
    )


class EntityResponse(BaseModel):
    """Individual named entity"""

    text: str = Field(
        ...,
        description="Entity text as it appears in the input",
        json_schema_extra={"example": "Apple Inc."},
    )
    label: str = Field(
        ...,
        description="Entity type label",
        json_schema_extra={"example": "ORG"},
    )
    start: int = Field(
        ...,
        description="Start character offset in input text",
        json_schema_extra={"example": 0},
    )
    end: int = Field(
        ...,
        description="End character offset in input text",
        json_schema_extra={"example": 10},
    )
    score: Optional[float] = Field(
        default=None,
        description="Confidence score (if available)",
        json_schema_extra={"example": 0.9987},
    )


class NERResponse(BaseModel):
    """Response model for Named Entity Recognition"""

    text: str = Field(
        ...,
        description="Original input text",
    )
    entities: List[EntityResponse] = Field(
        ...,
        description="List of extracted entities",
    )
    entity_counts: Dict[str, int] = Field(
        ...,
        description="Count of entities by type",
        json_schema_extra={"example": {"ORG": 2, "PERSON": 1, "DATE": 1}},
    )
    model: str = Field(
        ...,
        description="Model used for inference",
        json_schema_extra={"example": "en_core_web_sm"},
    )
    processing_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds",
        json_schema_extra={"example": 23.5},
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Tim Cook is the CEO of Apple Inc.",
                "entities": [
                    {"text": "Tim Cook", "label": "PERSON", "start": 0, "end": 8, "score": 0.99},
                    {"text": "Apple Inc.", "label": "ORG", "start": 23, "end": 33, "score": 0.98},
                ],
                "entity_counts": {"PERSON": 1, "ORG": 1},
                "model": "en_core_web_sm",
                "processing_time_ms": 23.5,
            }
        }
    }


class SimilarityResponse(BaseModel):
    """Response model for semantic similarity"""

    text1: str = Field(
        ...,
        description="First input text",
    )
    text2: str = Field(
        ...,
        description="Second input text",
    )
    similarity_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Cosine similarity score (-1 to 1, higher = more similar)",
        json_schema_extra={"example": 0.8523},
    )
    model: str = Field(
        ...,
        description="Model used for inference",
        json_schema_extra={"example": "sentence-transformers/all-MiniLM-L6-v2"},
    )
    processing_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds",
        json_schema_extra={"example": 35.7},
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text1": "The weather is beautiful today.",
                "text2": "It's a lovely sunny day outside.",
                "similarity_score": 0.8523,
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "processing_time_ms": 35.7,
            }
        }
    }


class QAAnswer(BaseModel):
    """Individual answer result"""

    answer: str = Field(
        ...,
        description="Extracted answer text",
        json_schema_extra={"example": "Paris"},
    )
    score: float = Field(
        ...,
        description="Confidence score",
        json_schema_extra={"example": 0.9234},
    )
    start: int = Field(
        ...,
        description="Start character offset in context",
        json_schema_extra={"example": 45},
    )
    end: int = Field(
        ...,
        description="End character offset in context",
        json_schema_extra={"example": 50},
    )


class QAResponse(BaseModel):
    """Response model for Question Answering"""

    question: str = Field(
        ...,
        description="Original question",
    )
    context: str = Field(
        ...,
        description="Context passage",
    )
    answers: List[QAAnswer] = Field(
        ...,
        description="List of extracted answers (ranked by confidence)",
    )
    model: str = Field(
        ...,
        description="Model used for inference",
        json_schema_extra={"example": "distilbert-base-cased-distilled-squad"},
    )
    processing_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds",
        json_schema_extra={"example": 52.3},
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is the capital of France?",
                "context": "France is a country in Western Europe. Its capital is Paris.",
                "answers": [{"answer": "Paris", "score": 0.9234, "start": 52, "end": 57}],
                "model": "distilbert-base-cased-distilled-squad",
                "processing_time_ms": 52.3,
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""

    status: str = Field(
        ...,
        description="Service health status",
        json_schema_extra={"example": "healthy"},
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of health check",
    )
    version: str = Field(
        ...,
        description="API version",
        json_schema_extra={"example": "1.0.0"},
    )
    uptime_seconds: float = Field(
        ...,
        description="Service uptime in seconds",
        json_schema_extra={"example": 3600.5},
    )


class ReadinessResponse(BaseModel):
    """Response model for readiness check endpoint"""

    status: str = Field(
        ...,
        description="Readiness status ('ready' or 'not_ready')",
        json_schema_extra={"example": "ready"},
    )
    checks: Dict[str, bool] = Field(
        ...,
        description="Individual check results",
        json_schema_extra={
            "example": {
                "model_cache": True,
                "config_loaded": True,
            }
        },
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of readiness check",
    )


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint (summary, not Prometheus format)"""

    requests_total: int = Field(
        ...,
        description="Total number of requests processed",
        json_schema_extra={"example": 15420},
    )
    requests_by_endpoint: Dict[str, int] = Field(
        ...,
        description="Request counts by endpoint",
        json_schema_extra={
            "example": {
                "/api/v1/sentiment": 8500,
                "/api/v1/summarize": 3200,
                "/api/v1/ner": 2100,
            }
        },
    )
    model_cache_stats: Dict[str, Any] = Field(
        ...,
        description="Model cache statistics",
    )
    average_latency_ms: Dict[str, float] = Field(
        ...,
        description="Average latency by endpoint in milliseconds",
        json_schema_extra={
            "example": {
                "/api/v1/sentiment": 45.2,
                "/api/v1/summarize": 1250.5,
            }
        },
    )


class ModelInfo(BaseModel):
    """Information about an available model"""

    key: str = Field(
        ...,
        description="Model key for API requests",
        json_schema_extra={"example": "twitter_roberta_multilingual"},
    )
    name: str = Field(
        ...,
        description="Human-readable model name",
        json_schema_extra={"example": "Twitter RoBERTa Multilingual"},
    )
    model_id: str = Field(
        ...,
        description="HuggingFace model identifier",
        json_schema_extra={"example": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"},
    )
    task: str = Field(
        ...,
        description="Model task type",
        json_schema_extra={"example": "sentiment-analysis"},
    )
    description: Optional[str] = Field(
        default=None,
        description="Model description",
    )


class ModelsListResponse(BaseModel):
    """Response model for listing available models"""

    category: str = Field(
        ...,
        description="Model category",
        json_schema_extra={"example": "sentiment_analysis"},
    )
    models: List[ModelInfo] = Field(
        ...,
        description="Available models in this category",
    )


class APIError(BaseModel):
    """Standard error response model"""

    error: str = Field(
        ...,
        description="Error type",
        json_schema_extra={"example": "validation_error"},
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
        json_schema_extra={"example": "Text cannot be empty"},
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking",
        json_schema_extra={"example": "req_abc123xyz"},
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "validation_error",
                "message": "Text must be at least 50 characters for summarization",
                "details": {"field": "text", "min_length": 50, "actual_length": 25},
                "request_id": "req_abc123xyz",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
    }


class RateLimitResponse(BaseModel):
    """Response model for rate limit information"""

    limit: int = Field(
        ...,
        description="Maximum requests allowed in the window",
        json_schema_extra={"example": 100},
    )
    remaining: int = Field(
        ...,
        description="Remaining requests in current window",
        json_schema_extra={"example": 87},
    )
    reset_at: datetime = Field(
        ...,
        description="When the rate limit window resets",
    )
    retry_after_seconds: Optional[int] = Field(
        default=None,
        description="Seconds to wait before retrying (if rate limited)",
    )
