"""
Sentiment Analysis API Router

Provides endpoints for analyzing text sentiment using transformer models.
"""

import asyncio
import time
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.middleware.auth import User, get_current_user
from api.schemas.requests import BatchTextRequest, SentimentRequest
from api.schemas.responses import SentimentResponse, SentimentResult
from utils.metrics import record_error, track_inference_time

router = APIRouter(prefix="/api/v1", tags=["Sentiment Analysis"])

# Default model for sentiment analysis
DEFAULT_SENTIMENT_MODEL = "twitter_roberta_multilingual"


async def run_sentiment_inference(text: str, model_key: str) -> tuple[dict, str, float]:
    """
    Run sentiment inference asynchronously

    Args:
        text: Text to analyze
        model_key: Model key from registry

    Returns:
        Tuple of (result, model_id, processing_time_ms)
    """
    from config.settings import get_model_registry
    from utils.model_cache import load_model

    registry = get_model_registry()

    # Get model config
    try:
        model_config = registry.get_model("sentiment_analysis", model_key)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_model",
                "message": f"Model '{model_key}' not found in sentiment_analysis category",
                "available_models": registry.list_models("sentiment_analysis"),
            },
        )

    # Run inference in thread pool to not block event loop
    start_time = time.time()

    def _inference():
        with track_inference_time(model_config.model_id, "sentiment"):
            pipeline = load_model("sentiment_analysis", model_key)
            return pipeline(text)

    result = await asyncio.to_thread(_inference)
    processing_time = (time.time() - start_time) * 1000

    return result, model_config.model_id, processing_time


@router.post(
    "/sentiment",
    response_model=SentimentResponse,
    summary="Analyze text sentiment",
    description="Analyze the sentiment of a text using transformer models. "
    "Returns sentiment label (positive/negative/neutral) with confidence scores.",
    responses={
        200: {
            "description": "Successful sentiment analysis",
            "content": {
                "application/json": {
                    "example": {
                        "text": "I love this product!",
                        "sentiment": {"label": "positive", "score": 0.9876},
                        "all_scores": [
                            {"label": "positive", "score": 0.9876},
                            {"label": "neutral", "score": 0.0098},
                            {"label": "negative", "score": 0.0026},
                        ],
                        "model": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
                        "processing_time_ms": 45.2,
                    }
                }
            },
        },
        400: {"description": "Invalid request"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def analyze_sentiment(
    request: Request,
    body: SentimentRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> SentimentResponse:
    """
    Analyze sentiment of provided text

    Supports multiple sentiment models. If no model is specified,
    uses the default Twitter RoBERTa multilingual model.

    **Authentication required:** API key or JWT token

    **Rate limits apply based on your API key tier.**
    """
    model_key = body.model or DEFAULT_SENTIMENT_MODEL

    try:
        result, model_id, processing_time = await run_sentiment_inference(body.text, model_key)

        # Parse the result (transformers returns list of dicts)
        if isinstance(result, list):
            # Handle pipeline output format
            if len(result) > 0 and isinstance(result[0], dict):
                # Single text result
                primary = result[0]
                all_scores = [SentimentResult(label=r["label"], score=r["score"]) for r in result]
            elif len(result) > 0 and isinstance(result[0], list):
                # Sometimes nested list
                primary = result[0][0]
                all_scores = [SentimentResult(label=r["label"], score=r["score"]) for r in result[0]]
            else:
                primary = result[0] if result else {"label": "unknown", "score": 0.0}
                all_scores = None
        else:
            primary = result
            all_scores = None

        # Normalize label names
        label = primary.get("label", "unknown").lower()
        # Map common label variations
        label_map = {
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
            "neg": "negative",
            "pos": "positive",
            "neu": "neutral",
        }
        label = label_map.get(label, label)

        return SentimentResponse(
            text=body.text,
            sentiment=SentimentResult(label=label, score=primary.get("score", 0.0)),
            all_scores=all_scores,
            model=model_id,
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        record_error("inference_error", "/api/v1/sentiment")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "inference_error",
                "message": f"Failed to analyze sentiment: {str(e)}",
            },
        )


@router.post(
    "/sentiment/batch",
    response_model=List[SentimentResponse],
    summary="Batch sentiment analysis",
    description="Analyze sentiment for multiple texts in a single request. "
    "More efficient than multiple individual requests.",
    responses={
        200: {"description": "Successful batch analysis"},
        400: {"description": "Invalid request"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def analyze_sentiment_batch(
    request: Request,
    body: BatchTextRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> List[SentimentResponse]:
    """
    Batch sentiment analysis for multiple texts

    Processes up to 100 texts in a single request.
    Results are returned in the same order as input texts.

    **Authentication required:** API key or JWT token

    **Rate limits apply - batch requests count as one request.**
    """
    from config.settings import get_model_registry
    from utils.model_cache import load_model

    model_key = body.model or DEFAULT_SENTIMENT_MODEL
    registry = get_model_registry()

    try:
        model_config = registry.get_model("sentiment_analysis", model_key)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_model",
                "message": f"Model '{model_key}' not found",
                "available_models": registry.list_models("sentiment_analysis"),
            },
        )

    try:
        start_time = time.time()

        def _batch_inference():
            with track_inference_time(model_config.model_id, "sentiment_batch"):
                pipeline = load_model("sentiment_analysis", model_key)
                return pipeline(body.texts)

        results = await asyncio.to_thread(_batch_inference)
        total_time = (time.time() - start_time) * 1000
        per_text_time = total_time / len(body.texts)

        responses = []
        for text, result in zip(body.texts, results):
            if isinstance(result, list):
                primary = result[0] if result else {"label": "unknown", "score": 0.0}
            else:
                primary = result

            label = primary.get("label", "unknown").lower()
            label_map = {
                "label_0": "negative",
                "label_1": "neutral",
                "label_2": "positive",
            }
            label = label_map.get(label, label)

            responses.append(
                SentimentResponse(
                    text=text,
                    sentiment=SentimentResult(label=label, score=primary.get("score", 0.0)),
                    all_scores=None,
                    model=model_config.model_id,
                    processing_time_ms=per_text_time,
                )
            )

        return responses

    except HTTPException:
        raise
    except Exception as e:
        record_error("inference_error", "/api/v1/sentiment/batch")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "inference_error",
                "message": f"Failed to analyze sentiment batch: {str(e)}",
            },
        )
