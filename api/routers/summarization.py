"""
Text Summarization API Router

Provides endpoints for generating text summaries using transformer models.
"""

import asyncio
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.middleware.auth import User, get_current_user
from api.schemas.requests import SummarizationRequest
from api.schemas.responses import SummarizationResponse
from utils.metrics import record_error, track_inference_time

router = APIRouter(prefix="/api/v1", tags=["Summarization"])

# Default model for summarization
DEFAULT_SUMMARIZATION_MODEL = "bart_large_cnn"


@router.post(
    "/summarize",
    response_model=SummarizationResponse,
    summary="Summarize text",
    description="Generate a concise summary of the provided text using transformer models. "
    "Supports adjustable summary length parameters.",
    responses={
        200: {
            "description": "Successful summarization",
            "content": {
                "application/json": {
                    "example": {
                        "original_text": "A long article about AI...",
                        "summary": "AI has transformed industries through automation and decision-making.",
                        "original_length": 450,
                        "summary_length": 85,
                        "compression_ratio": 0.189,
                        "model": "facebook/bart-large-cnn",
                        "processing_time_ms": 1250.5,
                    }
                }
            },
        },
        400: {"description": "Invalid request"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def summarize_text(
    request: Request,
    body: SummarizationRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> SummarizationResponse:
    """
    Generate a summary of the provided text

    Uses state-of-the-art summarization models like BART, T5, or Pegasus.
    You can control the summary length with min_length and max_length parameters.

    **Authentication required:** API key or JWT token

    **Rate limits apply based on your API key tier.**
    """
    from config.settings import get_model_registry
    from utils.model_cache import load_model

    model_key = body.model or DEFAULT_SUMMARIZATION_MODEL
    registry = get_model_registry()

    # Get model config
    try:
        model_config = registry.get_model("summarization", model_key)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_model",
                "message": f"Model '{model_key}' not found in summarization category",
                "available_models": registry.list_models("summarization"),
            },
        )

    try:
        start_time = time.time()

        def _inference():
            with track_inference_time(model_config.model_id, "summarization"):
                pipeline = load_model("summarization", model_key)
                return pipeline(
                    body.text,
                    min_length=body.min_length,
                    max_length=body.max_length,
                    do_sample=False,
                )

        result = await asyncio.to_thread(_inference)
        processing_time = (time.time() - start_time) * 1000

        # Extract summary text from result
        if isinstance(result, list) and len(result) > 0:
            summary = result[0].get("summary_text", "")
        else:
            summary = result.get("summary_text", "") if isinstance(result, dict) else str(result)

        # Calculate statistics
        original_length = len(body.text)
        summary_length = len(summary)
        compression_ratio = summary_length / original_length if original_length > 0 else 0

        return SummarizationResponse(
            original_text=body.text,
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=round(compression_ratio, 3),
            model=model_config.model_id,
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        record_error("inference_error", "/api/v1/summarize")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "inference_error",
                "message": f"Failed to summarize text: {str(e)}",
            },
        )
