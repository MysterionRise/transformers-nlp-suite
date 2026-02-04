"""
Question Answering API Router

Provides endpoints for extractive question answering.
"""

import asyncio
import time
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.middleware.auth import User, get_current_user
from api.schemas.requests import QARequest
from api.schemas.responses import QAAnswer, QAResponse
from utils.metrics import record_error, track_inference_time

router = APIRouter(prefix="/api/v1", tags=["Question Answering"])

# Default model for QA
DEFAULT_QA_MODEL = "distilbert_squad"


@router.post(
    "/qa",
    response_model=QAResponse,
    summary="Answer questions from context",
    description="Extract answers to questions from a given context passage. "
    "Uses extractive QA models trained on datasets like SQuAD.",
    responses={
        200: {
            "description": "Successful question answering",
            "content": {
                "application/json": {
                    "example": {
                        "question": "What is the capital of France?",
                        "context": "France is a country in Western Europe. Its capital is Paris.",
                        "answers": [{"answer": "Paris", "score": 0.9234, "start": 52, "end": 57}],
                        "model": "distilbert-base-cased-distilled-squad",
                        "processing_time_ms": 52.3,
                    }
                }
            },
        },
        400: {"description": "Invalid request"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def answer_question(
    request: Request,
    body: QARequest,
    user: Annotated[User, Depends(get_current_user)],
) -> QAResponse:
    """
    Answer a question based on the provided context

    Uses extractive question answering to find the answer span within the context.
    The model will identify the most relevant portion of the context that answers
    the question.

    **Tips for best results:**
    - Provide clear, specific questions
    - Ensure the context contains the answer
    - Use factual questions (who, what, when, where)

    **Authentication required:** API key or JWT token
    """
    from config.settings import get_model_registry
    from utils.model_cache import load_model

    model_key = body.model or DEFAULT_QA_MODEL
    registry = get_model_registry()

    # Get model config
    try:
        model_config = registry.get_model("question_answering", model_key)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_model",
                "message": f"Model '{model_key}' not found in question_answering category",
                "available_models": registry.list_models("question_answering"),
            },
        )

    try:
        start_time = time.time()

        def _inference():
            pipeline = load_model("question_answering", model_key)
            # Get top_k answers
            return pipeline(question=body.question, context=body.context, top_k=body.top_k)

        with track_inference_time(model_config.model_id, "qa"):
            result = await asyncio.to_thread(_inference)

        processing_time = (time.time() - start_time) * 1000

        # Parse results
        answers: List[QAAnswer] = []

        # Handle both single and multiple results
        if isinstance(result, dict):
            result = [result]

        for r in result:
            answers.append(
                QAAnswer(
                    answer=r.get("answer", ""),
                    score=r.get("score", 0.0),
                    start=r.get("start", 0),
                    end=r.get("end", 0),
                )
            )

        return QAResponse(
            question=body.question,
            context=body.context,
            answers=answers,
            model=model_config.model_id,
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        record_error("inference_error", "/api/v1/qa")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "inference_error",
                "message": f"Failed to answer question: {str(e)}",
            },
        )
