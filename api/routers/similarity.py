"""
Semantic Similarity API Router

Provides endpoints for computing text similarity using sentence embeddings.
"""

import asyncio
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.middleware.auth import User, get_current_user
from api.schemas.requests import SimilarityRequest
from api.schemas.responses import SimilarityResponse
from utils.metrics import record_error, track_inference_time

router = APIRouter(prefix="/api/v1", tags=["Semantic Similarity"])

# Default model for similarity
DEFAULT_SIMILARITY_MODEL = "minilm"


def compute_cosine_similarity(embedding1, embedding2) -> float:
    """Compute cosine similarity between two embeddings"""
    import numpy as np

    # Ensure numpy arrays
    e1 = np.array(embedding1).flatten()
    e2 = np.array(embedding2).flatten()

    # Compute cosine similarity
    dot_product = np.dot(e1, e2)
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


@router.post(
    "/similarity",
    response_model=SimilarityResponse,
    summary="Compute text similarity",
    description="Compute semantic similarity between two texts using sentence embeddings. "
    "Returns a score from -1 to 1, where higher values indicate more similar texts.",
    responses={
        200: {
            "description": "Successful similarity computation",
            "content": {
                "application/json": {
                    "example": {
                        "text1": "The weather is beautiful today.",
                        "text2": "It's a lovely sunny day outside.",
                        "similarity_score": 0.8523,
                        "model": "sentence-transformers/all-MiniLM-L6-v2",
                        "processing_time_ms": 35.7,
                    }
                }
            },
        },
        400: {"description": "Invalid request"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def compute_similarity(
    request: Request,
    body: SimilarityRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> SimilarityResponse:
    """
    Compute semantic similarity between two texts

    Uses sentence transformer models to generate embeddings and compute
    cosine similarity. Higher scores (closer to 1) indicate more similar texts.

    **Score interpretation:**
    - 0.9-1.0: Very similar (likely paraphrases)
    - 0.7-0.9: Similar (same topic/meaning)
    - 0.5-0.7: Somewhat similar
    - 0.3-0.5: Different topics
    - 0.0-0.3: Very different
    - Negative: Opposite meanings (rare)

    **Authentication required:** API key or JWT token
    """
    from config.settings import get_model_registry

    model_key = body.model or DEFAULT_SIMILARITY_MODEL
    registry = get_model_registry()

    # Get model config
    try:
        model_config = registry.get_model("embeddings", model_key)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_model",
                "message": f"Model '{model_key}' not found in embeddings category",
                "available_models": registry.list_models("embeddings"),
            },
        )

    try:
        start_time = time.time()

        def _inference():
            # For sentence-transformers models, use the dedicated library
            from sentence_transformers import SentenceTransformer

            # Load model (sentence-transformers has its own caching)
            model = SentenceTransformer(model_config.model_id)

            # Generate embeddings for both texts
            embeddings = model.encode([body.text1, body.text2])

            return embeddings[0], embeddings[1]

        with track_inference_time(model_config.model_id, "similarity"):
            emb1, emb2 = await asyncio.to_thread(_inference)

        # Compute cosine similarity
        similarity = compute_cosine_similarity(emb1, emb2)
        processing_time = (time.time() - start_time) * 1000

        return SimilarityResponse(
            text1=body.text1,
            text2=body.text2,
            similarity_score=round(similarity, 4),
            model=model_config.model_id,
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        record_error("inference_error", "/api/v1/similarity")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "inference_error",
                "message": f"Failed to compute similarity: {str(e)}",
            },
        )
