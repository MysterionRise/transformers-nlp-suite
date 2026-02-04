"""
Named Entity Recognition (NER) API Router

Provides endpoints for extracting named entities from text.
"""

import asyncio
import time
from collections import Counter
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.middleware.auth import User, get_current_user
from api.schemas.requests import NERRequest
from api.schemas.responses import EntityResponse, NERResponse
from utils.metrics import record_error, track_inference_time

router = APIRouter(prefix="/api/v1", tags=["Named Entity Recognition"])

# Default model for NER
DEFAULT_NER_MODEL = "spacy_sm"


async def run_spacy_ner(text: str, entity_types: Optional[List[str]] = None) -> tuple[list, str, float]:
    """
    Run NER using spaCy

    Args:
        text: Text to process
        entity_types: Optional filter for entity types

    Returns:
        Tuple of (entities, model_name, processing_time_ms)
    """
    start_time = time.time()

    def _inference():
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Try to download if not available
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm")

        doc = nlp(text)
        entities = []

        for ent in doc.ents:
            if entity_types is None or ent.label_ in entity_types:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "score": None,  # spaCy doesn't provide confidence scores
                    }
                )

        return entities

    with track_inference_time("en_core_web_sm", "ner"):
        entities = await asyncio.to_thread(_inference)

    processing_time = (time.time() - start_time) * 1000
    return entities, "en_core_web_sm", processing_time


async def run_transformer_ner(
    text: str, model_key: str, entity_types: Optional[List[str]] = None
) -> tuple[list, str, float]:
    """
    Run NER using transformer models

    Args:
        text: Text to process
        model_key: Model key from registry
        entity_types: Optional filter for entity types

    Returns:
        Tuple of (entities, model_id, processing_time_ms)
    """
    from config.settings import get_model_registry
    from utils.model_cache import load_model

    registry = get_model_registry()

    try:
        model_config = registry.get_model("ner", model_key)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_model",
                "message": f"Model '{model_key}' not found in ner category",
                "available_models": registry.list_models("ner"),
            },
        )

    start_time = time.time()

    def _inference():
        pipeline = load_model("ner", model_key)
        return pipeline(text)

    with track_inference_time(model_config.model_id, "ner"):
        result = await asyncio.to_thread(_inference)

    processing_time = (time.time() - start_time) * 1000

    # Parse transformer NER output
    entities = []
    for entity in result:
        label = entity.get("entity_group", entity.get("entity", "UNKNOWN"))
        # Remove B-, I- prefixes if present
        if label.startswith(("B-", "I-")):
            label = label[2:]

        if entity_types is None or label in entity_types:
            entities.append(
                {
                    "text": entity.get("word", ""),
                    "label": label,
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0),
                    "score": entity.get("score"),
                }
            )

    return entities, model_config.model_id, processing_time


@router.post(
    "/ner",
    response_model=NERResponse,
    summary="Extract named entities",
    description="Extract named entities (people, organizations, locations, dates, etc.) from text. "
    "Supports both spaCy and transformer-based models.",
    responses={
        200: {
            "description": "Successful entity extraction",
            "content": {
                "application/json": {
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
            },
        },
        400: {"description": "Invalid request"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def extract_entities(
    request: Request,
    body: NERRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> NERResponse:
    """
    Extract named entities from text

    Supports multiple NER backends:
    - **spacy_sm**: Fast spaCy model (default)
    - **spacy_trf**: Transformer-based spaCy model (more accurate)
    - **bert_ner**: BERT-based NER model

    You can filter results to specific entity types using the entity_types parameter.

    **Common entity types:**
    - PERSON: People names
    - ORG: Organizations
    - GPE: Geopolitical entities (countries, cities)
    - LOC: Locations
    - DATE: Dates
    - TIME: Times
    - MONEY: Monetary values
    - PERCENT: Percentages

    **Authentication required:** API key or JWT token
    """
    model_key = body.model or DEFAULT_NER_MODEL

    try:
        # Use spaCy for spacy_* models
        if model_key.startswith("spacy"):
            entities, model_id, processing_time = await run_spacy_ner(body.text, body.entity_types)
        else:
            entities, model_id, processing_time = await run_transformer_ner(body.text, model_key, body.entity_types)

        # Count entities by type
        entity_counts = dict(Counter(e["label"] for e in entities))

        # Convert to response format
        entity_responses = [
            EntityResponse(
                text=e["text"],
                label=e["label"],
                start=e["start"],
                end=e["end"],
                score=e.get("score"),
            )
            for e in entities
        ]

        return NERResponse(
            text=body.text,
            entities=entity_responses,
            entity_counts=entity_counts,
            model=model_id,
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        record_error("inference_error", "/api/v1/ner")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "inference_error",
                "message": f"Failed to extract entities: {str(e)}",
            },
        )
