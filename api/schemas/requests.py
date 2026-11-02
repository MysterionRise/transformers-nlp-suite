"""
Pydantic request models for API endpoints

All request models include validation, examples, and comprehensive documentation
for beautiful OpenAPI/Swagger UI documentation.
"""

from typing import Annotated, Any, Dict, List, Optional

from pydantic import AfterValidator, BaseModel, Field, model_validator


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace characters")
    return value


# Text fields reject whitespace-only input (it used to be classified as "negative").
NonBlank = AfterValidator(_not_blank)
BatchText = Annotated[str, Field(min_length=1, max_length=5000), NonBlank]


class SentimentRequest(BaseModel):
    """Request model for sentiment analysis"""

    text: Annotated[str, NonBlank] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze for sentiment",
        json_schema_extra={"example": "I absolutely love this product! It exceeded all my expectations."},
    )
    model: Optional[str] = Field(
        default=None,
        description="Model key from registry (e.g., 'twitter_roberta_multilingual'). "
        "If not specified, uses the default model.",
        json_schema_extra={"example": "twitter_roberta_multilingual"},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "I absolutely love this product! It exceeded all my expectations.",
                    "model": "twitter_roberta_multilingual",
                },
                {
                    "text": "The service was terrible and I'm very disappointed.",
                },
            ]
        }
    }


class BatchTextRequest(BaseModel):
    """Request model for batch text processing"""

    texts: List[BatchText] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to process (max 100 items, each 1-5000 characters)",
        json_schema_extra={
            "example": [
                "Great product, highly recommend!",
                "Terrible experience, would not buy again.",
                "It's okay, nothing special.",
            ]
        },
    )
    model: Optional[str] = Field(
        default=None,
        description="Model key from registry. If not specified, uses the default model.",
    )


class SummarizationRequest(BaseModel):
    """Request model for text summarization"""

    text: Annotated[str, NonBlank] = Field(
        ...,
        min_length=50,
        max_length=50000,
        description="Text to summarize (minimum 50 characters for meaningful summarization)",
        json_schema_extra={
            "example": (
                "Artificial intelligence (AI) has transformed numerous industries in recent years. "
                "From healthcare to finance, AI systems are being deployed to automate tasks, "
                "improve decision-making, and enhance customer experiences. Machine learning, "
                "a subset of AI, enables computers to learn from data without being explicitly "
                "programmed. Deep learning, which uses neural networks with multiple layers, "
                "has achieved remarkable results in image recognition, natural language processing, "
                "and game playing. As AI continues to advance, ethical considerations become "
                "increasingly important, including issues of bias, privacy, and job displacement."
            )
        },
    )
    model: Optional[str] = Field(
        default=None,
        description="Model key from registry (e.g., 'bart_large_cnn', 't5_base')",
        json_schema_extra={"example": "bart_large_cnn"},
    )
    min_length: Optional[int] = Field(
        default=30,
        ge=10,
        le=500,
        description="Minimum length of the summary in tokens",
    )
    max_length: Optional[int] = Field(
        default=150,
        ge=30,
        le=1000,
        description="Maximum length of the summary in tokens",
    )

    @model_validator(mode="after")
    def _check_lengths(self) -> "SummarizationRequest":
        if self.min_length is not None and self.max_length is not None and self.min_length >= self.max_length:
            raise ValueError("min_length must be smaller than max_length")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": (
                        "Artificial intelligence (AI) has transformed numerous industries. "
                        "From healthcare to finance, AI systems automate tasks and improve "
                        "decision-making. Machine learning enables computers to learn from data. "
                        "Deep learning has achieved remarkable results in image recognition and NLP."
                    ),
                    "model": "bart_large_cnn",
                    "min_length": 30,
                    "max_length": 100,
                }
            ]
        }
    }


class NERRequest(BaseModel):
    """Request model for Named Entity Recognition"""

    text: Annotated[str, NonBlank] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to extract named entities from",
        json_schema_extra={
            "example": (
                "Apple Inc. CEO Tim Cook announced new products at the keynote in Cupertino, "
                "California on September 12, 2024. The company's stock rose 5% following the event."
            )
        },
    )
    model: Optional[str] = Field(
        default=None,
        description="Model key from registry (e.g., 'spacy_sm', 'gliner')",
        json_schema_extra={"example": "spacy_sm"},
    )
    entity_types: Optional[List[str]] = Field(
        default=None,
        description="Filter to specific entity types (e.g., ['PERSON', 'ORG', 'GPE']). "
        "If not specified, returns all entity types.",
        json_schema_extra={"example": ["PERSON", "ORG", "GPE", "DATE"]},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Elon Musk founded SpaceX in 2002 in Hawthorne, California.",
                    "entity_types": ["PERSON", "ORG", "DATE", "GPE"],
                }
            ]
        }
    }


class SimilarityRequest(BaseModel):
    """Request model for semantic similarity computation"""

    text1: Annotated[str, NonBlank] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="First text for comparison",
        json_schema_extra={"example": "The cat sat on the mat."},
    )
    text2: Annotated[str, NonBlank] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Second text for comparison",
        json_schema_extra={"example": "A cat was sitting on a rug."},
    )
    model: Optional[str] = Field(
        default=None,
        description="Embedding model key from registry (e.g., 'all_minilm_l6', 'all_mpnet_base')",
        json_schema_extra={"example": "all_minilm_l6"},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text1": "The weather is beautiful today.",
                    "text2": "It's a lovely sunny day outside.",
                    "model": "all_minilm_l6",
                }
            ]
        }
    }


class QARequest(BaseModel):
    """Request model for Question Answering"""

    question: Annotated[str, NonBlank] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Question to answer",
        json_schema_extra={"example": "What is the capital of France?"},
    )
    context: Annotated[str, NonBlank] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Context passage containing the answer",
        json_schema_extra={
            "example": (
                "France is a country in Western Europe. Its capital is Paris, which is also "
                "the largest city in France. Paris is known for the Eiffel Tower and the Louvre Museum."
            )
        },
    )
    model: Optional[str] = Field(
        default=None,
        description="QA model key from registry (e.g., 'distilbert_squad', 'roberta_squad')",
        json_schema_extra={"example": "distilbert_squad"},
    )
    top_k: Optional[int] = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of top answers to return",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Who founded Microsoft?",
                    "context": (
                        "Microsoft Corporation was founded by Bill Gates and Paul Allen on April 4, 1975. "
                        "The company is headquartered in Redmond, Washington. Microsoft develops and sells "
                        "computer software, consumer electronics, and personal computers."
                    ),
                    "top_k": 1,
                }
            ]
        }
    }


class CustomerTextItem(BaseModel):
    """Single text item for customer intelligence analysis"""

    id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional caller-provided item identifier",
        json_schema_extra={"example": "review-001"},
    )
    text: Annotated[str, NonBlank] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Customer feedback, review, support note, or article text",
        json_schema_extra={"example": "The onboarding was smooth, but support took two days to reply."},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional item metadata returned unchanged in the response",
        json_schema_extra={"example": {"source": "app_store", "region": "US"}},
    )


class CustomerIntelligenceRequest(BaseModel):
    """Request model for portfolio-grade customer intelligence analysis"""

    items: List[CustomerTextItem] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Text items to analyze",
    )
    sentiment_model: Optional[str] = Field(
        default=None,
        description="Sentiment model registry key. Defaults to twitter_roberta_multilingual.",
        json_schema_extra={"example": "twitter_roberta_multilingual"},
    )
    ner_model: Optional[str] = Field(
        default=None,
        description="NER model registry key. Defaults to spacy_sm.",
        json_schema_extra={"example": "spacy_sm"},
    )
    summary_model: Optional[str] = Field(
        default=None,
        description="Summarization model registry key. Defaults to bart_large_cnn.",
        json_schema_extra={"example": "bart_large_cnn"},
    )
    include_summary: bool = Field(
        default=True,
        description="Generate a portfolio-level summary when enough text is available",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "items": [
                    {
                        "id": "review-001",
                        "text": "The product is excellent, but delivery took too long.",
                        "metadata": {"source": "retail_reviews"},
                    },
                    {
                        "id": "review-002",
                        "text": "Support resolved my issue quickly and the mobile app is easy to use.",
                        "metadata": {"source": "support_ticket"},
                    },
                ],
                "include_summary": True,
            }
        }
    }
