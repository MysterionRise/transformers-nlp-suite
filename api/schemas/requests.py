"""
Pydantic request models for API endpoints

All request models include validation, examples, and comprehensive documentation
for beautiful OpenAPI/Swagger UI documentation.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    """Request model for sentiment analysis"""

    text: str = Field(
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

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to process (max 100 items)",
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

    text: str = Field(
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

    text: str = Field(
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

    text1: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="First text for comparison",
        json_schema_extra={"example": "The cat sat on the mat."},
    )
    text2: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Second text for comparison",
        json_schema_extra={"example": "A cat was sitting on a rug."},
    )
    model: Optional[str] = Field(
        default=None,
        description="Embedding model key from registry (e.g., 'minilm', 'mpnet')",
        json_schema_extra={"example": "minilm"},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text1": "The weather is beautiful today.",
                    "text2": "It's a lovely sunny day outside.",
                    "model": "minilm",
                }
            ]
        }
    }


class QARequest(BaseModel):
    """Request model for Question Answering"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Question to answer",
        json_schema_extra={"example": "What is the capital of France?"},
    )
    context: str = Field(
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
