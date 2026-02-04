"""
API Routers for NLP endpoints
"""

from . import health, ner, qa, sentiment, similarity, summarization

__all__ = ["sentiment", "summarization", "ner", "similarity", "qa", "health"]
