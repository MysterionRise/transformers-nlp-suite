"""
Shared inference service for API workflows.

The API uses this layer instead of loading models directly in routers. It keeps
model lookup, caching, output normalization, and metrics in one place.
"""

import asyncio
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import get_model_registry
from utils.labels import normalize_sentiment_label, num_labels_of
from utils.logger import get_logger
from utils.metrics import track_inference_time
from utils.model_cache import get_model_cache
from utils.text_limits import token_limit, truncate_to_tokens

logger = get_logger(__name__)

DEFAULT_SENTIMENT_MODEL = "twitter_roberta_multilingual"
DEFAULT_SUMMARIZATION_MODEL = "bart_large_cnn"
DEFAULT_NER_MODEL = "spacy_sm"
DEFAULT_SIMILARITY_MODEL = "all_minilm_l6"
DEFAULT_QA_MODEL = "distilbert_squad"


class InferenceService:
    """High-level inference operations backed by the central model cache."""

    def __init__(self):
        self.registry = get_model_registry()
        self.cache = get_model_cache()

    def _get_config(self, category: str, model_key: str):
        return self.registry.get_model(category, model_key)

    @staticmethod
    def _log_success(task: str, model_key: str, model_id: str, processing_time_ms: float):
        logger.info(
            f"{task} inference completed with {model_key} ({processing_time_ms:.2f}ms)",
            extra={
                "task": task,
                "model_key": model_key,
                "model_id": model_id,
                "processing_time_ms": processing_time_ms,
                "status": "success",
            },
        )

    @staticmethod
    def _normalize_sentiment_label(label: str, num_labels: Optional[int] = None) -> str:
        return normalize_sentiment_label(label, num_labels)

    @staticmethod
    def _fit_to_model(model: Any, text: str, reserve: int = 0) -> str:
        """Cut text to the model's token limit (models without a tokenizer are left alone)."""
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return text
        clipped, count = truncate_to_tokens(text, tokenizer, max(1, token_limit(model) - reserve))
        if clipped != text:
            logger.info(f"Input truncated from {count} tokens to the model limit", extra={"tokens": count})
        return clipped

    @staticmethod
    def _as_score_list(raw_result: Any) -> List[Dict[str, Any]]:
        if isinstance(raw_result, list):
            if raw_result and isinstance(raw_result[0], list):
                return raw_result[0]
            if raw_result and isinstance(raw_result[0], dict):
                return raw_result
        if isinstance(raw_result, dict):
            return [raw_result]
        return [{"label": "unknown", "score": 0.0}]

    async def analyze_sentiment(self, text: str, model_key: str = DEFAULT_SENTIMENT_MODEL) -> Dict[str, Any]:
        """Run sentiment analysis and normalize labels."""
        model_config = self._get_config("sentiment_analysis", model_key)
        start_time = time.time()

        def _inference():
            with track_inference_time(model_config.model_id, "sentiment"):
                model = self.cache.get_model("sentiment_analysis", model_key)
                return model(self._fit_to_model(model, text), top_k=None), num_labels_of(model)

        raw_result, num_labels = await asyncio.to_thread(_inference)
        all_scores = []
        for item in self._as_score_list(raw_result):
            all_scores.append(
                {
                    "label": self._normalize_sentiment_label(item.get("label", "unknown"), num_labels),
                    "score": float(item.get("score", 0.0)),
                }
            )
        all_scores.sort(key=lambda item: item["score"], reverse=True)

        primary = max(all_scores, key=lambda item: item["score"]) if all_scores else {"label": "unknown", "score": 0.0}
        processing_time_ms = (time.time() - start_time) * 1000
        self._log_success("sentiment", model_key, model_config.model_id, processing_time_ms)
        return {
            "text": text,
            "sentiment": primary,
            "all_scores": all_scores or None,
            "model": model_config.model_id,
            "processing_time_ms": processing_time_ms,
        }

    async def analyze_sentiment_batch(
        self, texts: List[str], model_key: str = DEFAULT_SENTIMENT_MODEL
    ) -> List[Dict[str, Any]]:
        """Run batch sentiment analysis with one model load."""
        model_config = self._get_config("sentiment_analysis", model_key)
        start_time = time.time()

        def _inference():
            with track_inference_time(model_config.model_id, "sentiment_batch"):
                model = self.cache.get_model("sentiment_analysis", model_key)
                fitted = [self._fit_to_model(model, text) for text in texts]
                return model(fitted, top_k=None), num_labels_of(model)

        raw_results, num_labels = await asyncio.to_thread(_inference)
        total_time_ms = (time.time() - start_time) * 1000
        per_text_time_ms = total_time_ms / max(len(texts), 1)
        self._log_success("sentiment_batch", model_key, model_config.model_id, total_time_ms)

        responses = []
        for text, raw_result in zip(texts, raw_results):
            scores = sorted(
                (
                    {
                        "label": self._normalize_sentiment_label(item.get("label", "unknown"), num_labels),
                        "score": float(item.get("score", 0.0)),
                    }
                    for item in self._as_score_list(raw_result)
                ),
                key=lambda item: item["score"],
                reverse=True,
            )
            primary = max(scores, key=lambda item: item["score"]) if scores else {"label": "unknown", "score": 0.0}
            responses.append(
                {
                    "text": text,
                    "sentiment": primary,
                    "all_scores": scores or None,
                    "model": model_config.model_id,
                    "processing_time_ms": per_text_time_ms,
                }
            )
        return responses

    async def summarize(
        self,
        text: str,
        model_key: str = DEFAULT_SUMMARIZATION_MODEL,
        min_length: Optional[int] = 30,
        max_length: Optional[int] = 150,
    ) -> Dict[str, Any]:
        """Generate a summary."""
        model_config = self._get_config("summarization", model_key)
        start_time = time.time()

        def _inference():
            with track_inference_time(model_config.model_id, "summarization"):
                model = self.cache.get_model("summarization", model_key)
                source = (model_config.prefix or "") + text
                return model(
                    self._fit_to_model(model, source),
                    min_length=min_length,
                    max_length=max_length,
                    do_sample=False,
                )

        result = await asyncio.to_thread(_inference)
        if isinstance(result, list) and result:
            summary = result[0].get("summary_text", "")
        elif isinstance(result, dict):
            summary = result.get("summary_text", "")
        else:
            summary = str(result)

        original_length = len(text)
        summary_length = len(summary)
        processing_time_ms = (time.time() - start_time) * 1000
        self._log_success("summarization", model_key, model_config.model_id, processing_time_ms)
        return {
            "original_text": text,
            "summary": summary,
            "original_length": original_length,
            "summary_length": summary_length,
            "compression_ratio": round(summary_length / original_length, 3) if original_length else 0,
            "model": model_config.model_id,
            "processing_time_ms": processing_time_ms,
        }

    async def extract_entities(
        self,
        text: str,
        model_key: str = DEFAULT_NER_MODEL,
        entity_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract named entities using spaCy or a transformers NER pipeline."""
        model_config = self._get_config("ner", model_key)
        framework = (model_config.framework or "").lower()
        start_time = time.time()

        if framework == "spacy" or model_key.startswith("spacy"):

            def _spacy_inference():
                with track_inference_time(model_config.model_id, "ner"):
                    nlp = self.cache.get_spacy_model(model_key)
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
                                    "score": None,
                                }
                            )
                    return entities

            entities = await asyncio.to_thread(_spacy_inference)
        else:

            def _transformer_inference():
                with track_inference_time(model_config.model_id, "ner"):
                    model = self.cache.get_model("ner", model_key)
                    return model(text)

            result = await asyncio.to_thread(_transformer_inference)
            entities = []
            for entity in result:
                label = entity.get("entity_group", entity.get("entity", "UNKNOWN"))
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

        processing_time_ms = (time.time() - start_time) * 1000
        self._log_success("ner", model_key, model_config.model_id, processing_time_ms)
        return {
            "text": text,
            "entities": entities,
            "entity_counts": dict(Counter(entity["label"] for entity in entities)),
            "model": model_config.model_id,
            "processing_time_ms": processing_time_ms,
        }

    async def compute_similarity(
        self,
        text1: str,
        text2: str,
        model_key: str = DEFAULT_SIMILARITY_MODEL,
    ) -> Dict[str, Any]:
        """Compute cosine similarity using a cached sentence-transformers model."""
        model_config = self._get_config("embeddings", model_key)
        start_time = time.time()

        def _inference():
            with track_inference_time(model_config.model_id, "similarity"):
                model = self.cache.get_sentence_transformer("embeddings", model_key)
                return model.encode([text1, text2])

        embeddings = await asyncio.to_thread(_inference)
        e1 = np.array(embeddings[0]).flatten()
        e2 = np.array(embeddings[1]).flatten()
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        similarity = 0.0 if norm1 == 0 or norm2 == 0 else float(np.dot(e1, e2) / (norm1 * norm2))

        processing_time_ms = (time.time() - start_time) * 1000
        self._log_success("similarity", model_key, model_config.model_id, processing_time_ms)
        return {
            "text1": text1,
            "text2": text2,
            "similarity_score": round(similarity, 4),
            "model": model_config.model_id,
            "processing_time_ms": processing_time_ms,
        }

    async def answer_question(
        self,
        question: str,
        context: str,
        model_key: str = DEFAULT_QA_MODEL,
        top_k: int = 1,
    ) -> Dict[str, Any]:
        """Answer a question from a context passage."""
        model_config = self._get_config("question_answering", model_key)
        start_time = time.time()

        def _inference():
            with track_inference_time(model_config.model_id, "qa"):
                model = self.cache.get_model("question_answering", model_key)
                return model(question=question, context=context, top_k=top_k)

        result = await asyncio.to_thread(_inference)
        if isinstance(result, dict):
            result = [result]

        processing_time_ms = (time.time() - start_time) * 1000
        self._log_success("qa", model_key, model_config.model_id, processing_time_ms)
        return {
            "question": question,
            "context": context,
            "answers": [
                {
                    "answer": item.get("answer", ""),
                    "score": float(item.get("score", 0.0)),
                    "start": item.get("start", 0),
                    "end": item.get("end", 0),
                }
                for item in result
            ],
            "model": model_config.model_id,
            "processing_time_ms": processing_time_ms,
        }


_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Get the process-wide inference service."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
