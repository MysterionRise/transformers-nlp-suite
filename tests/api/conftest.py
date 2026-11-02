"""Fixtures shared by the API test modules."""

import pytest


@pytest.fixture
def mock_inference_service(monkeypatch):
    """Mock shared inference service for fast API tests."""

    class FakeInferenceService:
        async def analyze_sentiment(self, text, model_key="twitter_roberta_multilingual"):
            return {
                "text": text,
                "sentiment": {"label": "positive", "score": 0.91},
                "all_scores": [{"label": "positive", "score": 0.91}],
                "model": "fake-sentiment-model",
                "processing_time_ms": 1.0,
            }

        async def analyze_sentiment_batch(self, texts, model_key="twitter_roberta_multilingual"):
            return [await self.analyze_sentiment(text, model_key) for text in texts]

        async def summarize(self, text, model_key="bart_large_cnn", min_length=30, max_length=150):
            return {
                "original_text": text,
                "summary": "Customers like the product but mention delivery delays.",
                "original_length": len(text),
                "summary_length": 57,
                "compression_ratio": 0.5,
                "model": "fake-summary-model",
                "processing_time_ms": 1.0,
            }

        async def extract_entities(self, text, model_key="spacy_sm", entity_types=None):
            return {
                "text": text,
                "entities": [
                    {
                        "text": "Apple",
                        "label": "ORG",
                        "start": 0,
                        "end": 5,
                        "score": None,
                    }
                ],
                "entity_counts": {"ORG": 1},
                "model": "fake-ner-model",
                "processing_time_ms": 1.0,
            }

        async def compute_similarity(self, text1, text2, model_key="all_minilm_l6"):
            score = 0.83 if "weather" in text1.lower() else 0.21
            return {
                "text1": text1,
                "text2": text2,
                "similarity_score": score,
                "model": "fake-embedding-model",
                "processing_time_ms": 1.0,
            }

        async def answer_question(self, question, context, model_key="distilbert_squad", top_k=1):
            return {
                "question": question,
                "context": context,
                "answers": [{"answer": "Paris", "score": 0.95, "start": 52, "end": 57}],
                "model": "fake-qa-model",
                "processing_time_ms": 1.0,
            }

    service = FakeInferenceService()

    for module_path in [
        "api.routers.sentiment.get_inference_service",
        "api.routers.summarization.get_inference_service",
        "api.routers.ner.get_inference_service",
        "api.routers.similarity.get_inference_service",
        "api.routers.qa.get_inference_service",
        "api.routers.customer_intelligence.get_inference_service",
    ]:
        monkeypatch.setattr(module_path, lambda service=service: service)

    return service
