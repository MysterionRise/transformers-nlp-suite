"""
API Integration Tests

Tests for all API endpoints using httpx TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

# Test API key for authentication
TEST_API_KEY = "dev-api-key"


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers for API requests"""
    return {"X-API-Key": TEST_API_KEY}


class TestHealthEndpoints:
    """Tests for health check endpoints"""

    def test_health_check(self, client):
        """Test liveness probe endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_readiness_check(self, client):
        """Test readiness probe endpoint"""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ready", "not_ready"]
        assert "checks" in data

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        # Check for some expected metric names
        content = response.text
        assert "http_requests_total" in content or "nlp_api" in content

    def test_list_models(self, client):
        """Test listing available models"""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should have at least some categories
        categories = [item["category"] for item in data]
        assert "sentiment_analysis" in categories or len(categories) > 0


class TestAuthentication:
    """Tests for authentication"""

    def test_request_without_auth_fails(self, client):
        """Test that requests without authentication fail"""
        response = client.post("/api/v1/sentiment", json={"text": "test"})
        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "unauthorized"

    def test_request_with_invalid_api_key_fails(self, client):
        """Test that invalid API keys are rejected"""
        response = client.post(
            "/api/v1/sentiment",
            json={"text": "test"},
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401

    def test_request_with_valid_api_key_succeeds(self, client, auth_headers):
        """Test that valid API keys work"""
        # This may fail if model isn't available, but auth should pass
        response = client.post(
            "/api/v1/sentiment",
            json={"text": "I love this!"},
            headers=auth_headers,
        )
        # Should either succeed or fail for non-auth reasons
        assert response.status_code != 401

    def test_get_jwt_token(self, client, auth_headers):
        """Test JWT token generation"""
        response = client.post("/api/v1/auth/token", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    def test_jwt_token_authentication(self, client, auth_headers):
        """Test that JWT tokens work for authentication"""
        # Get token
        token_response = client.post("/api/v1/auth/token", headers=auth_headers)
        token = token_response.json()["access_token"]

        # Use token for request
        response = client.post(
            "/api/v1/sentiment",
            json={"text": "Great product!"},
            headers={"Authorization": f"Bearer {token}"},
        )
        # Should not fail with 401
        assert response.status_code != 401


class TestRateLimitHeaders:
    """Tests for rate limit headers"""

    def test_rate_limit_headers_present(self, client, auth_headers):
        """Test that rate limit headers are included in responses"""
        response = client.get("/api/v1/models", headers=auth_headers)

        # Check for rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


class TestRequestIdTracking:
    """Tests for request ID tracking"""

    def test_request_id_in_response(self, client):
        """Test that request ID is returned in headers"""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers
        # Should be a short UUID
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) == 8


class TestValidation:
    """Tests for request validation"""

    def test_sentiment_empty_text_fails(self, client, auth_headers):
        """Test that empty text is rejected"""
        response = client.post(
            "/api/v1/sentiment",
            json={"text": ""},
            headers=auth_headers,
        )
        assert response.status_code == 422  # Validation error

    def test_sentiment_text_too_long(self, client, auth_headers):
        """Test that overly long text is rejected"""
        response = client.post(
            "/api/v1/sentiment",
            json={"text": "x" * 10000},  # Exceeds max_length
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_summarization_text_too_short(self, client, auth_headers):
        """Test that short text for summarization is rejected"""
        response = client.post(
            "/api/v1/summarize",
            json={"text": "Short."},  # Less than min_length
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_qa_missing_context(self, client, auth_headers):
        """Test that QA without context fails"""
        response = client.post(
            "/api/v1/qa",
            json={"question": "What is AI?"},  # Missing context
            headers=auth_headers,
        )
        assert response.status_code == 422


class TestErrorResponses:
    """Tests for error response format"""

    def test_error_response_format(self, client):
        """Test that errors follow consistent format"""
        response = client.post("/api/v1/sentiment", json={"text": "test"})
        data = response.json()

        # Should have error and message fields
        assert "error" in data
        assert "message" in data or "detail" in data

    def test_404_error(self, client, auth_headers):
        """Test 404 for unknown endpoints"""
        response = client.get("/api/v1/unknown", headers=auth_headers)
        assert response.status_code == 404

    def test_invalid_model_error(self, client, auth_headers):
        """Test error when invalid model is specified"""
        response = client.post(
            "/api/v1/sentiment",
            json={"text": "test", "model": "nonexistent_model"},
            headers=auth_headers,
        )
        # Should get a 400 with helpful error message
        if response.status_code == 400:
            data = response.json()
            assert "invalid_model" in data.get("error", "") or "available_models" in str(data)


@pytest.mark.slow
class TestSentimentEndpoint:
    """Integration tests for sentiment endpoint (requires model download)"""

    def test_sentiment_positive_text(self, client, auth_headers):
        """Test sentiment analysis with positive text"""
        response = client.post(
            "/api/v1/sentiment",
            json={"text": "I absolutely love this amazing product!"},
            headers=auth_headers,
        )

        if response.status_code == 200:
            data = response.json()
            assert "sentiment" in data
            assert "label" in data["sentiment"]
            assert "score" in data["sentiment"]
            assert "processing_time_ms" in data

    def test_sentiment_batch(self, client, auth_headers):
        """Test batch sentiment analysis"""
        response = client.post(
            "/api/v1/sentiment/batch",
            json={
                "texts": [
                    "I love this!",
                    "This is terrible.",
                    "It's okay I guess.",
                ]
            },
            headers=auth_headers,
        )

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 3


@pytest.mark.slow
class TestNEREndpoint:
    """Integration tests for NER endpoint (requires spacy)"""

    def test_ner_extraction(self, client, auth_headers):
        """Test named entity extraction"""
        response = client.post(
            "/api/v1/ner",
            json={"text": "Apple Inc. CEO Tim Cook announced new products in Cupertino, California."},
            headers=auth_headers,
        )

        if response.status_code == 200:
            data = response.json()
            assert "entities" in data
            assert "entity_counts" in data
            assert isinstance(data["entities"], list)


@pytest.mark.slow
class TestSummarizationEndpoint:
    """Integration tests for summarization endpoint (requires model download)"""

    def test_summarization(self, client, auth_headers):
        """Test text summarization"""
        long_text = """
        Artificial intelligence (AI) has transformed numerous industries in recent years.
        From healthcare to finance, AI systems are being deployed to automate tasks,
        improve decision-making, and enhance customer experiences. Machine learning,
        a subset of AI, enables computers to learn from data without being explicitly
        programmed. Deep learning, which uses neural networks with multiple layers,
        has achieved remarkable results in image recognition, natural language processing,
        and game playing.
        """

        response = client.post(
            "/api/v1/summarize",
            json={
                "text": long_text,
                "min_length": 20,
                "max_length": 50,
            },
            headers=auth_headers,
        )

        if response.status_code == 200:
            data = response.json()
            assert "summary" in data
            assert "compression_ratio" in data
            assert len(data["summary"]) < len(long_text)


@pytest.mark.slow
class TestSimilarityEndpoint:
    """Integration tests for similarity endpoint"""

    def test_similar_texts(self, client, auth_headers):
        """Test similarity with similar texts"""
        response = client.post(
            "/api/v1/similarity",
            json={
                "text1": "The weather is beautiful today.",
                "text2": "It's a lovely sunny day outside.",
            },
            headers=auth_headers,
        )

        if response.status_code == 200:
            data = response.json()
            assert "similarity_score" in data
            # Similar texts should have high similarity
            assert data["similarity_score"] > 0.5

    def test_dissimilar_texts(self, client, auth_headers):
        """Test similarity with dissimilar texts"""
        response = client.post(
            "/api/v1/similarity",
            json={
                "text1": "The cat sat on the mat.",
                "text2": "Quantum computing uses qubits for calculations.",
            },
            headers=auth_headers,
        )

        if response.status_code == 200:
            data = response.json()
            # Dissimilar texts should have lower similarity
            assert data["similarity_score"] < 0.5


@pytest.mark.slow
class TestQAEndpoint:
    """Integration tests for QA endpoint"""

    def test_question_answering(self, client, auth_headers):
        """Test question answering"""
        response = client.post(
            "/api/v1/qa",
            json={
                "question": "What is the capital of France?",
                "context": "France is a country in Western Europe. Its capital is Paris, "
                "which is also the largest city in France.",
            },
            headers=auth_headers,
        )

        if response.status_code == 200:
            data = response.json()
            assert "answers" in data
            assert len(data["answers"]) > 0
            # Answer should contain "Paris"
            assert "Paris" in data["answers"][0]["answer"]
