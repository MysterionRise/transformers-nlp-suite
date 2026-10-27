"""
Regression tests for rate limiting, auth edge cases, error envelopes, readiness,
OpenAPI security and input limits.

Each test pins a behaviour that was broken before: throttled requests returned
500 instead of 429, made-up API keys got their own larger bucket, JWT users got
the anonymous limit, /ready answered 200 when not ready, Swagger "Authorize"
never sent credentials, and long inputs crashed inference.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from api.main import app
from api.middleware import rate_limit
from tests.unit.ui.conftest import FakePipeline, FakeTokenizer
from utils.metrics import auth_attempts_total

SECRET = "test-jwt-secret"
KEYS = {
    "admin-key": {"name": "Admin", "role": "admin", "rate_limit": 1000, "enabled": True},
    "low-key": {"name": "Low", "role": "user", "rate_limit": 2, "enabled": True},
    "five-key": {"name": "Five", "role": "user", "rate_limit": 5, "enabled": True},
    "disabled-key": {"name": "Off", "role": "user", "rate_limit": 100, "enabled": False},
}


@pytest.fixture(autouse=True)
def api_settings():
    from config.settings import get_settings

    settings = get_settings()
    saved = (
        settings.api.api_keys,
        settings.api.jwt_secret,
        settings.api.anonymous_rate_limit,
        settings.api.jwt_expiration_hours,
        settings.redis_url,
    )
    settings.api.api_keys = dict(KEYS)
    settings.api.jwt_secret = SECRET
    settings.api.anonymous_rate_limit = 3
    settings.redis_url = None
    rate_limit.clear_rate_limit_storage()
    yield settings
    (
        settings.api.api_keys,
        settings.api.jwt_secret,
        settings.api.anonymous_rate_limit,
        settings.api.jwt_expiration_hours,
        settings.redis_url,
    ) = saved
    rate_limit.clear_rate_limit_storage()


@pytest.fixture
def client():
    return TestClient(app)


def _token(sub: str, **claims) -> str:
    payload = {"sub": sub, "exp": datetime.now(timezone.utc) + timedelta(hours=1), **claims}
    return jwt.encode(payload, SECRET, algorithm="HS256")


def _counter(method: str, status: str) -> float:
    return auth_attempts_total.labels(method=method, status=status)._value.get()


class TestRateLimiting:
    def test_over_limit_is_429_with_headers_request_id_and_cors(self, client):
        headers = {"X-API-Key": "low-key", "Origin": "http://localhost:3000"}
        assert client.get("/api/v1/models", headers=headers).status_code == 200
        assert client.get("/api/v1/models", headers=headers).status_code == 200
        response = client.get("/api/v1/models", headers=headers)

        assert response.status_code == 429
        body = response.json()
        assert body["error"] == "rate_limit_exceeded"
        assert body["details"]["limit"] == 2
        assert body["request_id"] == response.headers["X-Request-ID"]
        assert int(response.headers["Retry-After"]) >= 1
        assert response.headers["X-RateLimit-Limit"] == "2"
        assert response.headers["X-RateLimit-Remaining"] == "0"
        assert response.headers["X-RateLimit-Backend"] == "memory"
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    def test_reset_header_is_the_window_end(self, client):
        first = client.get("/api/v1/models", headers={"X-API-Key": "admin-key"})
        second = client.get("/api/v1/models", headers={"X-API-Key": "admin-key"})
        assert first.headers["X-RateLimit-Reset"] == second.headers["X-RateLimit-Reset"]
        assert int(first.headers["X-RateLimit-Reset"]) % 60 == 0
        assert int(second.headers["X-RateLimit-Remaining"]) == int(first.headers["X-RateLimit-Remaining"]) - 1

    def test_made_up_keys_share_the_anonymous_ip_bucket(self, client):
        statuses = [client.get("/api/v1/models", headers={"X-API-Key": f"bogus-{i}"}).status_code for i in range(4)]
        assert statuses == [200, 200, 200, 429]

    def test_jwt_uses_the_key_bucket_and_limit(self, client):
        token = client.post("/api/v1/auth/token", headers={"X-API-Key": "five-key"}).json()["access_token"]
        by_jwt = client.get("/api/v1/models", headers={"Authorization": f"Bearer {token}"})
        by_key = client.get("/api/v1/models", headers={"X-API-Key": "five-key"})
        assert by_jwt.headers["X-RateLimit-Limit"] == "5"
        # token request + JWT request + key request share one budget
        assert by_key.headers["X-RateLimit-Remaining"] == "2"

    def test_bogus_key_with_valid_jwt_is_billed_to_the_token_owner(self, client):
        token = _token("low-key")
        headers = {"Authorization": f"Bearer {token}"}
        for i in range(2):
            assert client.get("/api/v1/models", headers={**headers, "X-API-Key": f"rotating-{i}"}).status_code == 200
        assert client.get("/api/v1/models", headers={**headers, "X-API-Key": "rotating-x"}).status_code == 429

    def test_exempt_paths_are_not_counted(self, client):
        for _ in range(5):
            assert client.get("/health").status_code == 200
        assert "X-RateLimit-Limit" not in client.get("/health").headers

    def test_memory_store_is_bounded_and_prunes_old_windows(self):
        store = rate_limit.MemoryWindowStore(max_buckets=5)
        for i in range(20):
            store.hit(f"ip:{i}", window=1)
        assert len(store) == 5
        store.hit("ip:new", window=2)
        assert len(store) == 1  # previous-window buckets are dropped first
        assert store.hit("ip:new", window=2) == 2
        assert store.hit("ip:new", window=3) == 1

    def test_unreachable_redis_is_not_retried_on_every_request(self, client, api_settings):
        api_settings.redis_url = "redis://redis.invalid:6379/0"
        with patch("redis.Redis.from_url", side_effect=ConnectionError("unreachable")) as from_url:
            for _ in range(3):
                response = client.get("/api/v1/models", headers={"X-API-Key": "admin-key"})
                assert response.status_code == 200
                assert response.headers["X-RateLimit-Backend"] == "memory"
        assert from_url.call_count == 1


class TestAuthentication:
    def test_api_key_request_counts_one_auth_attempt(self, client, mock_inference_service):
        before = _counter("api_key", "success")
        client.post("/api/v1/sentiment", headers={"X-API-Key": "admin-key"}, json={"text": "fine"})
        assert _counter("api_key", "success") - before == 1

    def test_jwt_request_counts_only_a_jwt_attempt(self, client, mock_inference_service):
        token = _token("admin-key")
        key_before, jwt_before = _counter("api_key", "success"), _counter("jwt", "success")
        client.post("/api/v1/sentiment", headers={"Authorization": f"Bearer {token}"}, json={"text": "fine"})
        assert _counter("api_key", "success") == key_before
        assert _counter("jwt", "success") - jwt_before == 1

    def test_public_endpoint_with_key_records_no_attempt(self, client):
        before = _counter("api_key", "success")
        client.get("/api/v1/models", headers={"X-API-Key": "admin-key"})
        assert _counter("api_key", "success") == before

    @pytest.mark.parametrize(
        "token",
        [
            jwt.encode({"sub": "admin-key"}, SECRET, algorithm="HS256"),  # no exp claim
            jwt.encode({"sub": "admin-key", "exp": 1}, SECRET, algorithm="HS256"),  # expired
            jwt.encode({"sub": "admin-key", "exp": 4102444800}, "other-secret", algorithm="HS256"),
            _token("disabled-key"),
            _token("not-a-key"),
        ],
        ids=["no-exp", "expired", "wrong-secret", "disabled-key", "unknown-key"],
    )
    def test_invalid_tokens_are_rejected(self, client, token):
        response = client.post("/api/v1/sentiment", headers={"Authorization": f"Bearer {token}"}, json={"text": "fine"})
        assert response.status_code == 401

    def test_disabled_key_is_rejected(self, client):
        response = client.post("/api/v1/sentiment", headers={"X-API-Key": "disabled-key"}, json={"text": "x"})
        assert response.status_code == 401

    def test_token_endpoint_without_secret_is_503_and_tokens_fail_closed(self, client, api_settings):
        token = _token("admin-key")
        api_settings.api.jwt_secret = ""
        response = client.post("/api/v1/auth/token", headers={"X-API-Key": "admin-key"})
        assert response.status_code == 503
        assert response.json()["error"] == "jwt_not_configured"
        assert (
            client.post(
                "/api/v1/sentiment", headers={"Authorization": f"Bearer {token}"}, json={"text": "x"}
            ).status_code
            == 401
        )

    def test_expires_in_follows_settings(self, client, api_settings):
        api_settings.api.jwt_expiration_hours = 2
        body = client.post("/api/v1/auth/token", headers={"X-API-Key": "admin-key"}).json()
        assert body["expires_in"] == 7200
        claims = jwt.get_unverified_claims(body["access_token"])
        assert 7000 < claims["exp"] - datetime.now(timezone.utc).timestamp() <= 7200


class TestErrorsAndReadiness:
    def test_ready_is_503_when_auth_not_configured(self, client, api_settings):
        api_settings.api.api_keys = {}
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"
        assert response.json()["checks"]["auth_configured"] is False

    def test_ready_is_200_when_configured(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_validation_error_uses_envelope(self, client):
        response = client.post("/api/v1/sentiment", headers={"X-API-Key": "admin-key"}, json={"text": ""})
        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "validation_error"
        assert body["details"][0]["field"] == "text"
        assert body["request_id"] == response.headers["X-Request-ID"]

    @pytest.mark.parametrize(
        "path,payload",
        [
            ("/api/v1/sentiment", {"text": "   \n\t "}),
            ("/api/v1/sentiment/batch", {"texts": ["fine", "  "]}),
            ("/api/v1/sentiment/batch", {"texts": ["fine", "x" * 5001]}),
            ("/api/v1/summarize", {"text": "word " * 20, "min_length": 100, "max_length": 40}),
            ("/api/v1/qa", {"question": " ", "context": "Paris is in France."}),
        ],
        ids=["blank-text", "blank-batch-item", "long-batch-item", "min-over-max", "blank-question"],
    )
    def test_invalid_inputs_are_rejected(self, client, path, payload):
        response = client.post(path, headers={"X-API-Key": "admin-key"}, json=payload)
        assert response.status_code == 422, response.text

    @pytest.mark.parametrize(
        "method,path,status,code",
        [("get", "/api/v1/nope", 404, "not_found"), ("delete", "/health", 405, "method_not_allowed")],
    )
    def test_routing_errors_use_envelope(self, client, method, path, status, code):
        response = getattr(client, method)(path)
        assert response.status_code == status
        assert response.json()["error"] == code
        assert response.json()["request_id"] == response.headers["X-Request-ID"]


class TestOpenAPISecurity:
    def test_operation_security_references_existing_schemes(self):
        schema = app.openapi()
        schemes = set(schema["components"]["securitySchemes"])
        assert {"ApiKeyAuth", "BearerAuth"} <= schemes
        for path, operations in schema["paths"].items():
            for operation in operations.values():
                for requirement in operation.get("security", []):
                    assert set(requirement) <= schemes, f"{path} references {requirement}"

    def test_inference_and_token_endpoints_are_secured_public_ones_are_not(self):
        schema = app.openapi()
        assert "security" not in schema
        sentiment = schema["paths"]["/api/v1/sentiment"]["post"]["security"]
        assert {"ApiKeyAuth": []} in sentiment and {"BearerAuth": []} in sentiment
        assert {"ApiKeyAuth": []} in schema["paths"]["/api/v1/auth/token"]["post"]["security"]
        assert "security" not in schema["paths"]["/health"]["get"]


class TestInferenceLimits:
    def _service(self, model, category_config=None):
        from utils.inference import InferenceService

        service = InferenceService.__new__(InferenceService)
        config = category_config or SimpleNamespace(model_id="fake/model", prefix=None)
        service.registry = SimpleNamespace(get_model=lambda category, key: config)
        service.cache = SimpleNamespace(get_model=lambda category, key: model)
        return service

    def test_sentiment_clips_long_text_and_returns_all_scores(self):
        model = FakePipeline(
            lambda text, **_: [{"label": "LABEL_1", "score": 0.8}, {"label": "LABEL_0", "score": 0.2}],
            FakeTokenizer(model_max_length=12),
            num_labels=2,
        )
        result = asyncio.run(self._service(model).analyze_sentiment(" ".join(f"w{i}" for i in range(40))))
        assert model.calls[0]["inputs"] == " ".join(f"w{i}" for i in range(10))
        assert model.calls[0]["top_k"] is None
        assert result["sentiment"] == {"label": "positive", "score": 0.8}
        assert [score["label"] for score in result["all_scores"]] == ["positive", "negative"]

    def test_batch_clips_each_text(self):
        model = FakePipeline(
            lambda texts, **_: [[{"label": "positive", "score": 0.9}] for _ in texts],
            FakeTokenizer(model_max_length=5),
        )
        results = asyncio.run(self._service(model).analyze_sentiment_batch(["a b c d e f g", "short"]))
        assert model.calls[0]["inputs"] == ["a b c", "short"]
        assert len(results) == 2

    def test_summarize_applies_prefix_and_clips(self):
        model = FakePipeline(lambda text, **_: [{"summary_text": "sum"}], FakeTokenizer(model_max_length=6))
        config = SimpleNamespace(model_id="t5-base", prefix="summarize: ")
        asyncio.run(self._service(model, config).summarize("one two three four five six seven", "t5_base"))
        assert model.calls[0]["inputs"] == "summarize: one two three"
