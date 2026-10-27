# Operations Guide

## Canonical Local Run

```bash
docker compose up api redis
```

Open:

- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Readiness: http://localhost:8000/ready
- Metrics: http://localhost:8000/metrics

The Compose API service uses `.env.example`. The included API key is only for local demos.

## Local Python Run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

export NLP_API__JWT_SECRET=replace-with-a-long-random-secret
export NLP_API__API_KEYS='{"demo-customer-intel-key":{"name":"Portfolio Demo","role":"admin","rate_limit":1000,"enabled":true}}'

python launch_ui.py api
```

Use `python launch_ui.py api --host 0.0.0.0 --port 8000` when binding outside localhost.

Do not `source .env.example` in a shell: bash strips the quotes inside the JSON value of `NLP_API__API_KEYS` and the API starts without valid keys. Copy it to `.env` (read by the settings loader) or export the values as shown.

## Required Environment

For shared or public deployments, configure:

- `NLP_API__JWT_SECRET`
- `NLP_API__API_KEYS`
- `NLP_REDIS_URL`
- `NLP_MAX_CACHED_MODELS`
- `NLP_LOG_LEVEL`
- `NLP_JSON_LOGS`

Optional:

- `NLP_API__CORS_ORIGINS`
- `NLP_OTEL_ENABLED`
- `NLP_CACHE_DIR`
- `NLP_DEVICE`

## Readiness And Health

- `/health`: liveness only; should remain cheap and always return healthy while the process is running.
- `/ready`: verifies configuration, registry, auth configuration, model cache initialization, and Redis availability when Redis is configured. Returns `503` with the failing check in `checks` when any check fails, so load balancers and Kubernetes probes stop routing traffic.
- `/api/v1/status`: request counts, average latency by endpoint, model-cache stats, and uptime.
- `/metrics`: Prometheus metrics.

## Monitoring Signals

Watch these first:

- HTTP request rate and latency from `http_requests_total` and `http_request_duration_seconds`.
- Inference latency and error counts from `model_inference_duration_seconds` and `model_inference_total`.
- Cache behavior from `model_cache_hits_total`, `model_cache_misses_total`, `model_cache_evictions_total`, and `model_cache_size`.
- Auth failures from `auth_attempts_total`.
- Rate-limit rejections from `rate_limit_hits_total` (labelled by `client_type`: `api_key`, `jwt`, `anonymous`).

## Security Defaults

- No API keys are hardcoded in Python source.
- Demo credentials live in `.env.example` for local use.
- Inference endpoints require API key or JWT.
- Public deployments must use HTTPS and real secrets.
- Avoid debug logging of request bodies in environments that may process sensitive customer text.

## Cloud Deployment Path

A simple production-like path:

1. Build the Docker image in CI.
2. Push the image to a registry.
3. Deploy one API service plus managed Redis.
4. Configure `NLP_API__JWT_SECRET`, `NLP_API__API_KEYS`, and `NLP_REDIS_URL` as platform secrets.
5. Mount or persist a Hugging Face cache volume if the platform supports it.
6. Expose port `8000` behind HTTPS.
7. Scrape `/metrics`.
8. Add alerts for elevated 5xx rate, p95 latency, auth failures, and Redis unavailability.

## Capacity Notes

- First request to a cold model includes download/load time.
- Each API replica has its own process-local model cache.
- `NLP_MAX_CACHED_MODELS` should be sized against available memory.
- Larger summarization and QA models have materially higher memory and latency costs than spaCy NER.

## Runbooks

High latency:

- Check whether the request is a cold start.
- Inspect `/api/v1/status` and cache metrics.
- Run `python scripts/benchmark_api.py --api-url http://localhost:8000 --api-key <key>`.
- Increase cache size or pre-warm critical models if memory allows.

Unexpected 401s:

- Verify `NLP_API__API_KEYS` JSON syntax.
- Confirm the key is enabled.
- Confirm JWTs were signed with the current `NLP_API__JWT_SECRET`.

Unexpected 429s:

- Check the `X-RateLimit-*` headers and `Retry-After`; `X-RateLimit-Reset` is the end of the current window.
- Requests without valid credentials (including unknown keys) share the per-IP anonymous limit (`NLP_API__ANONYMOUS_RATE_LIMIT`); behind a proxy, start uvicorn with `--proxy-headers` so client IPs are real.
- A JWT counts against the rate limit of the key it was issued for.
- Confirm Redis availability if deployed with multiple replicas (`X-RateLimit-Backend: memory` means counters are per replica).
- Inspect `rate_limit_hits_total`.

NER failures:

- Confirm `en_core_web_sm` is installed.
- Confirm the requested NER registry key exists.

Model loading failures:

- Confirm network/cache access for Hugging Face downloads.
- Confirm model registry task and model ID are compatible with the pinned `transformers` version.
- Check memory limits and cache eviction behavior.

## Optional UIs

UIs are not part of the canonical runtime. Use them for demos:

```bash
docker compose --profile ui up ui-launcher   # all UIs in one app on :7860
python launch_ui.py all                      # same, locally
python launch_ui.py sentiment                # a single UI
```

The combined app shares one model cache, so `NLP_MAX_CACHED_MODELS` bounds memory across every tool. UIs bind to `NLP_SERVER_NAME` (127.0.0.1 by default).
