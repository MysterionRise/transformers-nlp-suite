# Architecture

## Product Boundary

Transformers NLP Suite is an API-first service for extracting structured signals from customer-facing text: reviews, support tickets, survey comments, app-store feedback, and market notes.

The primary runtime is FastAPI. Gradio UIs are optional demos built on the same model registry and cache; they are not the main production proof point.

## Runtime Components

```text
Client
  -> FastAPI middleware (outermost first)
      -> CORS
      -> request ID, logging, metrics
      -> rate limiting (per verified identity)
  -> auth dependency (API key or JWT)
  -> API router
      -> validation and response schema
  -> shared inference service
      -> model registry lookup
      -> model-cache access
      -> output normalization
      -> inference timing/logging
  -> model cache
      -> transformers pipeline
      -> sentence-transformers model
      -> spaCy pipeline
```

## API Surface

Core endpoints:

- `POST /api/v1/customer-intelligence/analyze`
- `POST /api/v1/sentiment`
- `POST /api/v1/sentiment/batch`
- `POST /api/v1/summarize`
- `POST /api/v1/ner`
- `POST /api/v1/similarity`
- `POST /api/v1/qa`
- `POST /api/v1/auth/token`

Operational endpoints:

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /api/v1/status`
- `GET /api/v1/models`
- `GET /api/v1/models/{category}`

## Configuration

`config/settings.py` uses Pydantic settings with `NLP_` prefix and nested environment support through `__`.

Examples:

- `NLP_API__JWT_SECRET`
- `NLP_API__API_KEYS`
- `NLP_API__CORS_ORIGINS`
- `NLP_REDIS_URL`
- `NLP_MAX_CACHED_MODELS`
- `NLP_JSON_LOGS`
- `NLP_OTEL_ENABLED`

The model registry lives in `config/models.yaml`; the default API model keys live in `utils/inference.py`.

## Model Loading

Routers do not instantiate models. They call `utils.inference.InferenceService`, which handles:

- registry key validation
- access to the central cache
- clipping input to the model's real token limit (from the model config, since many tokenizers report no limit)
- normalized response shapes and sentiment labels
- task-level inference timing
- structured inference logs

`utils/model_cache.py` dispatches on each registry entry's `framework`:

- Hugging Face transformers pipelines for sentiment, summarization, QA, zero-shot, translation and vision tasks
- sentence-transformers for semantic similarity
- spaCy for fast local NER
- GLiNER for zero-shot NER

Models are loaded lazily and evicted with a process-local LRU strategy. Cache bookkeeping is guarded by a lock and each model has its own load lock, so concurrent requests for the same cold model load it once. Registry entries can pin a `revision`. Downloads use the Hugging Face cache (`HF_HOME`) unless `NLP_CACHE_DIR` is set.

## Auth And Rate Limiting

Inference endpoints require an API key or JWT. `/api/v1/auth/token` exchanges an API key for a JWT; tokens must carry `exp`, and nothing validates while `NLP_API__JWT_SECRET` is empty.

Rate limiting counts requests in fixed windows per verified identity: a valid API key has its own bucket and `rate_limit`, a JWT shares its key's bucket, and everything else (no credentials, unknown keys, bad tokens) shares a per-IP anonymous bucket. Over-limit requests receive `429` with `Retry-After`.

The counter backend is chosen at runtime:

- Redis fixed-window counters when `NLP_REDIS_URL` is configured and reachable (250 ms socket timeouts, 30 s back-off after a failure)
- a bounded in-memory store (10,000 buckets) for tests, local runs, or while Redis is unavailable

The memory fallback is intentionally not a multi-replica production strategy.

## Observability

Implemented observability paths:

- request ID in response headers
- request logs with route, status code, latency, auth method, and user role
- inference logs with task, model key, model ID, latency, and status
- Prometheus metrics for HTTP requests, inference, cache hits/misses/evictions, auth attempts, errors, and rate-limit hits
- `/api/v1/status` with request counts, endpoint counts, average latency summaries, cache stats, and uptime
- optional OpenTelemetry FastAPI instrumentation behind `NLP_OTEL_ENABLED=false`

## Testing Strategy

Fast tests mock inference so they do not download models. They verify auth (including expired, unsigned and revoked tokens), validation and error envelopes, response contracts, rate limiting (429s, bucket identity, Redis back-off), OpenAPI security, metrics/status shapes, and workflow aggregation.

Slow tests exercise selected real-model paths on Python 3.11 in CI. CI also runs fast tests on Python 3.11, 3.12, and 3.13.

## Demo UIs

The Gradio UIs in `ui/` share one shell, `ui/common.py` (see ADR 0004):

- model menus are generated from `config/models.yaml`
- models load through `utils/model_cache.py`, so the UIs and the API share weights and memory limits
- errors surface as Gradio error toasts, and truncation to a model limit is announced with a warning
- host, port, share links and theme come from `Settings`

`ui/catalog.py` lists titles and ports; `ui/app.py` renders every UI as a tab of one app, which is how `docker compose --profile ui up ui-launcher` and `python launch_ui.py all` run them.

## Tradeoffs

- Lazy loading creates cold-start latency, but keeps startup practical for local and portfolio environments.
- The cache is process-local; horizontal deployments need per-replica memory planning and cache warming.
- Redis fixed-window rate limiting is simple and inspectable, but not as smooth as token-bucket or sliding-window algorithms.
- The customer-intelligence workflow runs task models sequentially per item today. That is easier to reason about and test, but batch/parallel execution is a future performance improvement.

## Production Gap

This is production-minded portfolio software, not a regulated production service. Before real customer deployment, add domain-specific eval data, privacy review, deployment secrets management, HTTPS, persistent tracing/log export, SLOs, alerting, and model-risk review for the target use case.
