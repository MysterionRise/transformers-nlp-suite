# Transformers NLP Suite

[![CI](https://github.com/MysterionRise/transformers-nlp-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/MysterionRise/transformers-nlp-suite/actions/workflows/ci.yml)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

API-first NLP service for customer-intelligence work: it turns customer reviews, support notes and market text into structured signals with Hugging Face, sentence-transformers and spaCy models. The main proof point is the FastAPI service: authenticated endpoints, a shared model cache, per-client rate limiting backed by Redis, Prometheus metrics, structured logs, a Docker Compose runtime and CI-covered tests.

Ten Gradio demo UIs sit on top of the same model registry and cache. They are optional; the product story is the API.

## What It Does

- Analyzes sentiment for individual or batched customer text.
- Extracts named entities with spaCy or transformer-backed NER models.
- Computes semantic similarity with sentence-transformers embeddings.
- Generates summaries and extractive QA answers from supplied text.
- Combines these into `/api/v1/customer-intelligence/analyze` for customer feedback analysis.
- Exposes runtime state through `/health`, `/ready`, `/metrics`, and `/api/v1/status`.

## Current Status

Implemented and verified:

- Docker Compose API stack (`docker compose up api redis`), built and smoke-tested by the CI `docker` job
- API key and JWT authentication, with Swagger UI's **Authorize** button working for "Try it out"
- Nested environment loading for `NLP_API__...` settings
- Shared inference layer over transformers, sentence-transformers, and spaCy, with inputs cut to each model's token limit
- Thread-safe LRU model cache (one load per model under concurrency) with hit, miss, eviction, and size metrics
- Per-client rate limiting: Redis counters when `NLP_REDIS_URL` is reachable, a bounded in-memory fallback otherwise, and `429` with `Retry-After` when exceeded
- Demo UIs built on one shared shell (`ui/common.py`) and servable together from one process
- Fast mocked API tests, slow real-model integration tests, and an image build in CI
- Architecture, operations, evaluation, model-card, and ADR documentation

Important limitations:

- The bundled demo key is for local review only.
- First real-model requests can be slow because models are downloaded and loaded lazily.
- The tracked eval set is a smoke check, not a production quality benchmark.
- This is not intended for high-stakes or regulated automated decisions.

## Quick Start

### Docker Compose

The canonical local run is the API plus Redis:

```bash
docker compose up api redis
```

Then open:

- Swagger UI: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Readiness: http://localhost:8000/ready
- Metrics: http://localhost:8000/metrics

The Compose service uses `.env.example`, which contains the local demo API key used in the examples below. For any shared deployment, copy `.env.example` to `.env` or configure equivalent platform secrets, then replace `NLP_API__JWT_SECRET` and `NLP_API__API_KEYS`.

### Local Python

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

export NLP_API__JWT_SECRET=replace-with-a-long-random-secret
export NLP_API__API_KEYS='{"demo-customer-intel-key":{"name":"Portfolio Demo","role":"admin","rate_limit":1000,"enabled":true}}'

python launch_ui.py api
```

By default, local API docs are available at http://localhost:8000/docs.

Export the JSON value with single quotes as shown. Running `source .env.example` in a shell strips the inner quotes, so the API starts with no valid keys; copy the file to `.env` instead (the settings loader reads it) or use Docker's `env_file`.

On a CPU-only machine, `pip install --index-url https://download.pytorch.org/whl/cpu torch` before the requirements avoids several GB of CUDA libraries.

## API Examples

### Customer Intelligence Workflow

```bash
curl -X POST http://localhost:8000/api/v1/customer-intelligence/analyze \
  -H "X-API-Key: demo-customer-intel-key" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "id": "review-001",
        "text": "The product is excellent, but delivery took too long.",
        "metadata": {"source": "reviews"}
      },
      {
        "id": "ticket-002",
        "text": "Support resolved my issue quickly and the mobile app is easy to use.",
        "metadata": {"source": "support"}
      }
    ],
    "include_summary": true
  }'
```

Returns per-item sentiment and entities, plus aggregate sentiment distribution, top entities, summary, model IDs, and processing time.

### Sentiment

```bash
curl -X POST http://localhost:8000/api/v1/sentiment \
  -H "X-API-Key: demo-customer-intel-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product exceeded my expectations.",
    "model": "twitter_roberta_multilingual"
  }'
```

### Summarization

```bash
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "X-API-Key: demo-customer-intel-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence has transformed customer support operations by helping teams triage feedback, summarize conversations, and identify recurring product issues. The strongest systems still require human review, observability, and domain-specific evaluation before production rollout.",
    "min_length": 20,
    "max_length": 60
  }'
```

### Named Entity Recognition

```bash
curl -X POST http://localhost:8000/api/v1/ner \
  -H "X-API-Key: demo-customer-intel-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple CEO Tim Cook announced new products in Cupertino.",
    "entity_types": ["PERSON", "ORG", "GPE"]
  }'
```

### Semantic Similarity

```bash
curl -X POST http://localhost:8000/api/v1/similarity \
  -H "X-API-Key: demo-customer-intel-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "The onboarding flow was easy to complete.",
    "text2": "The setup process was simple and clear.",
    "model": "all_minilm_l6"
  }'
```

### Question Answering

```bash
curl -X POST http://localhost:8000/api/v1/qa \
  -H "X-API-Key: demo-customer-intel-key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What needs human review?",
    "context": "Generated summaries can help analysts review customer feedback, but important business decisions still need human review."
  }'
```

### JWT Flow

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -H "X-API-Key: demo-customer-intel-key" | jq -r ".access_token")

curl -X POST http://localhost:8000/api/v1/sentiment \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "The new workflow is much faster."}'
```

A token counts against the rate limit of the key it was issued for.

## Endpoint Reference

| Endpoint | Method | Auth | Purpose |
| --- | --- | --- | --- |
| `/api/v1/customer-intelligence/analyze` | POST | Required | Combined sentiment, NER, aggregates, and optional summary |
| `/api/v1/sentiment` | POST | Required | Single-text sentiment |
| `/api/v1/sentiment/batch` | POST | Required | Batch sentiment |
| `/api/v1/summarize` | POST | Required | Text summarization |
| `/api/v1/ner` | POST | Required | Named entity extraction |
| `/api/v1/similarity` | POST | Required | Semantic similarity |
| `/api/v1/qa` | POST | Required | Extractive question answering |
| `/api/v1/auth/token` | POST | API key | Exchange API key for JWT |
| `/api/v1/models` | GET | No | List model registry categories |
| `/api/v1/models/{category}` | GET | No | List models in one category |
| `/api/v1/status` | GET | Optional | Request, latency, cache, and uptime summary |
| `/health` | GET | No | Liveness probe |
| `/ready` | GET | No | Readiness: config, registry, auth, cache, and Redis; `503` when a check fails |
| `/metrics` | GET | No | Prometheus exposition |

Public endpoints other than `/health`, `/ready`, `/metrics` and the docs pages still count against the anonymous per-IP rate limit.

### Errors, Limits and Rate Limiting

Every error, including validation (`422`), routing (`404`, `405`) and rate-limit (`429`) errors, uses one envelope:

```json
{"error": "validation_error", "message": "Request validation failed", "details": [...], "request_id": "a1b2c3d4"}
```

`request_id` matches the `X-Request-ID` response header.

Rate limits are counted per verified client in fixed windows (60 seconds by default). Each API key has its own `rate_limit`; a JWT shares the budget of the key it was issued for; requests without valid credentials, including unknown keys, share a per-IP anonymous limit (20 by default). Every limited response carries `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` (end of the window, Unix time) and `X-RateLimit-Backend` (`redis` or `memory`).

Text longer than a model's context window is cut to the model's token limit, for example 512 tokens for the sentiment models and 1,024 for BART summarization. Whitespace-only text is rejected.

## Configuration

Key environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `NLP_API__JWT_SECRET` | empty | JWT signing secret. While empty, JWTs are rejected and `/api/v1/auth/token` returns 503 |
| `NLP_API__API_KEYS` | `{}` | JSON map of API keys to name, role, rate limit, and enabled flag |
| `NLP_API__JWT_EXPIRATION_HOURS` | `24` | Token lifetime |
| `NLP_API__DEFAULT_RATE_LIMIT` | `100` | Requests per window for keys without their own `rate_limit` |
| `NLP_API__ANONYMOUS_RATE_LIMIT` | `20` | Requests per window per IP without valid credentials |
| `NLP_API__RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window |
| `NLP_API__CORS_ORIGINS` | localhost:3000, localhost:8000 | Allowed browser origins |
| `NLP_REDIS_URL` | unset | Redis backend for shared rate limiting |
| `NLP_MAX_CACHED_MODELS` | `3` | LRU model-cache size (at least 1) |
| `NLP_CACHE_DIR` | unset | Model download directory; unset uses the Hugging Face default (`HF_HOME`) |
| `NLP_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `NLP_LOG_LEVEL` | `INFO` | Logging level |
| `NLP_JSON_LOGS` | `false` | Emit JSON logs when `true` |
| `NLP_OTEL_ENABLED` | `false` | OpenTelemetry FastAPI instrumentation (needs `opentelemetry-instrumentation-fastapi` installed) |
| `NLP_SERVER_NAME` | `127.0.0.1` | Bind address for the demo UIs; containers use `0.0.0.0` |
| `NLP_SERVER_PORT` | `7860` | Port of the combined demo UI app |
| `NLP_UI_THEME` | `soft` | Gradio theme: `soft`, `default`, `glass`, `monochrome`, `base` |
| `NLP_SHARE` | `false` | Create public Gradio share links |

Model defaults live in `utils/inference.py` and model definitions live in `config/models.yaml`.

Default API models:

- Sentiment: `twitter_roberta_multilingual`
- NER: `spacy_sm`
- Similarity: `all_minilm_l6`
- Summarization: `bart_large_cnn`
- QA: `distilbert_squad`

## Architecture

The API routers do not load models directly. They call `utils/inference.py`, which performs model lookup, input clipping, output normalization, timing, and access to the shared cache in `utils/model_cache.py`.

The cache supports:

- Hugging Face transformers pipelines
- sentence-transformers embedding models
- spaCy language pipelines
- GLiNER zero-shot NER (used by the demo UIs)

Operational instrumentation includes:

- Request logging with request ID, auth method, user role, status code, latency, and error type
- Inference logs with task, model key, model ID, status, and latency
- Prometheus counters and histograms for HTTP requests, inference, cache, auth attempts, errors, and rate-limit rejections
- `/api/v1/status` summaries from real in-process request and cache state

For more detail, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Testing And Evaluation

CI runs:

- Black, isort, and flake8
- Fast tests on Python 3.11, 3.12, and 3.13
- Slow real-model integration tests on Python 3.11
- Full test suite with coverage on Python 3.11
- A Docker image build that starts the API and Redis, waits for `/ready`, smoke-tests NER and rate limiting, and starts the combined UI app

Useful local commands:

```bash
pytest -m "not slow"
pytest -m slow
python scripts/run_eval.py --api-url http://localhost:8000 --api-key <key>
python scripts/benchmark_api.py --api-url http://localhost:8000 --api-key <key>
```

The tracked eval data is in `data/customer_intelligence_eval.json`. It is intentionally small and should be treated as a smoke/regression set, not as evidence of production model quality. See [EVAL_REPORT.md](EVAL_REPORT.md).

## Demo UIs

Ten Gradio UIs cover sentiment, similarity, NER, summarization, evaluation metrics, QA, generation, zero-shot classification, translation and vision-language. They are built on one shell (`ui/common.py`): model menus come from `config/models.yaml`, models load through the same cache as the API, and long inputs are cut with a visible warning. See [ADR 0004](docs/adr/0004-shared-ui-shell.md).

All UIs in one app (one process, one shared model cache):

```bash
python launch_ui.py all                 # http://127.0.0.1:7860
python launch_ui.py all --host 0.0.0.0  # reachable from other machines
```

One UI at a time:

```bash
python launch_ui.py --list
python launch_ui.py sentiment           # or similarity, ner, summarization, performance,
                                        # qa, generation, zero_shot, translation, vision
python -m ui.qa_system --port 9000      # any UI module also runs directly
```

Docker Compose:

```bash
docker compose --profile ui up ui-launcher            # all UIs on :7860
docker compose --profile individual up sentiment qa   # one container per UI
```

The `ui` and `individual` profiles both use host port 7860 for sentiment, so run one or the other.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): system design and tradeoffs
- [OPERATIONS.md](OPERATIONS.md): local, Docker, cloud, monitoring, and runbooks
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md): image, Compose and Kubernetes notes
- [EVAL_REPORT.md](EVAL_REPORT.md): current eval set, CI coverage, limitations, next steps
- [MODEL_CARD.md](MODEL_CARD.md): intended use, out-of-scope use, default models, risks
- [CHANGELOG.md](CHANGELOG.md): release history
- [ADR 0001](docs/adr/0001-shared-inference-cache.md): shared inference cache
- [ADR 0002](docs/adr/0002-redis-rate-limiting.md): Redis rate limiting
- [ADR 0003](docs/adr/0003-api-based-evaluation.md): API-based evaluation
- [ADR 0004](docs/adr/0004-shared-ui-shell.md): shared UI shell

## Portfolio Read

This repo is meant to show AI engineering judgment, not only model demos:

- coherent product framing
- API-first runtime
- real auth/rate-limit/metrics/logging paths
- model-cache and inference abstractions
- tests that separate fast mocked behavior from slow model downloads
- explicit limitations and risk boundaries

## License

MIT License. See `LICENSE`.
