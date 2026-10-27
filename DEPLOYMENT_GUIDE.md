# Deployment Guide

## Canonical Local Deployment

```bash
docker compose up api redis
```

This starts:

- `api`: FastAPI service on port `8000`
- `redis`: Redis rate-limit backend on port `6379`

The API service reads `.env.example` so a fresh clone can run immediately for local portfolio review.

## Build Image

```bash
docker build -t transformers-nlp-suite:latest .
```

The Dockerfile defaults to:

```bash
python launch_ui.py api --host 0.0.0.0
```

It exposes port `8000` for the API and `7860` for the combined demo UI app. Dependencies live in a virtualenv at `/opt/venv`, torch is the CPU build, the spaCy `en_core_web_sm` model is baked in, and the container runs as the unprivileged `nlpuser`. CI builds this image and smoke-tests it on every push.

## Run A Single API Container

For local testing without Compose:

```bash
docker run --rm \
  --name transformers-nlp-suite-api \
  -p 8000:8000 \
  --env-file .env.example \
  -e NLP_REDIS_URL= \
  -v huggingface_cache:/home/nlpuser/.cache/huggingface \
  transformers-nlp-suite:latest
```

The explicit empty `NLP_REDIS_URL` disables Redis for this single-container example. Use Docker Compose or a real Redis URL when you want shared rate limiting.

For any shared deployment, do not use `.env.example` directly. Configure real secrets:

```bash
NLP_API__JWT_SECRET=<secret>
NLP_API__API_KEYS=<json map of keys>
NLP_REDIS_URL=<redis url>
NLP_LOG_LEVEL=INFO
NLP_JSON_LOGS=true
```

## Docker Compose Operations

```bash
docker compose up api redis
docker compose logs -f api
docker compose down
```

Optional UI services:

```bash
docker compose --profile ui up ui-launcher                     # every UI in one app on :7860
docker compose --profile individual up sentiment qa generation  # one container per UI
```

The two profiles both publish host port 7860 for sentiment, so run one or the other.

## Access Points

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json
- Health: http://localhost:8000/health
- Readiness: http://localhost:8000/ready
- Prometheus metrics: http://localhost:8000/metrics

## Cloud Deployment Shape

Recommended target architecture:

```text
HTTPS load balancer
  -> API container(s)
      -> managed Redis for rate limiting
      -> persistent or cached Hugging Face model directory
      -> metrics/logging/tracing backend
```

Minimum platform configuration:

- one container image built from this Dockerfile
- port `8000` exposed behind HTTPS
- managed Redis URL in `NLP_REDIS_URL`
- `NLP_API__JWT_SECRET` stored as a secret
- `NLP_API__API_KEYS` stored as a secret
- memory sized for the largest expected loaded models

## Kubernetes Sketch

This repo does not ship a production Kubernetes manifest. A correct API-first manifest should expose container port `8000`, mount or cache Hugging Face models, configure Redis, and use `/health` and `/ready` probes.

Minimal deployment shape:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: transformers-nlp-suite-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: transformers-nlp-suite-api
  template:
    metadata:
      labels:
        app: transformers-nlp-suite-api
    spec:
      containers:
        - name: api
          image: transformers-nlp-suite:latest
          ports:
            - containerPort: 8000
          env:
            - name: NLP_REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: transformers-nlp-suite-secrets
                  key: redis-url
            - name: NLP_API__JWT_SECRET
              valueFrom:
                secretKeyRef:
                  name: transformers-nlp-suite-secrets
                  key: jwt-secret
            - name: NLP_API__API_KEYS
              valueFrom:
                secretKeyRef:
                  name: transformers-nlp-suite-secrets
                  key: api-keys
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
```

## Verification

After deployment:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/metrics
curl http://localhost:8000/api/v1/models
```

Authenticated smoke test:

```bash
curl -X POST http://localhost:8000/api/v1/customer-intelligence/analyze \
  -H "X-API-Key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"items":[{"text":"Support fixed my issue quickly."}],"include_summary":false}'
```

## Rollback

Keep the previous container image tag available. Roll back by redeploying the previous tag and keeping Redis data compatible. Rate-limit keys are ephemeral fixed-window counters, so Redis rollback usually does not require migration.
