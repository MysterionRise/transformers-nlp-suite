# Evaluation Report

## Purpose

The evaluation path proves that the public API wiring works and records known model-risk boundaries. It is not a benchmark-grade quality claim.

## What CI Covers

The GitHub Actions workflow currently runs:

- Black, isort, and flake8
- Fast tests on Python 3.11, 3.12, and 3.13
- Slow real-model integration tests on Python 3.11
- Full test suite with coverage on Python 3.11
- A Docker image build with API, rate-limit and UI smoke checks

Fast tests mock inference and assert exact auth, validation, response, metrics/status, and aggregation behavior. Slow tests cover selected real model paths for sentiment, NER, summarization, similarity, and QA.

## API Smoke Evaluation

The tracked customer-intelligence eval set is `data/customer_intelligence_eval.json`.

It currently covers:

- positive product feedback with a delivery concern
- support feedback with expected organization/location entities
- negative service feedback

Run against a local or deployed API:

```bash
export NLP_API_KEY=<key from .env.example or deployment secrets>
python scripts/run_eval.py --api-url http://localhost:8000
```

The script exercises:

- API key auth
- `/api/v1/customer-intelligence/analyze`
- per-item sentiment sanity checks
- required entity presence checks
- aggregate sentiment distribution
- top entities
- workflow processing time

## Benchmark Script

```bash
export NLP_API_KEY=<key>
python scripts/benchmark_api.py --api-url http://localhost:8000 --iterations 3
```

The benchmark records cold/warm API latency for the similarity endpoint and captures cache status before and after the run.

## Current Limitations

- The eval set is intentionally small and should not be used to claim production model quality.
- Sentiment labels are coarse and may hide mixed customer feedback.
- spaCy small NER is fast but can miss domain-specific products, teams, or customer-specific entities.
- Summaries can omit important details or overstate weak signals.
- QA is extractive and only answers from provided context.
- Model behavior can shift when model versions or tokenizer dependencies change.

## Recommended Next Evaluation Work

- Add at least 50 labeled customer feedback examples across support, app reviews, product feedback, and social/market notes.
- Add sensitive-data logging checks.
- Add prompt-injection-style summary inputs and expected behavior assertions.
- Track pass rate, p95 latency, cache hit rate, and error rate over time.
- Store eval outputs as artifacts in CI for review.
- Add per-model comparison reports for default and alternative registry keys.
