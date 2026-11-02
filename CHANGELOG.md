# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Shared UI shell (`ui/common.py`), UI catalog (`ui/catalog.py`) and a combined app (`ui/app.py`, `python launch_ui.py all`) that serves every demo UI from one process and one model cache. See ADR 0004.
- `--host` / `--port` options for every UI entry point and for `launch_ui.py <ui>`.
- `NLP_API__ANONYMOUS_RATE_LIMIT` setting for clients without valid credentials.
- `ModelCache.get_registry_model()` / `load_registry_model()` dispatching on the registry `framework` (transformers, sentence-transformers, spaCy, GLiNER); optional pinned `revision` per registry entry.
- CI `docker` job: builds the image, waits for `/ready`, smoke-tests NER and rate limiting, and starts the combined UI app.
- Regression tests for rate limiting, auth edge cases, error envelopes, readiness, OpenAPI security, input limits, the UI shell, every UI, and the UI architecture rules (about 200 new tests).
- A "Models and cache" tab in the evaluation UI showing the registry and the shared cache.

### Changed

- Project name is **Transformers NLP Suite** everywhere (API title, settings, launcher, docs); customer intelligence remains the use case and the `/api/v1/customer-intelligence/analyze` endpoint is unchanged.
- Rate limiting counts requests per verified identity in fixed windows: a valid key has its own bucket, a JWT shares its key's bucket, everything else shares a per-IP anonymous bucket. `X-RateLimit-Reset` is now the end of the window. The `rate_limit_hits_total` label is now `client_type`.
- `/ready` returns `503` when a check fails.
- `404`, `405` and `422` responses use the documented `{error, message, details, request_id}` envelope.
- Inference clips input to each model's real token limit; sentiment responses include every class score.
- Model downloads use the Hugging Face cache by default; `NLP_CACHE_DIR` is only applied when set.
- Demo UIs use registry models, the shared cache, `gr.Error`/`gr.Warning`, explicit truncation warnings and Settings-driven launch (default bind `127.0.0.1`; containers set `NLP_SERVER_NAME=0.0.0.0`).
- Docker image installs into `/opt/venv` with CPU-only torch and runs as an unprivileged user; Compose's `ui-launcher` runs the combined UI app on port 7860.
- Dependency floors raised to the versions from the open Dependabot PRs; `gradio` capped below 7; CI actions bumped (checkout v6, setup-python v6, cache v5, upload-artifact v6, codecov v5). CI installs CPU-only torch.
- Dependabot uses a patterns-only group and `increase-if-necessary`.

### Fixed

- Over-limit requests returned HTTP 500 with a dropped connection instead of 429.
- Made-up `X-API-Key` values each received a fresh 100/min bucket, bypassing the anonymous limit; JWT users received the anonymous limit instead of their key's.
- An unreachable Redis added a blocking reconnect to every request; it now uses short timeouts and a 30-second back-off.
- `auth_attempts_total` counted each keyed request twice.
- Swagger UI's Authorize button never sent credentials because operations referenced security schemes that did not exist.
- Inputs within the documented size limits crashed inference ("index out of range in self").
- Binary SST-2 sentiment models reported positive text as neutral.
- Concurrent requests for an unloaded model loaded it once per request.
- The Docker image could not import its dependencies (packages installed under `/root` and run as another user).
- `docker compose --profile ui up ui-launcher` started the API instead of any UI.
- The "DeBERTa MNLI" zero-shot entry pointed at a base checkpoint without an NLI head.
- `NLP_MAX_CACHED_MODELS=0` made the first request spin forever.
- Demo UI fixes found by browser and real-model testing:
  - NER showed no entities (the results table needed the uninstalled `tabulate`); it now uses a data table, loads spaCy on first use instead of at startup (about 140 s), and keeps the Extract button reachable.
  - mBART translation always failed (wrong pipeline arguments), and batch translation ignored the selected model; Helsinki OPUS-MT is now the small default.
  - The evaluation UI crashed when launched as documented and downloaded NLTK data at import; BERTScore (about 1.4 GB) is now opt-in, and its export is linked to the batch results.
  - QA returned too few outputs on empty input (raw Gradio errors), silently cut contexts to 2,000 characters, had no "no answer" handling, and could not read its own sample file.
  - The generation seed was never used, and long prompts left no room for new text.
  - Similarity advertised UMAP without implementing it and crashed on fewer than two sentences.
  - Summarization labelled token sliders as words, accepted min >= max, and crashed on long articles.
  - Sentiment coloured the winning bar by score, so a winning "negative" bar was green.

### Security

- JWTs without an `exp` claim are rejected, and nothing validates while `NLP_API__JWT_SECRET` is empty (the token endpoint returns 503).
- Demo UIs escape user text in HTML output and no longer bind to all interfaces by default.

### Removed

- Unused slowapi integration and dependency.
- The 7B LLaVA registry entry, which nothing could serve.

## [1.0.0] - 2026-07-08

First versioned state of the API-first platform (commit `4b4db16`).

### Added

- FastAPI service with sentiment, batch sentiment, summarization, NER, similarity, extractive QA and the combined customer-intelligence workflow.
- API-key and JWT authentication, Redis-backed rate limiting with an in-memory fallback, Prometheus metrics, structured JSON logs and optional OpenTelemetry.
- Shared inference service and LRU model cache over transformers, sentence-transformers and spaCy.
- Docker Compose runtime, architecture/operations/deployment docs, model card, evaluation report and ADRs 0001-0003.
- Ten Gradio demo UIs and the `launch_ui.py` launcher.
- CI with lint, fast tests on Python 3.11-3.13, slow real-model tests and a full coverage run.
