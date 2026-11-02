# ADR 0004: Shared UI Shell

## Status

Accepted (2026-10-27)

## Context

The ten Gradio demos were written as independent scripts. Each kept its own hand-written model list (drifting from `config/models.yaml`), its own unbounded model dictionary, its own `logging.basicConfig`, a hard-coded `0.0.0.0` bind, and its own error handling that returned `{"error": ...}` dicts or strings into output components. ADR 0001 expected the UIs to "migrate gradually" to the shared cache; none had.

A QA pass found the cost of that: features that could not work (mBART translation called with the wrong arguments, a DeBERTa base checkpoint used as an NLI model, a NER table that needed an uninstalled package, a dashboard that crashed when launched as documented), silent character-based truncation, raw tracebacks in toasts, unescaped user text in HTML, multi-GB downloads with no warning, and a Docker "launcher" service that started no UI.

## Decision

Every UI is built on `ui/common.py`:

- **Models** come from the registry through `ModelMenu`, and load through `utils.model_cache` (`get_registry_model`, which dispatches on the entry's `framework`). UIs do not import `transformers` or call `pipeline()` themselves.
- **Errors** are raised as `gr.Error` via `fail` / `fail_from`; non-blocking notes use `gr.Warning`.
- **Input limits** are enforced with `clip_for_model` / `truncate_to_tokens`, which derive the real token limit from the model config and tell the user when text was cut.
- **HTML** output escapes user text with `esc` and uses theme-aware CSS classes.
- **Launch** settings (host, port, share link, theme) come from `Settings` through `launch`, with `--host` / `--port` overrides; Gradio 6 receives the theme in `launch()`.

`ui/catalog.py` is the single list of UI titles and ports (used by the launcher and the combined app). `ui/app.py` renders every UI as a tab of one Gradio app so one process, one port and one model cache serve all tools; Docker's `ui-launcher` service runs it.

`tests/unit/ui/test_ui_build.py` enforces the decision: every UI must build without Gradio deprecation warnings, must not load a model while building its page, and must not contain its own pipelines, root-logger configuration, hard-coded binds or `theme=` on `gr.Blocks`.

## Consequences

- UIs and the API share model weights and the `NLP_MAX_CACHED_MODELS` memory bound; the combined app uses far less memory than ten processes.
- Adding a model to `config/models.yaml` makes it available in the matching UI menus without UI code changes.
- UI behaviour is consistent: one error channel, one truncation policy, one layout (header, content, about section, footer).
- The shell is a dependency of every UI, so changes to it need the full UI test suite (`pytest tests/unit/ui`).
- The cache is still process-local; running UIs in separate containers (`--profile individual`) gives each its own cache.
