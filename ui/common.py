"""
Shared shell for the Gradio demo UIs.

Every UI builds its page with these helpers so that all tools behave the same way:

- model menus come from ``config/models.yaml`` (no hand-written model lists),
- models load through the shared LRU cache in ``utils.model_cache``,
- problems are reported with ``gr.Error`` / ``gr.Warning`` instead of error dicts,
- users are told when input was cut to fit a model's token limit,
- user text is escaped before it is placed in HTML,
- host, port, share link and theme come from ``Settings`` (``NLP_*`` env vars).

A UI module imports this file as ``shell``::

    try:
        from ui import common as shell
    except ModuleNotFoundError:  # running as `python ui/<tool>.py`
        import common as shell

and then only talks to ``config``/``utils`` through it, so the script also works
when started directly from the ``ui/`` directory.
"""

import argparse
import html
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, NoReturn, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:  # allow `python ui/<tool>.py` from any working directory
    sys.path.insert(0, str(_ROOT))

import gradio as gr  # noqa: E402

from config.settings import ModelConfig, get_model_registry, get_settings  # noqa: E402
from ui.catalog import UIS, UIInfo  # noqa: E402
from utils.error_handler import format_error_message, get_error_suggestion  # noqa: E402
from utils.labels import normalize_sentiment_label, num_labels_of  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.text_limits import count_tokens, token_limit, truncate_to_tokens  # noqa: E402

__all__ = [
    "APP_NAME",
    "MAX_BATCH_ITEMS",
    "ModelChoice",
    "ModelMenu",
    "UIS",
    "UIInfo",
    "about",
    "clip_for_model",
    "clip_to_tokens",
    "count_tokens",
    "esc",
    "fail",
    "fail_from",
    "get_logger",
    "get_settings",
    "info",
    "launch",
    "load_by_id",
    "load_registry_model",
    "normalize_sentiment_label",
    "num_labels_of",
    "page",
    "parse_cli",
    "require_text",
    "split_items",
    "theme",
    "token_limit",
    "truncate_to_tokens",
    "warn",
]

logger = get_logger(__name__)

APP_NAME = get_settings().app_name
MAX_BATCH_ITEMS = 100
MAX_UPLOAD_SIZE = "10mb"

THEMES = {
    "soft": gr.themes.Soft,
    "default": gr.themes.Default,
    "glass": gr.themes.Glass,
    "monochrome": gr.themes.Monochrome,
    "base": gr.themes.Base,
}

# Classes the UIs use for highlighted text. Colours come from theme variables so
# highlights stay readable in light and dark mode.
CSS = """
.tns-footer { opacity: 0.75; font-size: 0.85rem; }
.tns-text { line-height: 1.9; padding: 0.75rem 1rem; border-radius: 8px;
            background: var(--block-background-fill); color: var(--body-text-color);
            border: 1px solid var(--border-color-primary); white-space: pre-wrap; }
.tns-mark { background: var(--color-accent-soft); color: inherit; padding: 0.1rem 0.25rem;
            border-radius: 4px; border-bottom: 2px solid var(--color-accent); }
.tns-label { font-size: 0.7em; font-weight: 600; margin-left: 0.25rem; opacity: 0.85;
             text-transform: uppercase; }
"""


# --------------------------------------------------------------------------- models


def load_registry_model(category: str, model_key: str) -> Any:
    """Load a registry model through the shared cache (patched in tests)."""
    from utils.model_cache import get_model_cache

    return get_model_cache().get_registry_model(category, model_key)


def load_by_id(model_id: str, task: str, action: Optional[str] = None) -> Any:
    """
    Load a Hugging Face model that is not a registry entry (for example one
    OPUS-MT language pair) through the shared cache.
    """
    from utils.model_cache import get_model_cache

    try:
        return get_model_cache().get_model_by_id(model_id, task)
    except Exception as exc:
        fail_from(exc, action or f"Loading {model_id}")


@dataclass(frozen=True)
class ModelChoice:
    """One entry of a model menu."""

    name: str
    category: str
    key: str
    config: ModelConfig

    @property
    def model_id(self) -> str:
        return self.config.model_id


class ModelMenu:
    """
    Model choices for one UI, built from one or more registry categories.

    The display name is the value Gradio sends back, so examples and dropdowns
    show readable names. Names must be unique across the selected categories.
    """

    def __init__(
        self,
        *categories: str,
        tasks: Optional[Iterable[str]] = None,
        frameworks: Optional[Iterable[str]] = None,
        keys: Optional[Iterable[str]] = None,
        exclude: Iterable[str] = (),
        default: Optional[str] = None,
    ):
        if not categories:
            raise ValueError("ModelMenu needs at least one registry category")
        registry = get_model_registry()
        tasks = set(tasks) if tasks else None
        frameworks = set(frameworks) if frameworks else None
        keys = list(keys) if keys else None
        exclude = set(exclude)

        self._choices: Dict[str, ModelChoice] = {}
        for category in categories:
            entries = registry.get_category(category)
            ordered = [k for k in keys if k in entries] if keys else list(entries)
            for key in ordered:
                config = entries[key]
                if key in exclude:
                    continue
                if tasks and config.task not in tasks:
                    continue
                if frameworks and (config.framework or "transformers") not in frameworks:
                    continue
                if config.name in self._choices:
                    raise ValueError(f"Duplicate model name {config.name!r} in categories {categories}")
                self._choices[config.name] = ModelChoice(config.name, category, key, config)

        if not self._choices:
            raise ValueError(f"No registry models match categories={categories} tasks={tasks}")

        names = list(self._choices)
        if default is None:
            self.default = names[0]
        else:
            matches = [c.name for c in self._choices.values() if default in (c.name, c.key)]
            if not matches:
                raise ValueError(f"Default model {default!r} is not in this menu: {names}")
            self.default = matches[0]

    @property
    def names(self) -> List[str]:
        return list(self._choices)

    def __contains__(self, name: str) -> bool:
        return name in self._choices

    def __iter__(self) -> Iterator[ModelChoice]:
        return iter(self._choices.values())

    def get(self, name: Optional[str]) -> ModelChoice:
        """Resolve a display name (or registry key) to its choice, or raise gr.Error."""
        if name in self._choices:
            return self._choices[name]
        for choice in self._choices.values():
            if choice.key == name:
                return choice
        fail(f"Unknown model {name!r}. Choose one of: {', '.join(self.names)}.")

    def load(self, name: Optional[str]) -> Any:
        """Load the chosen model through the shared cache, reporting failures as gr.Error."""
        choice = self.get(name)
        try:
            return load_registry_model(choice.category, choice.key)
        except gr.Error:
            raise
        except Exception as exc:
            fail_from(exc, f"Loading {choice.name}")

    def dropdown(self, label: str = "Model", value: Optional[str] = None, **kwargs) -> gr.Dropdown:
        kwargs.setdefault("info", "Downloaded on first use, then kept in the shared model cache.")
        return gr.Dropdown(choices=self.names, value=value or self.default, label=label, **kwargs)

    def radio(self, label: str = "Model", value: Optional[str] = None, **kwargs) -> gr.Radio:
        kwargs.setdefault("info", "Downloaded on first use, then kept in the shared model cache.")
        return gr.Radio(choices=self.names, value=value or self.default, label=label, **kwargs)

    def table_markdown(self) -> str:
        """Markdown table of the menu's models, generated from the registry."""
        rows = ["| Model | Source | Notes |", "| --- | --- | --- |"]
        for choice in self._choices.values():
            config = choice.config
            framework = config.framework or "transformers"
            if framework == "spacy":
                source = f"spaCy `{config.model_id}`"
            else:
                source = f"[{config.model_id}](https://huggingface.co/{config.model_id})"
            notes = config.description or ""
            rows.append(f"| {choice.name} | {source} | {notes} |")
        return "\n".join(rows)


# --------------------------------------------------------------------------- errors and input


def fail(message: str) -> NoReturn:
    """Stop the event and show `message` to the user as an error toast."""
    raise gr.Error(message)


def fail_from(exc: Exception, action: str) -> NoReturn:
    """Log `exc` with its traceback and show a short, user-facing error."""
    if isinstance(exc, gr.Error):
        raise exc
    logger.error(f"{action} failed: {exc}", exc_info=exc)
    message = format_error_message(exc)
    if len(message) > 300:
        message = message[:297] + "..."
    suggestion = get_error_suggestion(exc)
    text = f"{action} failed. {message}"
    if suggestion:
        text += f" {suggestion}"
    raise gr.Error(text) from exc


def warn(message: str) -> None:
    """Non-blocking warning toast (a UserWarning outside a Gradio event)."""
    gr.Warning(message)


def info(message: str) -> None:
    """Non-blocking information toast."""
    gr.Info(message)


def require_text(value: Optional[str], what: str = "some text") -> str:
    """Return the stripped text or stop with a friendly error if it is empty."""
    text = (value or "").strip()
    if not text:
        fail(f"Please enter {what}.")
    return text


def split_items(
    value: Optional[str],
    delimiter: str = "\n",
    max_items: int = MAX_BATCH_ITEMS,
    what: str = "one line of text",
) -> List[str]:
    """Split batch input into non-empty items, capping the count with a warning."""
    items = [item.strip() for item in (value or "").split(delimiter) if item.strip()]
    if not items:
        fail(f"Please enter at least {what}.")
    if len(items) > max_items:
        warn(f"Only the first {max_items} of {len(items)} items were processed.")
        items = items[:max_items]
    return items


def esc(value: Any) -> str:
    """HTML-escape user-provided text before putting it into gr.HTML output."""
    return html.escape("" if value is None else str(value), quote=True)


# --------------------------------------------------------------------------- token limits


def clip_to_tokens(text: str, tokenizer: Any, max_tokens: int, label: str = "Input") -> str:
    """
    Return the longest prefix of `text` that fits in `max_tokens` tokens.

    Cuts on a token boundary, so character offsets into the returned text stay
    valid, and warns the user when it cuts.
    """
    clipped, count = truncate_to_tokens(text, tokenizer, max_tokens)
    if clipped != text:
        warn(f"{label} is {count} tokens long; only the first {max_tokens} were used (model limit).")
    return clipped


def clip_for_model(text: str, model: Any, label: str = "Input", reserve: int = 0) -> str:
    """`clip_to_tokens` using the pipeline's own tokenizer and token limit."""
    return clip_to_tokens(text, getattr(model, "tokenizer", None), max(1, token_limit(model) - reserve), label)


# --------------------------------------------------------------------------- page layout


@contextmanager
def page(ui_key: str, intro: Optional[str] = None) -> Iterator[gr.Blocks]:
    """
    Standard page: header from the UI catalog, the UI's own content, then a footer.

    Usage::

        with shell.page("sentiment") as demo:
            ...components and events...
        return demo
    """
    entry = UIS[ui_key]
    with gr.Blocks(title=f"{entry.title} · {APP_NAME}", analytics_enabled=False) as demo:
        gr.Markdown(f"# {entry.emoji} {entry.title}\n\n{intro or entry.description}")
        yield demo
        settings = get_settings()
        gr.Markdown(
            f"{APP_NAME} · models download on first use and share one in-memory cache "
            f"(up to {settings.max_cached_models} at a time) · nothing you enter is stored",
            elem_classes=["tns-footer"],
        )


def about(markdown: str = "", *menus: ModelMenu, expanded: bool = False) -> None:
    """Collapsible 'About' section; model tables are generated from the registry."""
    with gr.Accordion("About this tool", open=expanded):
        parts = [markdown.strip()] if markdown.strip() else []
        for menu in menus:
            parts.append("### Models\n\n" + menu.table_markdown())
        parts.append(
            "### Limits\n\n"
            "Long inputs are cut to the selected model's token limit and you get a warning when that "
            "happens. Batch tabs process at most "
            f"{MAX_BATCH_ITEMS} items. The first run of a model downloads it (roughly 0.1 to 2.5 GB), "
            "which can take a few minutes on CPU."
        )
        gr.Markdown("\n\n".join(parts))


def theme() -> Any:
    """Theme instance selected by NLP_UI_THEME (defaults to Soft)."""
    return THEMES.get(get_settings().ui_theme.lower(), gr.themes.Soft)()


def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """`--host` / `--port` overrides accepted by every UI entry point."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args, _ = parser.parse_known_args(argv)
    return args


def launch(
    demo: gr.Blocks,
    ui_key: Optional[str] = None,
    port: Optional[int] = None,
    argv: Optional[Sequence[str]] = None,
    **overrides: Any,
) -> Any:
    """
    Launch a UI with Settings-driven host, port, share link and theme.

    Binds to NLP_SERVER_NAME (127.0.0.1 by default; containers set 0.0.0.0).
    Individual UIs use their catalog port, the combined app NLP_SERVER_PORT.
    ``--host`` / ``--port`` on the command line override both, for example
    ``python -m ui.qa_system --port 9000``.
    """
    settings = get_settings()
    cli = parse_cli(argv if argv is not None else [])
    if port is None:
        port = cli.port or (UIS[ui_key].port if ui_key else settings.server_port)
    kwargs: Dict[str, Any] = {
        "server_name": cli.host or settings.server_name,
        "server_port": port,
        "share": settings.share,
        "show_error": True,
        "theme": theme(),
        "css": CSS,
        "max_file_size": MAX_UPLOAD_SIZE,
    }
    kwargs.update(overrides)
    logger.info(f"Launching {ui_key or 'combined app'} on http://{kwargs['server_name']}:{kwargs['server_port']}")
    return demo.launch(**kwargs)
