"""
All demo UIs in one Gradio app.

One process, one port (NLP_SERVER_PORT, default 7860) and one shared model cache,
so a model loaded by one tool is reused by the others and memory stays bounded by
NLP_MAX_CACHED_MODELS. Start it with ``python launch_ui.py all`` or
``python -m ui.app``.
"""

import importlib
import sys
from typing import Iterable, Optional

try:
    from ui import common as shell
except ModuleNotFoundError:  # running as `python ui/app.py`
    import common as shell  # type: ignore[no-redef]

import gradio as gr


def create_app(keys: Optional[Iterable[str]] = None) -> gr.Blocks:
    """Build one Blocks app with a tab per UI (all UIs by default)."""
    selected = list(keys) if keys else list(shell.UIS)
    unknown = [key for key in selected if key not in shell.UIS]
    if unknown:
        raise ValueError(f"Unknown UI keys: {unknown}. Available: {list(shell.UIS)}")

    with gr.Blocks(title=shell.APP_NAME, analytics_enabled=False) as app:
        gr.Markdown(
            f"# {shell.APP_NAME}\n\nInteractive demos of the suite's NLP models. "
            "Pick a tool below; each model downloads the first time you use it."
        )
        with gr.Tabs():
            for key in selected:
                entry = shell.UIS[key]
                module = importlib.import_module(entry.import_path)
                with gr.Tab(f"{entry.emoji} {entry.title}", id=key):
                    module.create_ui().render()
    return app


if __name__ == "__main__":
    shell.launch(create_app(), argv=sys.argv[1:])
