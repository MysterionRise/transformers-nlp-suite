"""
Architecture and build checks that apply to every demo UI.

Each UI must build without Gradio deprecation warnings, without loading any model
while the page is built, and without bypassing the shared shell (own model
caches, direct pipeline() calls, root-logger configuration, hard-coded binds).
"""

import importlib
import re
import warnings
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import pytest

from ui.catalog import UIS

UI_DIR = Path(__file__).resolve().parents[3] / "ui"
FORBIDDEN = {
    "own pipeline() calls": re.compile(r"(?<![\w.])pipeline\("),
    "transformers import": re.compile(r"^\s*(from|import) transformers\b", re.M),
    "root logger configuration": re.compile(r"logging\.basicConfig"),
    "hard-coded bind address": re.compile(r"server_name\s*=\s*[\"']"),
    "theme passed to Blocks": re.compile(r"gr\.Blocks\([^)]*theme="),
}


def _source(entry):
    return (UI_DIR / f"{entry.module}.py").read_text()


@pytest.fixture(params=sorted(UIS), ids=sorted(UIS))
def entry(request):
    return UIS[request.param]


def test_catalog_matches_files():
    for ui in UIS.values():
        assert (UI_DIR / f"{ui.module}.py").exists(), ui.file
    assert len({ui.port for ui in UIS.values()}) == len(UIS)


def test_source_uses_shell_only(entry):
    source = _source(entry)
    problems = [name for name, pattern in FORBIDDEN.items() if pattern.search(source)]
    assert not problems, f"{entry.module}: {problems}"
    assert f'UI_KEY = "{entry.key}"' in source
    assert "import common as shell" in source, f"{entry.module} must build on ui/common.py"


def test_builds_without_warnings_or_model_loads(entry):
    def no_loading(*args, **kwargs):
        raise AssertionError(f"{entry.module} loaded a model while building the page: {args}")

    with (
        patch("ui.common.load_registry_model", side_effect=no_loading),
        patch("ui.common.load_by_id", side_effect=no_loading),
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings("error", category=UserWarning, module=r"gradio(\.|$)")
        warnings.filterwarnings("error", category=DeprecationWarning, module=r"gradio(\.|$)")
        module = importlib.import_module(entry.import_path)
        demo = module.create_ui()
    assert isinstance(demo, gr.Blocks)
    assert demo.title.startswith(entry.title)
