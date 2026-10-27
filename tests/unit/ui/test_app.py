"""The combined app (ui/app.py) renders every UI exactly once and keeps events wired."""

import warnings
from collections import Counter
from unittest.mock import patch

import gradio as gr
import pytest

from ui import app
from ui.catalog import UIS


@pytest.fixture(scope="module")
def config():
    def no_loading(*args, **kwargs):
        raise AssertionError(f"model loaded while building the app: {args}")

    with patch("ui.common.load_registry_model", side_effect=no_loading), warnings.catch_warnings():
        warnings.filterwarnings("error", category=UserWarning, module=r"gradio(\.|$)")
        built = app.create_app()
    assert isinstance(built, gr.Blocks)
    return built.get_config_file()


def _layout_ids(node, out):
    out.append(node["id"])
    for child in node.get("children", []):
        _layout_ids(child, out)
    return out


def test_no_component_appears_twice_in_the_layout(config):
    counts = Counter(_layout_ids(config["layout"], []))
    duplicated = {component_id: n for component_id, n in counts.items() if n > 1}
    assert not duplicated, f"components rendered more than once: {duplicated}"


def test_one_tab_per_ui_with_short_labels(config):
    labels = [c["props"].get("label") for c in config["components"] if c["type"] == "tabitem"]
    for entry in UIS.values():
        assert labels.count(f"{entry.emoji} {entry.tab_label}") == 1


def test_every_event_references_rendered_components(config):
    rendered = set(_layout_ids(config["layout"], []))
    for dependency in config["dependencies"]:
        for component_id in list(dependency["inputs"]) + list(dependency["outputs"]):
            assert component_id in rendered, f"{dependency.get('api_name')} uses unrendered component {component_id}"


def test_unknown_ui_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown UI keys"):
        app.create_app(["sentiment", "nope"])
