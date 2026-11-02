"""Tests for the zero-shot classification UI (ui/zero_shot_classifier.py)."""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import pytest

from tests.unit.ui.conftest import FakePipeline, FakeTokenizer
from ui import zero_shot_classifier as ui

BART = "zero_shot/bart_large_mnli"
DEBERTA = "zero_shot/deberta_mnli"
SINGLE_SCORES = [0.7, 0.2, 0.1]
MULTI_SCORES = [0.9, 0.6, 0.1]


def nli(text, candidate_labels, multi_label=False, **_kwargs):
    """Scores by label position, like a real pipeline: labels sorted by score, highest first."""
    scores = (MULTI_SCORES if multi_label else SINGLE_SCORES)[: len(candidate_labels)]
    return {"sequence": text, "labels": list(candidate_labels), "scores": scores}


def wiring(demo, fn):
    return next(dep for dep in demo.fns.values() if dep.fn is fn)


class TestParseLabels:
    def test_strips_and_dedupes_ignoring_case(self):
        with pytest.warns(UserWarning, match="2 duplicate label"):
            labels = ui.parse_labels(" sports ,  Politics,, SPORTS ,tech\n politics ")
        assert labels == ["sports", "Politics", "tech"]

    def test_empty(self):
        with pytest.raises(gr.Error, match="Please enter candidate labels"):
            ui.parse_labels(" , ,")

    def test_single_label_mode_needs_two_labels(self):
        with pytest.raises(gr.Error, match="at least 2 different labels"):
            ui.parse_labels("urgent")
        with pytest.warns(UserWarning, match="1 duplicate"), pytest.raises(gr.Error, match="at least 2 different"):
            ui.parse_labels("urgent, Urgent")
        assert ui.parse_labels("urgent", multi_label=True) == ["urgent"]

    def test_at_most_twenty_labels(self):
        assert len(ui.parse_labels(", ".join(f"label {i}" for i in range(20)))) == 20
        with pytest.raises(gr.Error, match="at most 20 labels"):
            ui.parse_labels(", ".join(f"label {i}" for i in range(21)))

    def test_label_length(self):
        with pytest.raises(gr.Error, match="at most 100 characters"):
            ui.parse_labels("short, " + "x" * 101)

    def test_presets_are_valid_label_sets(self):
        assert len(ui.INDUSTRY_PRESETS) == 12
        for preset in [*ui.LABEL_PRESETS.values(), *ui.INDUSTRY_PRESETS.values()]:
            assert 2 <= len(ui.parse_labels(preset)) <= ui.MAX_LABELS


class TestClassifyText:
    def test_empty_text_raises_before_loading(self, fake_models):
        with pytest.raises(gr.Error, match="Please enter some text"):
            ui.classify_text("  ", "a, b", ui.MODELS.default)

    def test_bad_labels_raise_before_loading(self, fake_models):
        with pytest.raises(gr.Error, match="Please enter candidate labels"):
            ui.classify_text("Great product", "", ui.MODELS.default)

    def test_single_label_result(self, fake_models):
        fake = fake_models[BART] = FakePipeline(nli)
        outputs = ui.classify_text("Great product!", "positive, negative, neutral", ui.MODELS.default)

        assert len(outputs) == len(wiring(ui.create_ui(), ui.classify_text).outputs) == 2
        result, fig = outputs
        assert result["mode"] == "single-label"
        assert result["top_label"] == "positive" and result["score"] == "70.0%"
        assert result["how_to_read"] == ui.SINGLE_LABEL_NOTE and "add up to 100%" in result["how_to_read"]
        assert "labels_that_apply" not in result
        assert result["scores"] == {"positive": 0.7, "negative": 0.2, "neutral": 0.1}
        assert result["model"] == "facebook/bart-large-mnli" and result["truncated"] is False
        call = fake.calls[0]
        assert call["candidate_labels"] == ["positive", "negative", "neutral"]
        assert call["multi_label"] is False and "multi_class" not in call
        assert isinstance(fig, go.Figure)
        assert "adds up to 100%" in fig.layout.xaxis.title.text
        assert not fig.layout.shapes

    def test_multi_label_result_is_explained_differently(self, fake_models):
        fake = fake_models[BART] = FakePipeline(nli)
        result, fig = ui.classify_text(
            "Slow shipping, great quality", "shipping, quality, price", "BART Large MNLI", True
        )
        call = fake.calls[0]
        assert call["multi_label"] is True and "multi_class" not in call
        assert result["mode"] == "multi-label"
        assert result["how_to_read"] == ui.MULTI_LABEL_NOTE and "do not add up" in result["how_to_read"]
        assert result["labels_that_apply"] == ["shipping", "quality"]
        assert "independent" in fig.layout.xaxis.title.text
        assert fig.layout.shapes[0].x0 == ui.APPLIES_THRESHOLD
        bar = fig.data[0]
        colors = dict(zip(bar.y, bar.marker.color))
        assert colors["shipping"] == colors["quality"] == ui.COLOR_TOP and colors["price"] == ui.COLOR_OTHER

    def test_single_label_in_multi_label_mode(self, fake_models):
        fake_models[BART] = FakePipeline(nli)
        result, _ = ui.classify_text("Ship it now", "urgent", ui.MODELS.default, True)
        assert result["labels_that_apply"] == ["urgent"]

    def test_deberta_model(self, fake_models):
        fake_models[DEBERTA] = FakePipeline(nli)
        result, _ = ui.classify_text("Text", "a, b", "DeBERTa v3 NLI")
        assert result["model"] == "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    def test_long_text_is_clipped_leaving_room_for_the_hypothesis(self, fake_models):
        # token_limit = min(40, 514) - 2 special tokens = 38; minus the 32-token hypothesis reserve = 6.
        fake = fake_models[BART] = FakePipeline(nli, FakeTokenizer(model_max_length=40))
        text = " ".join(f"w{i}" for i in range(20))
        with pytest.warns(UserWarning, match="only the first 6 were used"):
            result, _ = ui.classify_text(text, "a, b", ui.MODELS.default)
        assert fake.calls[0]["inputs"] == "w0 w1 w2 w3 w4 w5"
        assert result["truncated"] is True

    def test_inference_failure_is_reported(self, fake_models):
        def boom(*_a, **_k):
            raise RuntimeError("index out of range in self")

        fake_models[BART] = FakePipeline(boom)
        with pytest.raises(gr.Error, match="Zero-shot classification failed"):
            ui.classify_text("Text", "a, b", ui.MODELS.default)


class TestBatchClassify:
    def test_rows_summary_and_isolated_failures(self, fake_models):
        def fn(text, **kwargs):
            if text == "bad":
                raise RuntimeError("broken line")
            return nli(text, **kwargs)

        fake = fake_models[BART] = FakePipeline(fn)
        df, summary = ui.batch_classify("first\nbad\n\nthird", "sports, politics", ui.MODELS.default, True)
        assert isinstance(df, pd.DataFrame) and list(df.columns) == ui.BATCH_COLUMNS
        assert df["Top label"].tolist() == ["sports", "error", "sports"]
        assert df["All scores"].tolist()[0] == "sports 90% · politics 60%"
        assert "3 texts classified" in summary and "multi-label" in summary and "**sports**: 2" in summary
        assert all(call["multi_label"] is True and "multi_class" not in call for call in fake.calls)

    def test_long_lines_warn_once(self, fake_models):
        fake_models[BART] = FakePipeline(nli, FakeTokenizer(model_max_length=40))
        with pytest.warns(UserWarning, match="2 line"):
            ui.batch_classify(" ".join("x" * 10) + "\nshort\n" + " ".join("y" * 10), "a, b", ui.MODELS.default)

    def test_validation(self, fake_models):
        with pytest.raises(gr.Error, match="at least one line"):
            ui.batch_classify("\n \n", "a, b", ui.MODELS.default)
        with pytest.raises(gr.Error, match="at least 2 different labels"):
            ui.batch_classify("text", "only", ui.MODELS.default)


def test_examples_reference_menu_models():
    for _, labels, model, multi_label in ui.EXAMPLES:
        assert model in ui.MODELS
        ui.parse_labels(labels, multi_label)


def test_about_explains_both_score_modes():
    assert "Single-label" in ui.ABOUT and "Multi-label" in ui.ABOUT
    assert "add up" in ui.ABOUT and "independently" in ui.ABOUT


def test_preset_buttons_fill_labels_without_api_endpoints():
    presets = [dep for dep in ui.create_ui().fns.values() if dep.fn.__name__ == "<lambda>"]
    assert len(presets) == len(ui.LABEL_PRESETS) + len(ui.INDUSTRY_PRESETS)
    assert {dep.fn() for dep in presets} == {*ui.LABEL_PRESETS.values(), *ui.INDUSTRY_PRESETS.values()}
    assert {dep.api_visibility for dep in presets} == {"private"}
