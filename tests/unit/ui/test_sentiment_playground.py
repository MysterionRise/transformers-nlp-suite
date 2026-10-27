"""Tests for the sentiment UI (ui/sentiment_playground.py)."""

import gradio as gr
import pandas as pd
import pytest

from tests.unit.ui.conftest import FakePipeline, FakeTokenizer
from ui import sentiment_playground as ui

DEFAULT = "sentiment_analysis/twitter_roberta_multilingual"


def three_class(_text, **_kwargs):
    return [
        {"label": "negative", "score": 0.7},
        {"label": "neutral", "score": 0.2},
        {"label": "positive", "score": 0.1},
    ]


class TestAnalyzeSentiment:
    def test_empty_input_is_a_friendly_error(self):
        with pytest.raises(gr.Error, match="Please enter some text"):
            ui.analyze_sentiment("   ", ui.MODELS.default)

    def test_result_and_chart(self, fake_models):
        fake = fake_models[DEFAULT] = FakePipeline(three_class)
        result, fig = ui.analyze_sentiment("The delivery was late.", ui.MODELS.default)
        assert result["prediction"] == "negative"
        assert result["confidence"] == "70.0%"
        assert list(result["scores"]) == ["negative", "neutral", "positive"]
        assert result["model"] == "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        assert result["truncated"] is False
        assert fake.calls[0]["top_k"] is None
        bar = fig.data[0]
        colors = dict(zip(bar.y, bar.marker.color))
        assert colors["negative"] == ui.LABEL_COLORS["negative"]  # winner is red, not green

    def test_binary_generic_labels_are_normalised(self, fake_models):
        fake_models["sentiment_analysis/bert_sst2"] = FakePipeline(
            lambda *_a, **_k: [{"label": "LABEL_1", "score": 0.9}, {"label": "LABEL_0", "score": 0.1}],
            num_labels=2,
        )
        result, _ = ui.analyze_sentiment("Great!", "BERT Base (SST-2)")
        assert result["prediction"] == "positive"
        assert set(result["scores"]) == {"positive", "negative"}

    def test_long_text_is_clipped_with_warning(self, fake_models):
        fake = fake_models[DEFAULT] = FakePipeline(three_class, FakeTokenizer(model_max_length=12))
        text = " ".join(f"w{i}" for i in range(50))
        with pytest.warns(UserWarning, match="only the first 10 were used"):
            result, _ = ui.analyze_sentiment(text, ui.MODELS.default)
        assert result["truncated"] is True
        assert fake.calls[0]["inputs"] == " ".join(f"w{i}" for i in range(10))

    def test_inference_failure_is_reported(self, fake_models):
        def boom(*_a, **_k):
            raise RuntimeError("index out of range in self")

        fake_models[DEFAULT] = FakePipeline(boom)
        with pytest.raises(gr.Error, match="Sentiment analysis failed"):
            ui.analyze_sentiment("text", ui.MODELS.default)

    def test_unknown_model(self):
        with pytest.raises(gr.Error, match="Unknown model"):
            ui.analyze_sentiment("text", "Not A Model")


class TestAnalyzeBatch:
    def test_rows_summary_and_isolated_failures(self, fake_models):
        def fn(text, **_kwargs):
            if text == "bad":
                raise RuntimeError("broken line")
            return three_class(text)

        fake_models[DEFAULT] = FakePipeline(fn)
        df, summary = ui.analyze_batch("first\nbad\n\nthird", ui.MODELS.default)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ui.BATCH_COLUMNS
        assert df["Sentiment"].tolist() == ["negative", "error", "negative"]
        assert "3 texts analyzed" in summary and "**negative**: 2" in summary

    def test_long_lines_warn_once(self, fake_models):
        fake_models[DEFAULT] = FakePipeline(three_class, FakeTokenizer(model_max_length=4))
        with pytest.warns(UserWarning, match="2 line"):
            ui.analyze_batch("a b c d e f\nshort\ng h i j k l", ui.MODELS.default)

    def test_empty_batch(self):
        with pytest.raises(gr.Error):
            ui.analyze_batch("\n \n", ui.MODELS.default)


def test_examples_reference_menu_models():
    for _, model in ui.EXAMPLES:
        assert model in ui.MODELS
