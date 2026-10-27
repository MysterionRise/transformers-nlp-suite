"""Tests for the summarization UI (ui/summarization_studio.py)."""

from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
import pandas as pd
import pytest

from tests.unit.ui.conftest import FakePipeline, FakeTokenizer
from ui import summarization_studio as ui

BART = "summarization/bart_large_cnn"
T5 = "summarization/t5_base"
DISTILBART = "summarization/distilbart_cnn"

ARTICLE = " ".join(f"w{i}" for i in range(200))  # 200 whitespace tokens with the fake tokenizer


def summary_of(text, **_kwargs):
    return [{"summary_text": "  A short summary of the text.  "}]


def summarizer(max_tokens: int = 1024) -> FakePipeline:
    """Fake pipeline whose token limit is `max_tokens - 2` (two special tokens)."""
    return FakePipeline(summary_of, FakeTokenizer(model_max_length=max_tokens), max_position_embeddings=max_tokens)


class TestValidation:
    def test_empty_input_is_a_friendly_error(self):
        with pytest.raises(gr.Error, match="Please enter some text to summarize"):
            ui.summarize_text("   ", ui.MODELS.default, 30, 130)

    def test_too_short_input(self):
        with pytest.raises(gr.Error, match="at least 30 words"):
            ui.summarize_text("Only a few words here.", ui.MODELS.default, 30, 130)

    @pytest.mark.parametrize("min_length, max_length", [(130, 130), (150, 100)])
    def test_min_must_be_below_max(self, min_length, max_length):
        with pytest.raises(gr.Error, match="must be smaller than the maximum"):
            ui.summarize_text(ARTICLE, ui.MODELS.default, min_length, max_length)

    def test_unknown_model(self):
        with pytest.raises(gr.Error, match="Unknown model"):
            ui.summarize_text(ARTICLE, "google/flan-t5-large", 30, 130)


class TestSummarize:
    def test_summary_and_stats(self, fake_models):
        fake = fake_models[BART] = summarizer()
        summary, stats = ui.summarize_text(ARTICLE, "BART Large CNN", 30, 130)

        assert summary == "A short summary of the text."
        call = fake.calls[0]
        assert call["inputs"] == ARTICLE  # no prefix for BART
        assert (call["min_length"], call["max_length"], call["do_sample"]) == (30, 130, False)
        assert "max_new_tokens" not in call
        assert stats["Model"] == "facebook/bart-large-cnn"
        assert stats["Input"] == "200 words"
        assert stats["Summary"] == "6 words"
        assert stats["Compression"] == "97% shorter"
        assert stats["Length limits (tokens)"] == "30 to 130"

    def test_t5_prefix_is_applied(self, fake_models):
        fake = fake_models[T5] = summarizer(512)
        ui.summarize_text(ARTICLE, "T5 Base", 30, 130)
        sent = fake.calls[0]["inputs"]
        assert sent == "summarize: " + ARTICLE
        assert sent.count("summarize:") == 1

    def test_t5_prefix_is_not_doubled_when_the_pipeline_adds_it(self, fake_models):
        # transformers sets pipeline.prefix from the checkpoint's task_specific_params and prepends it itself.
        fake = fake_models[T5] = summarizer(512)
        fake.prefix = "summarize: "
        ui.summarize_text(ARTICLE, "T5 Base", 30, 130)
        assert fake.calls[0]["inputs"] == ARTICLE

    def test_long_input_is_clipped_with_warning(self, fake_models):
        fake = fake_models[BART] = summarizer(42)  # 40 usable tokens
        with pytest.warns(UserWarning, match="Text is 200 tokens long; only the first 40 were used"):
            _, stats = ui.summarize_text(ARTICLE, "BART Large CNN", 10, 30)
        assert fake.calls[0]["inputs"] == " ".join(ARTICLE.split()[:40])
        assert stats["Input"] == "40 words (truncated to the model's limit)"

    def test_clipping_leaves_room_for_the_t5_prefix(self, fake_models):
        fake = fake_models[T5] = summarizer(42)
        with pytest.warns(UserWarning, match="only the first 39 were used"):
            ui.summarize_text(ARTICLE, "T5 Base", 10, 30)
        sent = fake.calls[0]["inputs"]
        assert sent == "summarize: " + " ".join(ARTICLE.split()[:39])
        assert len(sent.split()) + 2 <= 42  # prefix + text + special tokens fit the model

    def test_max_length_is_capped_for_short_input(self, fake_models):
        fake = fake_models[BART] = summarizer()
        text = " ".join(f"w{i}" for i in range(35))  # 35 tokens + 2 special = 37
        with patch("ui.common.info") as info:
            _, stats = ui.summarize_text(text, "BART Large CNN", 30, 130)
        assert (fake.calls[0]["min_length"], fake.calls[0]["max_length"]) == (30, 37)
        assert stats["Length limits (tokens)"] == "30 to 37"
        assert "only 37 tokens" in info.call_args.args[0]

    def test_min_length_is_lowered_when_the_cap_reaches_it(self, fake_models):
        fake = fake_models[BART] = summarizer()
        text = " ".join(f"w{i}" for i in range(35))
        with patch("ui.common.info"):
            ui.summarize_text(text, "BART Large CNN", 60, 130)
        assert (fake.calls[0]["min_length"], fake.calls[0]["max_length"]) == (18, 37)

    def test_no_cap_or_notice_for_long_enough_input(self, fake_models):
        fake = fake_models[BART] = summarizer()
        with patch("ui.common.info") as info:
            ui.summarize_text(ARTICLE, "BART Large CNN", 30, 130)
        assert fake.calls[0]["max_length"] == 130
        info.assert_not_called()

    def test_inference_failure_is_reported(self, fake_models):
        def boom(*_a, **_k):
            raise RuntimeError("index out of range in self")

        fake_models[BART] = FakePipeline(boom)
        with pytest.raises(gr.Error, match="Summarization failed"):
            ui.summarize_text(ARTICLE, "BART Large CNN", 30, 130)


class TestLengthDefaults:
    def test_come_from_the_registry(self):
        assert ui.length_defaults("BART Large CNN") == (30, 130)
        assert ui.length_defaults("Pegasus XSum") == (20, 100)

    def test_registry_t5_large_is_t5_large(self):
        assert ui.MODELS.get("T5 Large").model_id == "t5-large"
        assert ui.MODELS.get("T5 Large").config.prefix == "summarize: "


class TestCompareModels:
    def test_needs_a_model(self):
        with pytest.raises(gr.Error, match="select at least one model"):
            list(ui.compare_models(ARTICLE, [], 30, 130))

    def test_validates_lengths_before_loading(self):
        with pytest.raises(gr.Error, match="must be smaller"):
            list(ui.compare_models(ARTICLE, ["BART Large CNN"], 100, 50))

    def test_runs_models_one_after_another_and_streams_rows(self, fake_models):
        order = []

        def tracking(name):
            def fn(text, **kwargs):
                order.append(name)
                return [{"summary_text": f"Summary by {name}."}]

            return FakePipeline(fn, FakeTokenizer())

        fake_models[BART] = tracking("bart")
        fake_models[T5] = tracking("t5")
        tables = list(ui.compare_models(ARTICLE, ["BART Large CNN", "T5 Base", "BART Large CNN"], 30, 130))

        assert order == ["bart", "t5"]  # duplicates removed, sequential
        assert [len(t) for t in tables] == [1, 2]
        final = tables[-1]
        assert isinstance(final, pd.DataFrame)
        assert list(final.columns) == ui.COMPARE_COLUMNS
        assert final["Model"].tolist() == ["BART Large CNN", "T5 Base"]
        assert final["Summary"].tolist() == ["Summary by bart.", "Summary by t5."]
        assert final["Words"].tolist() == [3, 3]
        assert fake_models[T5].calls[0]["inputs"].startswith("summarize: ")

    def test_warns_when_selection_exceeds_the_cache(self, fake_models):
        for key in (BART, T5, DISTILBART):
            fake_models[key] = summarizer()
        with patch("ui.common.get_settings", return_value=SimpleNamespace(max_cached_models=2)):
            with pytest.warns(UserWarning, match="3 models selected but the model cache holds 2"):
                tables = list(ui.compare_models(ARTICLE, ["BART Large CNN", "T5 Base", "DistilBART CNN"], 30, 130))
        assert len(tables[-1]) == 3

    def test_no_cache_warning_within_capacity(self, fake_models, recwarn):
        fake_models[BART] = summarizer()
        list(ui.compare_models(ARTICLE, ["BART Large CNN"], 30, 130))
        assert not [w for w in recwarn if "evict" in str(w.message)]

    def test_a_failing_model_is_reported_in_its_row(self, fake_models):
        fake_models[BART] = summarizer()  # T5 Base is not registered, so loading it fails
        with pytest.warns(UserWarning, match="T5 Base failed"):
            tables = list(ui.compare_models(ARTICLE, ["T5 Base", "BART Large CNN"], 30, 130))
        final = tables[-1]
        assert final["Summary"].iloc[0].startswith("Error: Loading T5 Base failed")
        assert final["Summary"].iloc[1] == "A short summary of the text."


class TestPage:
    def test_use_article_switches_to_the_summarize_tab(self):
        text, tabs = ui.use_article("Some article")
        assert text == "Some article"
        assert isinstance(tabs, gr.Tabs)
        assert tabs.selected == ui.TAB_SUMMARIZE

    def test_article_buttons_update_text_and_tabs(self):
        demo = ui.create_ui()
        main_tabs = next(b for b in demo.blocks.values() if isinstance(b, gr.Tabs))
        assert main_tabs.selected == ui.TAB_SUMMARIZE
        tab_ids = [child.id for child in main_tabs.children]
        assert tab_ids == [ui.TAB_SUMMARIZE, ui.TAB_COMPARE, ui.TAB_EXAMPLES]

        article_events = [fn for fn in demo.fns.values() if main_tabs in fn.outputs]
        assert len(article_events) == len(ui.EXAMPLE_ARTICLES)
        for fn, article in zip(article_events, ui.EXAMPLE_ARTICLES.values()):
            text, tabs = fn.fn()
            assert text == article
            assert tabs.selected == ui.TAB_SUMMARIZE

    def test_sliders_are_labelled_in_tokens_with_registry_defaults(self):
        with patch("ui.common.load_registry_model", side_effect=AssertionError("eager load")) as loader:
            demo = ui.create_ui()
        loader.assert_not_called()
        sliders = [b for b in demo.blocks.values() if isinstance(b, gr.Slider)]
        assert len(sliders) == 4
        assert all("(tokens)" in s.label and "words" not in s.label for s in sliders)
        assert (sliders[0].value, sliders[1].value) == ui.length_defaults(ui.MODELS.default)

        length_events = [fn for fn in demo.fns.values() if fn.fn is ui.length_defaults]
        assert len(length_events) == 1
        assert length_events[0].outputs == sliders[:2]

    def test_comparison_defaults_are_menu_models(self):
        assert set(ui.COMPARE_DEFAULT) <= set(ui.MODELS.names)
        assert len(ui.COMPARE_DEFAULT) <= 3
