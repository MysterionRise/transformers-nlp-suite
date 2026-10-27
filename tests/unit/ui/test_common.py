"""Tests for the shared UI shell (ui/common.py)."""

from unittest.mock import MagicMock, patch

import gradio as gr
import pytest

from tests.unit.ui.conftest import FakePipeline, FakeTokenizer
from ui import common as shell


class TestModelMenu:
    def test_choices_come_from_registry_in_order(self):
        menu = shell.ModelMenu("sentiment_analysis")
        assert menu.names[0] == "Twitter RoBERTa (Multilingual)"
        assert menu.default == menu.names[0]
        assert all(choice.category == "sentiment_analysis" for choice in menu)

    def test_multiple_categories_and_default_by_key(self):
        menu = shell.ModelMenu("sentiment_analysis", "financial_sentiment", default="finbert")
        assert menu.default == "FinBERT"
        assert menu.get("FinBERT").category == "financial_sentiment"

    def test_filters(self):
        menu = shell.ModelMenu("vision_language", tasks=["image-to-text"])
        assert all(choice.config.task == "image-to-text" for choice in menu)
        menu = shell.ModelMenu("ner", frameworks=["spacy"])
        assert {choice.key for choice in menu} == {"spacy_sm", "spacy_trf"}
        menu = shell.ModelMenu("ner", keys=["gliner_multi", "spacy_sm"])
        assert [choice.key for choice in menu] == ["gliner_multi", "spacy_sm"]

    def test_get_accepts_key_and_rejects_unknown(self):
        menu = shell.ModelMenu("question_answering")
        assert menu.get("roberta_squad").name == "RoBERTa SQuAD"
        with pytest.raises(gr.Error):
            menu.get("nope")

    def test_bad_default_and_empty_menu(self):
        with pytest.raises(ValueError):
            shell.ModelMenu("question_answering", default="nope")
        with pytest.raises(ValueError):
            shell.ModelMenu("question_answering", tasks=["nope"])

    def test_load_uses_shared_cache_entry_point(self, fake_models):
        fake = FakePipeline(lambda *_a, **_k: [])
        fake_models["question_answering/distilbert_squad"] = fake
        assert shell.ModelMenu("question_answering").load("DistilBERT SQuAD") is fake

    def test_load_failure_becomes_gr_error(self):
        with patch("ui.common.load_registry_model", side_effect=OSError("hub unreachable")):
            with pytest.raises(gr.Error) as excinfo:
                shell.ModelMenu("question_answering").load("DistilBERT SQuAD")
        assert "Loading DistilBERT SQuAD failed" in str(excinfo.value)

    def test_table_markdown_links_models(self):
        table = shell.ModelMenu("question_answering").table_markdown()
        assert "https://huggingface.co/deepset/roberta-base-squad2" in table
        assert table.startswith("| Model |")


class TestInputHelpers:
    def test_require_text(self):
        assert shell.require_text("  hi  ") == "hi"
        for empty in (None, "", "   \n"):
            with pytest.raises(gr.Error):
                shell.require_text(empty)

    def test_split_items_caps_with_warning(self):
        assert shell.split_items("a\n\n b \nc") == ["a", "b", "c"]
        with pytest.warns(UserWarning, match="Only the first 2 of 3"):
            assert shell.split_items("a\nb\nc", max_items=2) == ["a", "b"]
        with pytest.raises(gr.Error):
            shell.split_items(" \n ")

    def test_esc(self):
        assert shell.esc('<img src=x onerror="alert(1)">') == "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;"
        assert shell.esc(None) == ""

    def test_fail_from_keeps_gr_error_and_wraps_others(self):
        with pytest.raises(gr.Error, match="kept"):
            shell.fail_from(gr.Error("kept"), "Thing")
        with pytest.raises(gr.Error, match="Thing failed"):
            shell.fail_from(ValueError("x" * 1000), "Thing")


class TestTokenLimits:
    def test_token_limit_uses_smaller_bound_minus_specials(self):
        pipe = FakePipeline(lambda *_: None, FakeTokenizer(model_max_length=512), max_position_embeddings=514)
        assert shell.token_limit(pipe) == 510
        pipe.tokenizer.model_max_length = int(1e30)  # HF sentinel for "unknown"
        assert shell.token_limit(pipe) == 512

    def test_clip_short_text_untouched(self):
        assert shell.clip_to_tokens("one two three", FakeTokenizer(), 5) == "one two three"

    def test_clip_long_text_on_token_boundary_with_warning(self):
        with pytest.warns(UserWarning, match="only the first 3 were used"):
            clipped = shell.clip_to_tokens("one two three four five", FakeTokenizer(), 3, label="Text")
        assert clipped == "one two three"

    def test_clip_for_model_reserves_tokens(self):
        pipe = FakePipeline(lambda *_: None, FakeTokenizer(model_max_length=6), max_position_embeddings=6)
        with pytest.warns(UserWarning):
            assert shell.clip_for_model("a b c d e f g", pipe, reserve=1) == "a b c"


class TestLaunch:
    def test_launch_uses_settings_and_catalog_port(self):
        demo = MagicMock()
        shell.launch(demo, "qa")
        kwargs = demo.launch.call_args.kwargs
        settings = shell.get_settings()
        assert kwargs["server_name"] == settings.server_name
        assert kwargs["server_port"] == 7865
        assert kwargs["share"] is settings.share
        assert kwargs["css"] == shell.CSS
        assert isinstance(kwargs["theme"], gr.themes.ThemeClass)

    def test_launch_combined_app_uses_server_port_and_overrides(self):
        demo = MagicMock()
        shell.launch(demo, server_name="0.0.0.0")
        kwargs = demo.launch.call_args.kwargs
        assert kwargs["server_port"] == shell.get_settings().server_port
        assert kwargs["server_name"] == "0.0.0.0"
