"""Tests for the translation UI (ui/translation_hub.py)."""

from unittest.mock import patch

import gradio as gr
import pandas as pd
import pytest

from config.settings import get_model_registry
from tests.unit.ui.conftest import FakePipeline, FakeTokenizer
from ui import translation_hub as ui

HELSINKI = "translation/helsinki_opus"
MBART = "translation/mbart_50"


def tagged(text, **kwargs):
    """Fake translation: prefix the target language code so tests can see which call produced it."""
    return [{"translation_text": f"[{kwargs.get('tgt_lang', 'pair')}] {text}"}]


@pytest.fixture(autouse=True)
def fresh_announcements():
    ui._announced.clear()
    yield
    ui._announced.clear()


@pytest.fixture
def by_id(fake_models):
    """Route ``ui.common.load_by_id`` (OPUS-MT pairs other than the registry's en-es) to fakes."""
    loaded = {}

    def loader(model_id, task, action=None):
        assert task == "translation"
        if model_id not in loaded:
            raise AssertionError(f"Unexpected by-id load: {model_id}")
        return loaded[model_id]

    with patch("ui.common.load_by_id", side_effect=loader):
        yield loaded


class TestMenu:
    def test_helsinki_is_the_default_and_mbart_the_opt_in(self):
        assert ui.MODELS.names == ["Helsinki OPUS-MT", "mBART-50"]
        assert ui.MODELS.default == "Helsinki OPUS-MT"

    def test_helsinki_pairs_come_from_the_registry(self):
        expected = get_model_registry().get_model("translation", "helsinki_opus").language_pairs
        assert ui.helsinki_pairs() == expected
        assert "en-fr" in ui.helsinki_pairs()


class TestTranslate:
    def test_empty_text_is_a_friendly_error(self):
        with pytest.raises(gr.Error, match="Please enter some text to translate"):
            ui.translate("  ", "Helsinki OPUS-MT", "English", "Spanish")

    def test_helsinki_registry_pair_uses_max_new_tokens(self, fake_models):
        fake = fake_models[HELSINKI] = FakePipeline(tagged)
        translation, details = ui.translate("Hello world", "Helsinki OPUS-MT", "English", "Spanish")
        assert translation == "[pair] Hello world"
        assert details == {
            "model": "Helsinki OPUS-MT",
            "model_id": "Helsinki-NLP/opus-mt-en-es",
            "source": "English",
            "target": "Spanish",
            "truncated": False,
        }
        call = fake.calls[0]
        assert call["inputs"] == "Hello world"
        assert call["max_new_tokens"] == 3 * 2 + 20
        assert "max_length" not in call and "src_lang" not in call and "tgt_lang" not in call

    def test_other_helsinki_pairs_load_by_id(self, by_id):
        fake = by_id["Helsinki-NLP/opus-mt-en-fr"] = FakePipeline(tagged)
        translation, details = ui.translate("Good night", "Helsinki OPUS-MT", "English", "French")
        assert translation == "[pair] Good night"
        assert details["model_id"] == "Helsinki-NLP/opus-mt-en-fr"
        assert len(fake.calls) == 1

    def test_unsupported_helsinki_pair_points_to_mbart(self, by_id):
        with pytest.raises(gr.Error, match="no English → Chinese model.*Choose mBART-50"):
            ui.translate("Hello", "Helsinki OPUS-MT", "English", "Chinese")

    def test_same_source_and_target(self):
        with pytest.raises(gr.Error, match="same"):
            ui.translate("Hello", "mBART-50", "German", "German")

    def test_unknown_language_and_model(self):
        with pytest.raises(gr.Error, match="source language"):
            ui.translate("Hello", "mBART-50", "Klingon", "German")
        with pytest.raises(gr.Error, match="Unknown model"):
            ui.translate("Hello", "mbart", "English", "German")

    def test_mbart_passes_language_codes_per_call_without_mutating_the_tokenizer(self, fake_models):
        fake = fake_models[MBART] = FakePipeline(tagged)
        with patch("ui.common.info") as info:
            first, details = ui.translate("Good morning", "mBART-50", "English", "German")
            second, _ = ui.translate("Buenos días", "mBART-50", "Spanish", "French")

        assert first == "[de_DE] Good morning"
        assert second == "[fr_XX] Buenos días"
        assert (fake.calls[0]["src_lang"], fake.calls[0]["tgt_lang"]) == ("en_XX", "de_DE")
        assert (fake.calls[1]["src_lang"], fake.calls[1]["tgt_lang"]) == ("es_XX", "fr_XX")
        assert all("target_lang" not in call and "max_length" not in call for call in fake.calls)
        assert not hasattr(fake.tokenizer, "src_lang")
        assert details["model_id"] == "facebook/mbart-large-50-many-to-many-mmt"
        info.assert_called_once()
        assert "2.4 GB" in info.call_args.args[0]

    def test_long_text_is_clipped_with_warning(self, fake_models):
        fake = fake_models[HELSINKI] = FakePipeline(tagged, FakeTokenizer(model_max_length=12))
        text = " ".join(f"w{i}" for i in range(40))
        with pytest.warns(UserWarning, match="only the first 10 were used"):
            _, details = ui.translate(text, "Helsinki OPUS-MT", "English", "Spanish")
        assert details["truncated"] is True
        assert fake.calls[0]["inputs"] == " ".join(f"w{i}" for i in range(10))
        assert fake.calls[0]["max_new_tokens"] == 3 * 10 + 20

    def test_new_tokens_are_capped(self, fake_models):
        fake = fake_models[HELSINKI] = FakePipeline(tagged, FakeTokenizer(model_max_length=1000))
        ui.translate(" ".join(["word"] * 400), "Helsinki OPUS-MT", "English", "Spanish")
        assert fake.calls[0]["max_new_tokens"] == ui.MAX_NEW_TOKENS

    def test_inference_failure_is_reported(self, fake_models):
        def boom(*_a, **_k):
            raise RuntimeError("index out of range in self")

        fake_models[HELSINKI] = FakePipeline(boom)
        with pytest.raises(gr.Error, match="Translation failed"):
            ui.translate("Hello", "Helsinki OPUS-MT", "English", "Spanish")


class TestBatchTranslate:
    def test_batch_uses_the_selected_mbart_model_and_languages(self, fake_models):
        fake = fake_models[MBART] = FakePipeline(tagged)
        with patch("ui.common.info"):
            df, summary = ui.batch_translate("Hello\n\nThank you\n", "mBART-50", "English", "Italian")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ui.BATCH_COLUMNS
        assert df["Translation"].tolist() == ["[it_IT] Hello", "[it_IT] Thank you"]
        assert {(call["src_lang"], call["tgt_lang"]) for call in fake.calls} == {("en_XX", "it_IT")}
        assert "2 of 2 lines" in summary and "mBART-50" in summary

    def test_batch_uses_the_selected_helsinki_pair(self, by_id):
        fake = by_id["Helsinki-NLP/opus-mt-de-en"] = FakePipeline(tagged)
        df, summary = ui.batch_translate("Guten Morgen\nDanke", "Helsinki OPUS-MT", "German", "English")
        assert df["Original"].tolist() == ["Guten Morgen", "Danke"]
        assert len(fake.calls) == 2
        assert "from German to English with Helsinki OPUS-MT" in summary

    def test_failing_line_is_reported_in_its_row(self, fake_models):
        def fn(text, **kwargs):
            if text == "bad":
                raise RuntimeError("broken line")
            return tagged(text, **kwargs)

        fake_models[HELSINKI] = FakePipeline(fn)
        df, summary = ui.batch_translate("one\nbad\nthree", "Helsinki OPUS-MT", "English", "Spanish")
        assert df["Translation"].tolist() == ["[pair] one", "error: broken line", "[pair] three"]
        assert "2 of 3 lines" in summary and "1 line(s) failed" in summary

    def test_long_lines_warn_once(self, fake_models):
        fake_models[HELSINKI] = FakePipeline(tagged, FakeTokenizer(model_max_length=5))
        with pytest.warns(UserWarning, match="2 line"):
            ui.batch_translate("a b c d e f\nshort\ng h i j k l", "Helsinki OPUS-MT", "English", "Spanish")

    def test_validation_happens_before_loading(self):
        with pytest.raises(gr.Error, match="at least one line"):
            ui.batch_translate("\n \n", "mBART-50", "English", "German")
        with pytest.raises(gr.Error, match="no English → Japanese model"):
            ui.batch_translate("Hello", "Helsinki OPUS-MT", "English", "Japanese")


class TestSamples:
    def test_samples_are_parsed_from_the_data_file(self):
        assert len(ui.SAMPLES) == 10
        text, source, target = ui.SAMPLES["Sample 1: English → Spanish"]
        assert (source, target) == ("English", "Spanish")
        assert text.startswith("Hello, how are you today?")
        assert "---" not in text and "Sample" not in text
        assert ui.SAMPLES["Sample 10: English → Chinese"][2] == "Chinese"

    def test_use_sample_fills_text_and_languages(self):
        text, source, target = ui.use_sample("Sample 7: German → English")
        assert text.startswith("Die digitale Transformation")
        assert (source, target) == ("German", "English")
        with pytest.raises(gr.Error):
            ui.use_sample(None)

    def test_parser_skips_unknown_languages_and_missing_files(self, tmp_path):
        sample_file = tmp_path / "samples.txt"
        sample_file.write_text(
            "Sample 1 (english to french):\nHello there.\n\n  ---  \n"
            "Sample 2 (Klingon to English):\nnuqneH\n---\n"
            "Not a header\nignored\n",
            encoding="utf-8",
        )
        assert ui.load_samples(sample_file) == {"Sample 1: English → French": ("Hello there.", "English", "French")}
        assert ui.load_samples(tmp_path / "missing.txt") == {}
