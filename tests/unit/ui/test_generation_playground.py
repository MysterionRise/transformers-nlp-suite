"""Tests for the text generation UI (ui/generation_playground.py)."""

import re
from types import SimpleNamespace

import gradio as gr
import pandas as pd
import pytest
import torch

from tests.unit.ui.conftest import FakePipeline, FakeTokenizer
from ui import generation_playground as ui

GPT2 = "text_generation/gpt2"
DISTIL = "text_generation/distilgpt2"
EOS = 7  # deliberately not GPT-2's 50256, to prove the tokenizer's value is used


def gpt_tokenizer(model_max_length=1024, eos=EOS):
    tokenizer = FakeTokenizer(model_max_length=model_max_length, special_tokens=0)
    tokenizer.eos_token_id = eos
    return tokenizer


def continues(text=" and then it rained."):
    def fn(_prompt, **kwargs):
        return [{"generated_text": f"{text}{i}"} for i in range(kwargs.get("num_return_sequences", 1))]

    return fn


def random_draw(_prompt, **_kwargs):
    """Output depends on torch's global RNG, like real sampling."""
    return [{"generated_text": f" {torch.randint(0, 10**9, (1,)).item()}"}]


def gpt(fn, **kwargs):
    return FakePipeline(fn, gpt_tokenizer(**kwargs), max_position_embeddings=1024)


def wiring(demo, fn):
    return next(dep for dep in demo.fns.values() if dep.fn is fn)


class TestGenerateText:
    def test_empty_prompt_raises_before_loading(self, fake_models):
        with pytest.raises(gr.Error, match="Please enter a prompt"):
            ui.generate_text("  ", ui.MODELS.default)

    def test_samples_details_and_generation_kwargs(self, fake_models):
        fake = fake_models[GPT2] = gpt(continues())
        outputs = ui.generate_text("Once upon a <time>", "GPT-2", 40, 0.8, 0.95, 20, 2, 123)

        assert len(outputs) == len(wiring(ui.create_ui(), ui.generate_text).outputs) == 2
        html, details = outputs
        assert "<strong>Once upon a &lt;time&gt;</strong> and then it rained.0" in html
        assert "Sample 2" in html and 'class="tns-text"' in html
        assert details["samples"] == [
            "Once upon a <time> and then it rained.0",
            "Once upon a <time> and then it rained.1",
        ]
        assert details["seed"] == 123 and details["model"] == "gpt2" and details["prompt_truncated"] is False
        call = fake.calls[0]
        assert call["inputs"] == "Once upon a <time>"
        assert call["max_new_tokens"] == 40 and "max_length" not in call
        assert call["pad_token_id"] == EOS
        assert call["return_full_text"] is False and call["do_sample"] is True
        assert (call["temperature"], call["top_p"], call["top_k"], call["num_return_sequences"]) == (0.8, 0.95, 20, 2)

    def test_no_pad_token_when_tokenizer_has_no_eos(self, fake_models):
        fake = fake_models[GPT2] = gpt(continues(), eos=None)
        ui.generate_text("Hello", "GPT-2")
        assert "pad_token_id" not in fake.calls[0]

    def test_seed_input_is_wired_to_the_handler(self):
        dep = wiring(ui.create_ui(), ui.generate_text)
        labels = [getattr(component, "label", None) for component in dep.inputs]
        assert labels[-1] == "Seed" and isinstance(dep.inputs[-1], gr.Number)
        assert len(dep.inputs) == ui.generate_text.__code__.co_argcount

    def test_seed_is_passed_to_torch(self, fake_models, monkeypatch):
        fake_models[GPT2] = gpt(continues())
        seeds = []
        monkeypatch.setattr(ui.torch, "manual_seed", seeds.append)
        ui.generate_text("Hello", "GPT-2", seed=123)
        ui.generate_text("Hello", "GPT-2", seed=None)
        ui.generate_text("Hello", "GPT-2", seed=7.0)  # gr.Number may send floats
        assert seeds == [123, 7]

    def test_same_seed_same_text(self, fake_models):
        fake_models[GPT2] = gpt(random_draw)
        first = ui.generate_text("Hello", "GPT-2", seed=99)[1]["samples"]
        second = ui.generate_text("Hello", "GPT-2", seed=99)[1]["samples"]
        other = ui.generate_text("Hello", "GPT-2", seed=100)[1]["samples"]
        assert first == second != other

    def test_generation_runs_under_the_seed_lock(self, fake_models):
        def check_lock(prompt, **kwargs):
            assert ui._GENERATE_LOCK.locked()
            return continues()(prompt, **kwargs)

        fake_models[GPT2] = gpt(check_lock)
        ui.generate_text("Hello", "GPT-2")
        assert not ui._GENERATE_LOCK.locked()

    def test_prompt_is_clipped_to_leave_room_for_new_tokens(self, fake_models):
        fake = fake_models[GPT2] = FakePipeline(continues(), gpt_tokenizer(50), max_position_embeddings=50)
        prompt = " ".join(f"w{i}" for i in range(45))
        with pytest.warns(UserWarning, match="only the first 30 were used"):
            _, details = ui.generate_text(prompt, "GPT-2", max_new_tokens=20)
        assert fake.calls[0]["inputs"] == " ".join(f"w{i}" for i in range(30))
        assert details["prompt_truncated"] is True

    @pytest.mark.parametrize("seed", [-1, 2**32, "abc"])
    def test_invalid_seed(self, fake_models, seed):
        with pytest.raises(gr.Error, match="Seed must be a whole number"):
            ui.generate_text("Hello", "GPT-2", seed=seed)

    def test_invalid_setting(self, fake_models):
        with pytest.raises(gr.Error, match="New tokens must be between 1 and 200"):
            ui.generate_text("Hello", "GPT-2", max_new_tokens=5000)

    def test_inference_failure_is_reported(self, fake_models):
        def boom(*_a, **_k):
            raise RuntimeError("CUDA out of memory")

        fake_models[GPT2] = gpt(boom)
        with pytest.raises(gr.Error, match="Text generation failed"):
            ui.generate_text("Hello", "GPT-2")


class TestCompareModels:
    def test_requires_a_model(self, fake_models):
        with pytest.raises(gr.Error, match="Select at least one model"):
            ui.compare_models("Hello", [])

    def test_models_run_one_at_a_time_with_the_same_seed(self, fake_models, monkeypatch):
        fake_models[GPT2] = gpt(random_draw)
        fake_models[DISTIL] = gpt(random_draw)
        events = []
        real_load = ui.MODELS.load

        def logged_load(name):
            events.append(f"load {name}")
            model = real_load(name)
            events.append(f"loaded {name}")
            return model

        monkeypatch.setattr(ui.MODELS, "load", logged_load)
        html = ui.compare_models("Hello <there>", ["GPT-2", "DistilGPT-2"], 30, 0.7, 0.9, 5)

        assert events == ["load GPT-2", "loaded GPT-2", "load DistilGPT-2", "loaded DistilGPT-2"]
        assert len(fake_models[GPT2].calls) == len(fake_models[DISTIL].calls) == 1
        assert fake_models[GPT2].calls[0]["max_new_tokens"] == 30
        assert "GPT-2 · " in html and "DistilGPT-2 · " in html
        assert "<strong>Hello &lt;there&gt;</strong>" in html
        # Same seed before each model: the same RNG draw, so the fakes "generate" the same text.
        texts = re.findall(r"</strong>([^<]*)</div>", html)
        assert len(texts) == 2 and texts[0] == texts[1]

    def test_warns_when_selection_exceeds_cache(self, fake_models, monkeypatch):
        fake_models[GPT2] = gpt(continues())
        fake_models[DISTIL] = gpt(continues())
        monkeypatch.setattr(ui.shell, "get_settings", lambda: SimpleNamespace(max_cached_models=1))
        with pytest.warns(UserWarning, match="2 models selected but the model cache holds 1"):
            ui.compare_models("Hello", ["GPT-2", "DistilGPT-2"])

    def test_one_failing_model_does_not_hide_the_others(self, fake_models):
        def boom(*_a, **_k):
            raise RuntimeError("weights <corrupt>")

        fake_models[GPT2] = gpt(boom)
        fake_models[DISTIL] = gpt(continues())
        html = ui.compare_models("Hello", ["GPT-2", "DistilGPT-2"])
        assert "GPT-2 · failed" in html and "weights &lt;corrupt&gt;" in html
        assert "and then it rained.0" in html

    def test_all_models_failing_is_an_error(self, fake_models):
        def boom(*_a, **_k):
            raise RuntimeError("broken")

        fake_models[GPT2] = gpt(boom)
        with pytest.raises(gr.Error, match="Every selected model failed"):
            ui.compare_models("Hello", ["GPT-2"])


class TestBatchGenerate:
    def test_rows_summary_and_isolated_failures(self, fake_models):
        def fn(prompt, **kwargs):
            if prompt == "bad":
                raise RuntimeError("broken prompt")
            return continues()(prompt, **kwargs)

        fake = fake_models[GPT2] = gpt(fn)
        df, summary = ui.batch_generate("first\nbad\n\nthird", "GPT-2", 25, 0.7, 11)
        assert isinstance(df, pd.DataFrame) and list(df.columns) == ui.BATCH_COLUMNS
        assert df["Continuation"].tolist() == [" and then it rained.0", "error: broken prompt", " and then it rained.0"]
        assert "3 prompts continued with GPT-2 (seed 11" in summary and "1 failed" in summary
        assert all(call["max_new_tokens"] == 25 and call["pad_token_id"] == EOS for call in fake.calls)

    def test_prompt_count_is_capped(self, fake_models):
        fake_models[GPT2] = gpt(continues())
        prompts = "\n".join(f"prompt {i}" for i in range(ui.MAX_BATCH_PROMPTS + 2))
        with pytest.warns(UserWarning, match=f"Only the first {ui.MAX_BATCH_PROMPTS} of"):
            df, _ = ui.batch_generate(prompts, "GPT-2")
        assert len(df) == ui.MAX_BATCH_PROMPTS

    def test_empty_batch(self, fake_models):
        with pytest.raises(gr.Error, match="Please enter at least one prompt"):
            ui.batch_generate("\n \n", "GPT-2")


def test_examples_reference_menu_models():
    for _, model, _ in ui.EXAMPLES:
        assert model in ui.MODELS
