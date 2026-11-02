"""Tests for the vision-language UI (ui/vision_language_explorer.py)."""

import gradio as gr
import pandas as pd
import pytest
from PIL import Image

from tests.unit.ui.conftest import FakePipeline, FakeTokenizer
from ui import vision_language_explorer as ui

CAPTION = "vision_language/git_base"
CLIP = "vision_language/clip_vit_base"


def describe(image, **_kwargs):
    """Fake captioner that reports what it received."""
    return [{"generated_text": f" a {image.width}x{image.height} {image.mode} picture "}]


def rank_by_position(image, candidate_labels, **_kwargs):
    """Fake CLIP: later labels score higher; scores sum to 1 like a softmax."""
    total = sum(range(1, len(candidate_labels) + 1))
    scored = [{"label": label, "score": (i + 1) / total} for i, label in enumerate(candidate_labels)]
    return sorted(scored, key=lambda item: item["score"], reverse=True)


def save_png(path, color="red", size=(32, 24)):
    Image.new("RGB", size, color).save(path)
    return str(path)


def test_menus_use_registry_pipelines_with_base_defaults():
    assert ui.CAPTION_MODELS.default == "GIT Base"
    assert {choice.config.task for choice in ui.CAPTION_MODELS} == {"image-to-text"}
    assert ui.MATCH_MODELS.default == "CLIP ViT-B/32"
    assert {choice.config.task for choice in ui.MATCH_MODELS} == {"zero-shot-image-classification"}


def test_page_copy_only_promises_captioning_and_matching():
    demo = ui.create_ui()
    copy = " ".join(
        f"{getattr(block, 'value', '') or ''} {getattr(block, 'label', '') or ''}" for block in demo.blocks.values()
    ).lower()
    assert "caption" in copy and "relative" in copy
    for promise in ("vqa", "question answering", "embeddings"):
        assert promise not in copy
    assert not hasattr(ui, "load_vqa_model")


class TestCaption:
    def test_missing_image(self):
        with pytest.raises(gr.Error, match="Please upload an image"):
            ui.generate_caption(None, ui.CAPTION_MODELS.default)

    def test_caption_converts_to_rgb_and_limits_new_tokens(self, fake_models):
        fake = fake_models[CAPTION] = FakePipeline(describe)
        caption, details = ui.generate_caption(Image.new("RGBA", (64, 32)), ui.CAPTION_MODELS.default)
        assert caption == "a 64x32 RGB picture"
        assert details == {"model": "GIT Base", "model_id": "microsoft/git-base", "image_size": "64x32"}
        assert fake.calls[0]["max_new_tokens"] == ui.MAX_CAPTION_TOKENS

    def test_sample_image_needs_no_network(self, fake_models):
        fake_models[CAPTION] = FakePipeline(describe)
        image = ui.sample_image()
        assert image.mode == "RGB" and image.size == (384, 288)
        assert len(image.getcolors(maxcolors=64)) > 3  # a drawn scene, not a blank canvas
        caption, _ = ui.generate_caption(image, ui.CAPTION_MODELS.default)
        assert caption == "a 384x288 RGB picture"

    def test_captioning_failure_is_reported(self, fake_models):
        def boom(*_a, **_k):
            raise RuntimeError("CUDA out of memory")

        fake_models[CAPTION] = FakePipeline(boom)
        with pytest.raises(gr.Error, match="Image captioning failed"):
            ui.generate_caption(Image.new("RGB", (8, 8)), ui.CAPTION_MODELS.default)


class TestMatch:
    def test_scores_are_relative_and_descriptions_escaped(self, fake_models):
        fake = fake_models[CLIP] = FakePipeline(rank_by_position)
        result, html = ui.match_image(Image.new("RGB", (16, 16)), "<b>cat</b>\na dog & a ball", "CLIP ViT-B/32")

        assert fake.calls[0]["candidate_labels"] == ["<b>cat</b>", "a dog & a ball"]
        assert fake.calls[0]["hypothesis_template"] == "{}"
        assert result["best_match"] == "a dog & a ball"
        assert sum(result["relative_scores"].values()) == pytest.approx(1.0)
        assert "softmax over your 2 descriptions" in result["note"]
        assert result["model"] == "openai/clip-vit-base-patch32"
        assert "&lt;b&gt;cat&lt;/b&gt;" in html and "a dog &amp; a ball" in html
        assert "<b>cat</b>" not in html
        assert "softmax over your 2 descriptions" in html

    def test_single_comma_separated_line(self, fake_models):
        fake = fake_models[CLIP] = FakePipeline(rank_by_position)
        ui.match_image(Image.new("RGB", (16, 16)), "a cat, a dog, a bird", ui.MATCH_MODELS.default)
        assert fake.calls[0]["candidate_labels"] == ["a cat", "a dog", "a bird"]

    def test_needs_two_different_descriptions(self, fake_models):
        with pytest.raises(gr.Error, match="at least two"):
            ui.match_image(Image.new("RGB", (16, 16)), "a cat", ui.MATCH_MODELS.default)
        with pytest.raises(gr.Error, match="at least two different"):
            ui.match_image(Image.new("RGB", (16, 16)), "a cat\na cat", ui.MATCH_MODELS.default)
        with pytest.raises(gr.Error, match="Please upload an image"):
            ui.match_image(None, "a cat\na dog", ui.MATCH_MODELS.default)

    def test_long_description_is_clipped_with_warning(self, fake_models):
        fake = fake_models[CLIP] = FakePipeline(rank_by_position, FakeTokenizer(model_max_length=7))
        long_text = " ".join(f"w{i}" for i in range(12))
        with pytest.warns(UserWarning, match="Description 2 is 12 tokens long"):
            ui.match_image(Image.new("RGB", (16, 16)), f"a cat\n{long_text}", ui.MATCH_MODELS.default)
        assert fake.calls[0]["candidate_labels"][1] == "w0 w1 w2 w3 w4"

    def test_description_count_is_capped(self, fake_models):
        fake = fake_models[CLIP] = FakePipeline(rank_by_position)
        texts = "\n".join(f"thing {i}" for i in range(ui.MAX_DESCRIPTIONS + 5))
        with pytest.warns(UserWarning, match=f"Only the first {ui.MAX_DESCRIPTIONS}"):
            ui.match_image(Image.new("RGB", (16, 16)), texts, ui.MATCH_MODELS.default)
        assert len(fake.calls[0]["candidate_labels"]) == ui.MAX_DESCRIPTIONS

    def test_matching_failure_is_reported(self, fake_models):
        def boom(*_a, **_k):
            raise RuntimeError("index out of range in self")

        fake_models[CLIP] = FakePipeline(boom)
        with pytest.raises(gr.Error, match="Image-text matching failed"):
            ui.match_image(Image.new("RGB", (16, 16)), "a\nb", ui.MATCH_MODELS.default)


class TestBatchCaption:
    def test_upload_count_is_capped_with_warning(self, fake_models, tmp_path):
        fake = fake_models[CAPTION] = FakePipeline(describe)
        paths = [save_png(tmp_path / f"img{i}.png") for i in range(ui.MAX_IMAGES + 2)]
        with pytest.warns(UserWarning, match=f"Only the first {ui.MAX_IMAGES} of {ui.MAX_IMAGES + 2} images"):
            df, summary = ui.batch_caption(paths, ui.CAPTION_MODELS.default)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ui.BATCH_COLUMNS
        assert len(df) == ui.MAX_IMAGES == len(fake.calls)
        assert df["File"].tolist()[:2] == ["img0.png", "img1.png"]
        assert f"{ui.MAX_IMAGES} of {ui.MAX_IMAGES} images captioned" in summary

    def test_files_open_as_rgb_and_non_images_are_reported(self, fake_models, tmp_path):
        fake_models[CAPTION] = FakePipeline(describe)
        palette = tmp_path / "palette.png"
        Image.new("P", (10, 20)).save(palette)
        notes = tmp_path / "notes.txt"
        notes.write_text("not an image")
        with pytest.warns(UserWarning, match="1 file\\(s\\) were skipped"):
            df, summary = ui.batch_caption([str(palette), str(notes)], ui.CAPTION_MODELS.default)
        assert df.to_dict("records") == [
            {"#": 1, "File": "palette.png", "Size": "10x20", "Caption": "a 10x20 RGB picture"},
            {"#": 2, "File": "notes.txt", "Size": "", "Caption": "not a readable image"},
        ]
        assert "1 of 2 images captioned" in summary

    def test_single_path_string_and_empty_upload(self, fake_models, tmp_path):
        fake_models[CAPTION] = FakePipeline(describe)
        df, _ = ui.batch_caption(save_png(tmp_path / "one.png"), ui.CAPTION_MODELS.default)
        assert df["Caption"].tolist() == ["a 32x24 RGB picture"]
        with pytest.raises(gr.Error, match="at least one image"):
            ui.batch_caption(None, ui.CAPTION_MODELS.default)
        with pytest.raises(gr.Error, match="at least one image"):
            ui.batch_caption([], ui.CAPTION_MODELS.default)

    def test_failing_image_is_reported_in_its_row(self, fake_models, tmp_path):
        def fn(image, **kwargs):
            if image.width == 7:
                raise RuntimeError("decoder crashed")
            return describe(image, **kwargs)

        fake_models[CAPTION] = FakePipeline(fn)
        paths = [save_png(tmp_path / "ok.png"), save_png(tmp_path / "bad.png", size=(7, 7))]
        df, summary = ui.batch_caption(paths, ui.CAPTION_MODELS.default)
        assert df["Caption"].tolist() == ["a 32x24 RGB picture", "error: decoder crashed"]
        assert "1 of 2 images captioned" in summary and "1 image(s) failed" in summary
