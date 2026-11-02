"""
Vision-Language - caption images and match an image against text descriptions.

Models come from the ``vision_language`` registry category and load through the
shared model cache as pipelines: ``image-to-text`` models (GIT, ViT-GPT2) write
captions and ``zero-shot-image-classification`` models (CLIP) score how well each
of the user's descriptions fits an image.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gradio as gr
import pandas as pd
from PIL import Image, ImageDraw

try:
    from ui import common as shell
except ModuleNotFoundError:  # running as `python ui/vision_language_explorer.py`
    import common as shell  # type: ignore[no-redef]

logger = shell.get_logger(__name__)

UI_KEY = "vision"
CAPTION_MODELS = shell.ModelMenu("vision_language", tasks=["image-to-text"], default="git_base")
MATCH_MODELS = shell.ModelMenu("vision_language", tasks=["zero-shot-image-classification"], default="clip_vit_base")

MAX_IMAGES = 10
MAX_DESCRIPTIONS = 20
MAX_CAPTION_TOKENS = 40
BATCH_COLUMNS = ["#", "File", "Size", "Caption"]
RELATIVE_NOTE = (
    "Scores are a softmax over your {n} descriptions: they sum to 100% and only rank these descriptions "
    "against each other. They are not an absolute measure of how well any one description fits the image."
)
SAMPLE_DESCRIPTIONS = "a drawing of a house with a red roof\na photo of a cat\na bowl of fruit\na city at night"


# --------------------------------------------------------------------------- images


def sample_image() -> Image.Image:
    """A small scene drawn at runtime (sky, sun, grass and a house), so the demo needs no downloads."""
    image = Image.new("RGB", (384, 288), (135, 196, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 200, 384, 288), fill=(76, 153, 0))
    draw.ellipse((290, 25, 350, 85), fill=(255, 221, 51))
    draw.rectangle((120, 130, 260, 230), fill=(222, 184, 135))
    draw.polygon([(105, 135), (190, 70), (275, 135)], fill=(178, 34, 34))
    draw.rectangle((175, 180, 205, 230), fill=(101, 67, 33))
    draw.rectangle((135, 150, 165, 175), fill=(173, 216, 230))
    draw.rectangle((220, 150, 250, 175), fill=(173, 216, 230))
    return image


def _rgb(image: Optional[Image.Image]) -> Image.Image:
    if image is None:
        shell.fail("Please upload an image.")
    return image if image.mode == "RGB" else image.convert("RGB")


def _open_image(path: str) -> Optional[Image.Image]:
    """Open an uploaded file as an RGB image, or return None if it is not a readable image."""
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        logger.warning(f"Could not open {Path(path).name} as an image: {exc}")
        return None


def _file_paths(files: Union[None, str, Sequence[Any]]) -> List[str]:
    """Paths of the uploaded files (Gradio passes path strings), capped at MAX_IMAGES with a warning."""
    if files is None:
        files = []
    elif isinstance(files, str):
        files = [files]
    paths = [str(item) for item in files if item]
    if not paths:
        shell.fail("Please upload at least one image.")
    if len(paths) > MAX_IMAGES:
        shell.warn(f"Only the first {MAX_IMAGES} of {len(paths)} images were captioned.")
        paths = paths[:MAX_IMAGES]
    return paths


# --------------------------------------------------------------------------- captioning


def _caption(captioner: Any, image: Image.Image) -> str:
    result = captioner(image, max_new_tokens=MAX_CAPTION_TOKENS)
    if result and isinstance(result[0], list):
        result = result[0]
    return result[0]["generated_text"].strip()


def generate_caption(image: Optional[Image.Image], model_name: str) -> Tuple[str, Dict[str, Any]]:
    """Caption one image; returns the caption and a details summary."""
    image = _rgb(image)
    choice = CAPTION_MODELS.get(model_name)
    captioner = CAPTION_MODELS.load(model_name)
    try:
        caption = _caption(captioner, image)
    except Exception as exc:
        shell.fail_from(exc, "Image captioning")
    details = {"model": choice.name, "model_id": choice.model_id, "image_size": f"{image.width}x{image.height}"}
    return caption, details


def batch_caption(files: Union[None, str, Sequence[Any]], model_name: str) -> Tuple[pd.DataFrame, str]:
    """Caption up to MAX_IMAGES uploaded images; unreadable files are reported in their row."""
    paths = _file_paths(files)
    captioner = CAPTION_MODELS.load(model_name)
    rows: List[List] = []
    skipped = failed = 0
    for index, path in enumerate(paths, 1):
        name = Path(path).name
        image = _open_image(path)
        if image is None:
            skipped += 1
            rows.append([index, name, "", "not a readable image"])
            continue
        try:
            rows.append([index, name, f"{image.width}x{image.height}", _caption(captioner, image)])
        except Exception as exc:
            logger.warning(f"Captioning {name} failed: {exc}")
            failed += 1
            rows.append([index, name, f"{image.width}x{image.height}", f"error: {str(exc)[:120]}"])

    if skipped:
        shell.warn(f"{skipped} file(s) were skipped because they are not readable images.")
    captioned = len(paths) - skipped - failed
    summary = f"{captioned} of {len(paths)} images captioned with {CAPTION_MODELS.get(model_name).name}."
    if failed:
        summary += f" {failed} image(s) failed; see the Caption column."
    return pd.DataFrame(rows, columns=BATCH_COLUMNS), summary


# --------------------------------------------------------------------------- image-text matching


def _descriptions(value: Optional[str]) -> List[str]:
    """One description per line, or a single comma-separated line; duplicates removed."""
    text = (value or "").strip()
    delimiter = "," if "\n" not in text and "," in text else "\n"
    items = shell.split_items(text, delimiter, max_items=MAX_DESCRIPTIONS, what="two descriptions")
    unique = list(dict.fromkeys(items))
    if len(unique) < 2:
        shell.fail("Enter at least two different descriptions: the scores rank them against each other.")
    return unique


def _scores_html(scores: List[Tuple[str, float]]) -> str:
    rows = "".join(
        "<tr>"
        f"<td style='padding:6px 8px'>{shell.esc(label)}</td>"
        "<td style='padding:6px 8px;width:45%'>"
        f"<div style='background:var(--color-accent);height:14px;border-radius:4px;width:{score * 100:.1f}%'></div>"
        "</td>"
        f"<td style='padding:6px 8px;text-align:right'>{score:.1%}</td>"
        "</tr>"
        for label, score in scores
    )
    note = shell.esc(RELATIVE_NOTE.format(n=len(scores)))
    return (
        "<div>"
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr><th style='text-align:left;padding:6px 8px'>Description</th><th></th>"
        "<th style='text-align:right;padding:6px 8px'>Relative score</th></tr>"
        f"{rows}</table>"
        f"<p style='opacity:0.8;font-size:0.9em'>{note}</p>"
        "</div>"
    )


def match_image(image: Optional[Image.Image], descriptions: str, model_name: str) -> Tuple[Dict[str, Any], str]:
    """Rank the user's descriptions for one image with CLIP; returns a summary and an HTML table."""
    image = _rgb(image)
    texts = _descriptions(descriptions)
    choice = MATCH_MODELS.get(model_name)
    matcher = MATCH_MODELS.load(model_name)
    labels = [shell.clip_for_model(text, matcher, label=f"Description {i}") for i, text in enumerate(texts, 1)]
    try:
        raw = matcher(image, candidate_labels=labels, hypothesis_template="{}")
    except Exception as exc:
        shell.fail_from(exc, "Image-text matching")

    scores = sorted(((item["label"], float(item["score"])) for item in raw), key=lambda pair: pair[1], reverse=True)
    result = {
        "best_match": scores[0][0],
        "relative_scores": {label: round(score, 4) for label, score in scores},
        "note": RELATIVE_NOTE.format(n=len(scores)),
        "model": choice.model_id,
    }
    return result, _scores_html(scores)


ABOUT = """
**Caption** writes a short description of an image with an image-to-text model. **Match** scores how well
each of your descriptions fits an image with CLIP.

**CLIP scores are relative.** They are a softmax over the descriptions you enter, so they always sum to 100%
and only say which of *your* descriptions fits best. Adding or removing a description changes every score,
and a top score of 90% does not mean the description is accurate. Phrase descriptions like captions
("a photo of a dog on a beach") rather than single words.

**Uploads:** images only (PNG, JPEG, WebP, ...), up to 10 MB each. *Batch captions* handles up to
{max_images} images per run; extra files are ignored with a warning and unreadable files are reported in
their row. Use *Use the sample image* to try the tools without uploading anything.

Captions describe what is visible and can miss context; both tools work best on natural photos.
""".format(max_images=MAX_IMAGES)


def create_ui() -> gr.Blocks:
    """Build the vision-language page."""
    with shell.page(UI_KEY) as demo:
        with gr.Tab("Caption"):
            with gr.Row():
                with gr.Column():
                    caption_image = gr.Image(label="Image", type="pil", image_mode="RGB")
                    caption_model = CAPTION_MODELS.dropdown()
                    with gr.Row():
                        caption_sample = gr.Button("Use the sample image", size="sm")
                        caption_btn = gr.Button("Generate caption", variant="primary")
                with gr.Column():
                    caption_output = gr.Textbox(label="Caption", lines=2, interactive=False)
                    caption_details = gr.JSON(label="Details")

        with gr.Tab("Match image to text"):
            gr.Markdown(
                "Enter two or more descriptions (one per line, or one comma-separated line). CLIP ranks them "
                "against each other for this image; the scores are relative, not absolute."
            )
            with gr.Row():
                with gr.Column():
                    match_input_image = gr.Image(label="Image", type="pil", image_mode="RGB")
                    match_texts = gr.Textbox(
                        label=f"Descriptions (up to {MAX_DESCRIPTIONS})",
                        placeholder="a photo of a cat\na photo of a dog\na photo of a bird",
                        lines=5,
                    )
                    match_model = MATCH_MODELS.dropdown()
                    with gr.Row():
                        match_sample = gr.Button("Use the sample image", size="sm")
                        match_btn = gr.Button("Rank descriptions", variant="primary")
                with gr.Column():
                    match_html = gr.HTML(label="Relative scores")
                    match_json = gr.JSON(label="Scores")

        with gr.Tab("Batch captions"):
            gr.Markdown(f"Upload up to {MAX_IMAGES} images; each one gets a caption.")
            with gr.Row():
                with gr.Column(scale=2):
                    batch_files = gr.File(label="Images", file_count="multiple", file_types=["image"])
                    batch_model = CAPTION_MODELS.dropdown()
                    batch_btn = gr.Button("Caption all", variant="primary")
                with gr.Column(scale=3):
                    batch_summary = gr.Markdown()
                    batch_output = gr.Dataframe(headers=BATCH_COLUMNS, label="Captions", wrap=True, interactive=False)

        shell.about(ABOUT, CAPTION_MODELS, MATCH_MODELS)

        caption_sample.click(sample_image, outputs=caption_image)
        caption_btn.click(
            generate_caption, inputs=[caption_image, caption_model], outputs=[caption_output, caption_details]
        )
        match_sample.click(lambda: (sample_image(), SAMPLE_DESCRIPTIONS), outputs=[match_input_image, match_texts])
        match_btn.click(
            match_image, inputs=[match_input_image, match_texts, match_model], outputs=[match_json, match_html]
        )
        batch_btn.click(batch_caption, inputs=[batch_files, batch_model], outputs=[batch_output, batch_summary])
    return demo


if __name__ == "__main__":
    shell.launch(create_ui(), UI_KEY, argv=sys.argv[1:])
