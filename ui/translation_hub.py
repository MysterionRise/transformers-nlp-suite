"""
Translation - translate text with Helsinki OPUS-MT pair models or mBART-50.

Models come from the ``translation`` registry category and load through the
shared model cache. Helsinki OPUS-MT (the default) is one small model per
language pair: the registry entry is the en-es model and lists the other
supported pairs, which load by id (``Helsinki-NLP/opus-mt-{pair}``). mBART-50
is one large multilingual model (~2.4 GB) that takes the source and target
language codes on every call.
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import gradio as gr
import pandas as pd

try:
    from ui import common as shell
except ModuleNotFoundError:  # running as `python ui/translation_hub.py`
    import common as shell  # type: ignore[no-redef]

logger = shell.get_logger(__name__)

UI_KEY = "translation"
MODELS = shell.ModelMenu("translation", keys=["helsinki_opus", "mbart_50"], default="helsinki_opus")
HELSINKI_KEY = "helsinki_opus"
MBART_KEY = "mbart_50"
MBART_NOTICE = "mBART-50 is loading. Its first use downloads about 2.4 GB, which can take several minutes."

# Display name -> (ISO 639-1 code used by OPUS-MT model ids, mBART-50 language code).
LANGUAGES: Dict[str, Tuple[str, str]] = {
    "English": ("en", "en_XX"),
    "Spanish": ("es", "es_XX"),
    "French": ("fr", "fr_XX"),
    "German": ("de", "de_DE"),
    "Italian": ("it", "it_IT"),
    "Portuguese": ("pt", "pt_XX"),
    "Russian": ("ru", "ru_RU"),
    "Chinese": ("zh", "zh_CN"),
    "Japanese": ("ja", "ja_XX"),
    "Korean": ("ko", "ko_KR"),
    "Arabic": ("ar", "ar_AR"),
    "Hindi": ("hi", "hi_IN"),
}
LANGUAGE_NAMES = {iso: name for name, (iso, _) in LANGUAGES.items()}

MAX_NEW_TOKENS = 512
BATCH_COLUMNS = ["#", "Original", "Translation"]
SAMPLES_FILE = Path(__file__).resolve().parent.parent / "data" / "translation_samples.txt"
_SAMPLE_HEADER = re.compile(r"^Sample\s+(\d+)\s*\(\s*(\w+)\s+to\s+(\w+)\s*\):?$", re.IGNORECASE)
_SEPARATOR = re.compile(r"(?m)^\s*-{3,}\s*$")

# Models announced with a "first use downloads..." toast in this process.
_announced: Set[str] = set()


# --------------------------------------------------------------------------- languages and models


def helsinki_pairs() -> List[str]:
    """OPUS-MT language pairs listed in the registry (for example "en-es")."""
    config = MODELS.get(HELSINKI_KEY).config
    return list(config.language_pairs or [config.model_id.rsplit("opus-mt-", 1)[-1]])


def _pair_label(pair: str) -> str:
    source, _, target = pair.partition("-")
    return f"{LANGUAGE_NAMES.get(source, source)} → {LANGUAGE_NAMES.get(target, target)}"


def _language(name: Optional[str], role: str) -> Tuple[str, str]:
    if name not in LANGUAGES:
        shell.fail(f"Choose a {role} language from the list.")
    return LANGUAGES[name]


def _load_helsinki(pair: str) -> Tuple[Any, str]:
    """Pipeline and model id for one OPUS-MT pair; the registry entry serves its own pair."""
    choice = MODELS.get(HELSINKI_KEY)
    if choice.model_id.endswith(f"opus-mt-{pair}"):
        return MODELS.load(choice.name), choice.model_id
    model_id = f"Helsinki-NLP/opus-mt-{pair}"
    return shell.load_by_id(model_id, "translation", action=f"Loading Helsinki OPUS-MT {pair}"), model_id


def _resolve(model_name: str, source: str, target: str) -> Tuple[Any, Dict[str, str], str]:
    """
    Validate the request and load its model.

    Returns the pipeline, the per-call language arguments and the model id.
    """
    choice = MODELS.get(model_name)
    source_iso, source_code = _language(source, "source")
    target_iso, target_code = _language(target, "target")
    if source == target:
        shell.fail("Source and target languages are the same. Pick two different languages.")

    if choice.key == MBART_KEY:
        if MBART_KEY not in _announced:
            shell.info(MBART_NOTICE)
        translator = MODELS.load(choice.name)
        _announced.add(MBART_KEY)
        return translator, {"src_lang": source_code, "tgt_lang": target_code}, choice.model_id

    pair = f"{source_iso}-{target_iso}"
    pairs = helsinki_pairs()
    if pair not in pairs:
        supported = ", ".join(_pair_label(p) for p in pairs)
        shell.fail(
            f"Helsinki OPUS-MT has no {source} → {target} model here. Supported pairs: {supported}. "
            "Choose mBART-50 for other combinations."
        )
    translator, model_id = _load_helsinki(pair)
    return translator, {}, model_id


def _generate(translator: Any, text: str, lang_kwargs: Dict[str, str]) -> str:
    """Translate one (already clipped) text. New tokens scale with the input, capped at MAX_NEW_TOKENS."""
    budget = min(MAX_NEW_TOKENS, 3 * shell.count_tokens(translator.tokenizer, text) + 20)
    result = translator(text, max_new_tokens=budget, **lang_kwargs)
    return result[0]["translation_text"].strip()


# --------------------------------------------------------------------------- handlers


def translate(text: str, model_name: str, source_lang: str, target_lang: str) -> Tuple[str, Dict[str, Any]]:
    """Translate one text; returns the translation and a details summary."""
    text = shell.require_text(text, "some text to translate")
    translator, lang_kwargs, model_id = _resolve(model_name, source_lang, target_lang)
    used = shell.clip_for_model(text, translator, label="Text")
    try:
        translation = _generate(translator, used, lang_kwargs)
    except Exception as exc:
        shell.fail_from(exc, "Translation")

    details = {
        "model": MODELS.get(model_name).name,
        "model_id": model_id,
        "source": source_lang,
        "target": target_lang,
        "truncated": used != text,
    }
    return translation, details


def _translate_rows(translator: Any, items: List[str], lang_kwargs: Dict[str, str]) -> Tuple[List[List], int, int]:
    """Translate each line; a failing line is reported in its row. Returns rows, truncated and failed counts."""
    limit = shell.token_limit(translator)
    rows: List[List] = []
    truncated = failed = 0
    for index, item in enumerate(items, 1):
        used, _ = shell.truncate_to_tokens(item, translator.tokenizer, limit)
        truncated += used != item
        try:
            rows.append([index, item, _generate(translator, used, lang_kwargs)])
        except Exception as exc:
            logger.warning(f"Batch line {index} failed: {exc}")
            failed += 1
            rows.append([index, item, f"error: {str(exc)[:120]}"])
    return rows, truncated, failed


def batch_translate(text: str, model_name: str, source_lang: str, target_lang: str) -> Tuple[pd.DataFrame, str]:
    """Translate one text per line with the selected model and languages."""
    items = shell.split_items(text, what="one line of text to translate")
    translator, lang_kwargs, _ = _resolve(model_name, source_lang, target_lang)
    rows, truncated, failed = _translate_rows(translator, items, lang_kwargs)
    if truncated:
        shell.warn(f"{truncated} line(s) exceeded the model's token limit and were truncated.")

    summary = f"{len(items) - failed} of {len(items)} lines translated from {source_lang} to {target_lang}"
    summary += f" with {MODELS.get(model_name).name}."
    if failed:
        summary += f" {failed} line(s) failed; see the Translation column."
    return pd.DataFrame(rows, columns=BATCH_COLUMNS), summary


# --------------------------------------------------------------------------- samples


def load_samples(path: Path = SAMPLES_FILE) -> Dict[str, Tuple[str, str, str]]:
    """
    Parse ``data/translation_samples.txt``: blocks separated by ``---`` lines, each
    starting with a ``Sample N (Source to Target):`` header line.

    Returns {title: (text, source language, target language)}; blocks with an
    unknown language or no text are skipped.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Translation samples not loaded: {exc}")
        return {}
    samples: Dict[str, Tuple[str, str, str]] = {}
    for block in _SEPARATOR.split(raw):
        header, _, body = block.strip().partition("\n")
        match = _SAMPLE_HEADER.match(header.strip())
        if not match or not body.strip():
            continue
        number, source, target = match.groups()
        source, target = source.capitalize(), target.capitalize()
        if source in LANGUAGES and target in LANGUAGES:
            samples[f"Sample {number}: {source} → {target}"] = (body.strip(), source, target)
    return samples


SAMPLES = load_samples()


def use_sample(title: Optional[str]) -> Tuple[str, str, str]:
    """Fill the text and language inputs from a sample."""
    if title not in SAMPLES:
        shell.fail("Choose a sample from the list.")
    return SAMPLES[title]


BATCH_EXAMPLE = (
    "Where is the nearest train station?\n"
    "The meeting has been moved to Thursday afternoon.\n"
    "Please send me the invoice by email.\n"
    "Our store opens at nine in the morning."
)

ABOUT = """
**Helsinki OPUS-MT** (default) uses one small model (about 300 MB) per language pair, so it is fast but only
covers the pairs listed below. **mBART-50** is a single multilingual model that translates between any two of
the languages in the menus, but its first use downloads about 2.4 GB and it is slower on CPU.

**How to use:** pick the model and languages, type or paste text (or load one of the samples from
`data/translation_samples.txt`) and press *Translate*. The *Batch* tab translates one text per line with the
model and languages selected in that tab.

Neural translation can get names, numbers and technical terms wrong; check important translations.
"""


def _language_dropdowns(source: str = "English", target: str = "Spanish") -> Tuple[gr.Dropdown, gr.Dropdown]:
    names = list(LANGUAGES)
    return (
        gr.Dropdown(choices=names, value=source, label="From"),
        gr.Dropdown(choices=names, value=target, label="To"),
    )


def _model_radio() -> gr.Radio:
    return MODELS.radio(
        info="Helsinki OPUS-MT: small, one model per pair. mBART-50: any pair below, ~2.4 GB download on first use."
    )


def create_ui() -> gr.Blocks:
    """Build the translation page."""
    pairs_note = "Helsinki OPUS-MT pairs: " + ", ".join(_pair_label(pair) for pair in helsinki_pairs()) + "."
    with shell.page(UI_KEY) as demo:
        with gr.Tab("Translate"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(label="Text to translate", placeholder="Type or paste text...", lines=6)
                    sample_menu = gr.Dropdown(
                        choices=list(SAMPLES),
                        value=None,
                        label="Load a sample",
                        info="Fills the text and languages from data/translation_samples.txt.",
                        visible=bool(SAMPLES),
                    )
                with gr.Column(scale=2):
                    model_radio = _model_radio()
                    with gr.Row():
                        source_dropdown, target_dropdown = _language_dropdowns()
                    gr.Markdown(pairs_note)
                    translate_btn = gr.Button("Translate", variant="primary")
            with gr.Row():
                translation_output = gr.Textbox(label="Translation", lines=6, buttons=["copy"], interactive=False)
                details_output = gr.JSON(label="Details")

        with gr.Tab("Batch"):
            gr.Markdown(
                f"One text per line, up to {shell.MAX_BATCH_ITEMS} lines. Every line is translated with the model "
                "and languages selected here."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    batch_input = gr.Textbox(
                        label="Texts (one per line)", placeholder="First text\nSecond text", lines=10
                    )
                    gr.Button("Load example lines", size="sm").click(lambda: BATCH_EXAMPLE, outputs=batch_input)
                    batch_model = _model_radio()
                    with gr.Row():
                        batch_source, batch_target = _language_dropdowns()
                    batch_btn = gr.Button("Translate all", variant="primary")
                with gr.Column(scale=3):
                    batch_summary = gr.Markdown()
                    batch_output = gr.Dataframe(headers=BATCH_COLUMNS, label="Results", wrap=True, interactive=False)

        shell.about(ABOUT + "\n" + pairs_note, MODELS)

        single_inputs = [text_input, model_radio, source_dropdown, target_dropdown]
        translate_btn.click(translate, inputs=single_inputs, outputs=[translation_output, details_output])
        sample_menu.input(use_sample, inputs=sample_menu, outputs=[text_input, source_dropdown, target_dropdown])
        batch_btn.click(
            batch_translate,
            inputs=[batch_input, batch_model, batch_source, batch_target],
            outputs=[batch_output, batch_summary],
        )
    return demo


if __name__ == "__main__":
    shell.launch(create_ui(), UI_KEY, argv=sys.argv[1:])
