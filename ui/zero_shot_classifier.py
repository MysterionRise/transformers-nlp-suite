"""
Zero-Shot Classification - sort text into labels you define, without training data.

Models come from the ``zero_shot`` registry category (natural language inference
checkpoints) and load through the shared model cache. Each label is turned into a
hypothesis ("This example is {label}.") and the model scores whether the text
entails it.
"""

import re
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

try:
    from ui import common as shell
except ModuleNotFoundError:  # running as `python ui/zero_shot_classifier.py`
    import common as shell  # type: ignore[no-redef]

logger = shell.get_logger(__name__)

UI_KEY = "zero_shot"
MODELS = shell.ModelMenu("zero_shot", default="bart_large_mnli")

MAX_LABELS = 20
MAX_LABEL_CHARS = 100
HYPOTHESIS_RESERVE = 32  # tokens left free for "This example is {label}."
APPLIES_THRESHOLD = 0.5
BATCH_COLUMNS = ["#", "Text", "Top label", "Score", "All scores"]

SINGLE_LABEL_NOTE = (
    "Single-label: the scores are a softmax across your labels and add up to 100%. They say which label fits best "
    "relative to the others, not whether any of them fits."
)
MULTI_LABEL_NOTE = (
    "Multi-label: each label is scored on its own (entailment against contradiction), so the scores do not add up "
    f"to 100%. Several labels, or none, can score high; labels at {APPLIES_THRESHOLD:.0%} or more are listed "
    "as applying."
)

COLOR_TOP = "#2a9d8f"
COLOR_OTHER = "#8d99ae"

# Pre-defined label sets for common tasks
LABEL_PRESETS = {
    "Sentiment": "positive, negative, neutral",
    "Intent": "question, statement, command, small talk",
    "Category": "sports, politics, technology, entertainment, science",
    "Emotion": "joy, sadness, anger, surprise, fear, disgust",
    "Toxicity": "toxic, hateful, offensive, safe",
    "Language": "English, Spanish, French, German, Chinese",
}

# Industry-specific label presets
INDUSTRY_PRESETS = {
    # Retail / E-commerce
    "Retail: Feedback Type": "complaint, praise, question, suggestion, return request",
    "Retail: Issue Category": "shipping, product quality, pricing, customer service, website issue",
    "Retail: Purchase Intent": "ready to buy, comparing options, just browsing, seeking information",
    "Retail: Customer Sentiment": "satisfied, dissatisfied, neutral, frustrated, delighted",
    # Publishing / Media
    "Publishing: Article Type": "news, opinion, analysis, review, interview, breaking news",
    "Publishing: Topic": "politics, business, technology, science, health, entertainment, sports",
    "Publishing: Tone": "objective, persuasive, informative, sensational, critical",
    "Publishing: Audience": "general public, experts, students, professionals, policymakers",
    # Life Sciences / Healthcare
    "Medical: Note Type": "diagnosis, treatment plan, progress note, discharge summary, consultation",
    "Medical: Urgency": "emergency, urgent, routine, elective, preventive",
    "Medical: Specialty": "cardiology, oncology, neurology, orthopedics, pediatrics, psychiatry",
    "Medical: Clinical Intent": "symptom report, medication inquiry, test results, follow-up, adverse event",
}
INDUSTRY_GROUPS = {"Retail": "Retail", "Publishing": "Publishing", "Medical": "Medical / life sciences"}


# --------------------------------------------------------------------------- input


def parse_labels(value: str, multi_label: bool = False) -> List[str]:
    """Comma- or newline-separated labels: stripped, de-duplicated (ignoring case) and validated."""
    entered = [" ".join(item.split()) for item in re.split(r"[,\n]", value or "")]
    entered = [label for label in entered if label]
    labels: List[str] = []
    seen = set()
    for label in entered:
        if label.casefold() not in seen:
            seen.add(label.casefold())
            labels.append(label)
    if not labels:
        shell.fail("Please enter candidate labels, separated by commas.")
    duplicates = len(entered) - len(labels)
    if duplicates:
        shell.warn(f"{duplicates} duplicate label(s) were ignored.")
    if len(labels) > MAX_LABELS:
        shell.fail(f"Use at most {MAX_LABELS} labels (got {len(labels)}); each label costs one model pass.")
    too_long = [label for label in labels if len(label) > MAX_LABEL_CHARS]
    if too_long:
        shell.fail(f"Labels must be at most {MAX_LABEL_CHARS} characters; shorten {too_long[0][:40]!r}...")
    if not multi_label and len(labels) < 2:
        shell.fail(
            "Single-label mode needs at least 2 different labels to choose between. "
            "Add another label or switch on multi-label mode to score one label on its own."
        )
    return labels


# --------------------------------------------------------------------------- inference


def _scores(raw: Dict[str, Any]) -> List[Tuple[str, float]]:
    pairs = [(str(label), float(score)) for label, score in zip(raw["labels"], raw["scores"])]
    return sorted(pairs, key=lambda pair: pair[1], reverse=True)


def _chart(scores: List[Tuple[str, float]], multi_label: bool) -> go.Figure:
    labels = [label for label, _ in reversed(scores)]
    values = [score for _, score in reversed(scores)]
    if multi_label:
        colors = [COLOR_TOP if value >= APPLIES_THRESHOLD else COLOR_OTHER for value in values]
        title = "Probability that each label applies (independent)"
    else:
        colors = [COLOR_TOP if index == len(values) - 1 else COLOR_OTHER for index in range(len(values))]
        title = "Share of probability across labels (adds up to 100%)"
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{value:.1%}" for value in values],
            textposition="auto",
        )
    )
    if multi_label:
        fig.add_vline(x=APPLIES_THRESHOLD, line_dash="dot", line_color=COLOR_OTHER)
    fig.update_layout(
        xaxis=dict(range=[0, 1], title=title, tickformat=".0%"),
        height=80 + 40 * len(labels),
        margin=dict(l=10, r=10, t=10, b=50),
    )
    return fig


def classify_text(
    text: str,
    candidate_labels: str,
    model_name: str,
    multi_label: bool = False,
) -> Tuple[Dict[str, Any], go.Figure]:
    """Classify one text; returns a result summary and a score chart."""
    text = shell.require_text(text, "some text to classify")
    multi_label = bool(multi_label)
    labels = parse_labels(candidate_labels, multi_label)
    choice = MODELS.get(model_name)
    classifier = MODELS.load(model_name)
    used = shell.clip_for_model(text, classifier, label="Text", reserve=HYPOTHESIS_RESERVE)
    try:
        raw = classifier(used, candidate_labels=labels, multi_label=multi_label)
    except Exception as exc:
        shell.fail_from(exc, "Zero-shot classification")

    scores = _scores(raw)
    top_label, top_score = scores[0]
    result: Dict[str, Any] = {
        "mode": "multi-label" if multi_label else "single-label",
        "top_label": top_label,
        "score": f"{top_score:.1%}",
        "how_to_read": MULTI_LABEL_NOTE if multi_label else SINGLE_LABEL_NOTE,
    }
    if multi_label:
        result["labels_that_apply"] = [label for label, score in scores if score >= APPLIES_THRESHOLD]
    result.update(
        {
            "scores": {label: round(score, 4) for label, score in scores},
            "model": choice.model_id,
            "truncated": used != text,
        }
    )
    return result, _chart(scores, multi_label)


def batch_classify(
    batch_text: str,
    candidate_labels: str,
    model_name: str,
    multi_label: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """Classify one text per line against the same labels; a failing line is reported in its row."""
    items = shell.split_items(batch_text, what="one line of text")
    multi_label = bool(multi_label)
    labels = parse_labels(candidate_labels, multi_label)
    choice = MODELS.get(model_name)
    classifier = MODELS.load(model_name)
    budget = max(1, shell.token_limit(classifier) - HYPOTHESIS_RESERVE)

    rows, counts, truncated = [], Counter(), 0
    for index, item in enumerate(items, 1):
        used, _ = shell.truncate_to_tokens(item, classifier.tokenizer, budget)
        truncated += used != item
        try:
            scores = _scores(classifier(used, candidate_labels=labels, multi_label=multi_label))
        except Exception as exc:
            logger.warning(f"Batch item {index} failed: {exc}")
            rows.append([index, item, "error", "", str(exc)[:120]])
            continue
        label, score = scores[0]
        counts[label] += 1
        rows.append([index, item, label, f"{score:.1%}", " · ".join(f"{lbl} {s:.0%}" for lbl, s in scores)])

    if truncated:
        shell.warn(f"{truncated} line(s) exceeded the model's {budget}-token limit and were truncated.")
    mode = "multi-label, scores are independent" if multi_label else "single-label, scores add up to 100%"
    summary = " · ".join(f"**{label}**: {count}" for label, count in counts.most_common()) or "No results"
    return (
        pd.DataFrame(rows, columns=BATCH_COLUMNS),
        f"{len(items)} texts classified with {choice.name} ({mode}). {summary}",
    )


EXAMPLES = [
    ["This movie was amazing! I loved every second of it.", "positive, negative, neutral", "BART Large MNLI", False],
    ["What time does the store open tomorrow?", "question, statement, command, small talk", "BART Large MNLI", False],
    [
        "I've been waiting 2 weeks for my order and customer service won't respond to my emails!",
        "complaint, praise, question, suggestion, return request",
        "BART Large MNLI",
        False,
    ],
    [
        "The product quality is excellent but shipping took too long. Price was fair though.",
        "shipping, product quality, pricing, customer service, website issue",
        "BART Large MNLI",
        True,
    ],
    [
        "In my view, the government's new climate policy falls short of what scientists recommend.",
        "objective, persuasive, informative, sensational, critical",
        "DeBERTa v3 NLI",
        False,
    ],
    [
        "Patient presents with chest pain radiating to left arm, shortness of breath. ECG shows ST elevation.",
        "emergency, urgent, routine, elective, preventive",
        "DeBERTa v3 NLI",
        False,
    ],
]

ABOUT = f"""
Zero-shot classification reuses a natural language inference (NLI) model: for each label it asks whether the text
entails *"This example is {{label}}."* No training data is needed, but the wording of the labels matters.

**Reading the scores**

- **Single-label** (default): the entailment scores are normalised with a softmax across your labels, so they add up
  to 100%. The top label is the best fit *among the labels you gave*; a high score does not mean the label is right
  if none of them really fits.
- **Multi-label**: every label is scored independently (entailment against contradiction for that label alone), so
  the scores do not add up to 100% and several labels can apply at once. Labels at
  {APPLIES_THRESHOLD:.0%} or more are listed as applying.

**Labels**: separate them with commas; duplicates are ignored. Single-label mode needs at least 2 labels; at most
{MAX_LABELS} are allowed, because each label is one model pass. Short, concrete labels ("shipping delay") work better
than abstract ones.
"""


def _preset_button(label: str, preset: str, target: gr.Textbox) -> None:
    """Button that fills `target` with a preset label set (a page helper, not an API endpoint)."""
    gr.Button(label, size="sm").click(lambda: preset, outputs=target, api_visibility="private")


def create_ui() -> gr.Blocks:
    """Build the zero-shot classification page."""
    with shell.page(UI_KEY) as demo:
        with gr.Tab("Single text"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Text to classify", placeholder="Type or paste your text here...", lines=5
                    )
                    labels_input = gr.Textbox(
                        label="Candidate labels (comma-separated)",
                        placeholder="positive, negative, neutral",
                        lines=2,
                    )
                    model_dropdown = MODELS.dropdown()
                    multi_label_toggle = gr.Checkbox(
                        label="Multi-label: score each label independently (several can apply)", value=False
                    )
                    classify_btn = gr.Button("Classify", variant="primary")
                with gr.Column(scale=2):
                    output_json = gr.JSON(label="Result")
                    output_plot = gr.Plot(label="Scores")

            with gr.Accordion("Label presets", open=True):
                with gr.Row():
                    gr.Markdown("**General**")
                    for name, preset in LABEL_PRESETS.items():
                        _preset_button(name, preset, labels_input)
                for prefix, title in INDUSTRY_GROUPS.items():
                    with gr.Row():
                        gr.Markdown(f"**{title}**")
                        for name, preset in INDUSTRY_PRESETS.items():
                            if name.startswith(f"{prefix}:"):
                                _preset_button(name.split(": ", 1)[1], preset, labels_input)

            gr.Examples(
                examples=EXAMPLES,
                inputs=[text_input, labels_input, model_dropdown, multi_label_toggle],
                label="Try an example",
            )

        with gr.Tab("Batch"):
            gr.Markdown(f"One text per line, up to {shell.MAX_BATCH_ITEMS} lines, all scored against the same labels.")
            with gr.Row():
                with gr.Column(scale=2):
                    batch_input = gr.Textbox(label="Texts (one per line)", placeholder="Text 1\nText 2", lines=10)
                    batch_labels = gr.Textbox(label="Candidate labels", placeholder="label1, label2, label3", lines=2)
                    batch_model = MODELS.dropdown()
                    batch_multi_label = gr.Checkbox(label="Multi-label", value=False)
                    batch_btn = gr.Button("Classify all", variant="primary")
                with gr.Column(scale=3):
                    batch_summary = gr.Markdown()
                    batch_output = gr.Dataframe(headers=BATCH_COLUMNS, label="Results", wrap=True, interactive=False)

        shell.about(ABOUT, MODELS)

        inputs = [text_input, labels_input, model_dropdown, multi_label_toggle]
        classify_btn.click(classify_text, inputs=inputs, outputs=[output_json, output_plot])
        batch_btn.click(
            batch_classify,
            inputs=[batch_input, batch_labels, batch_model, batch_multi_label],
            outputs=[batch_output, batch_summary],
        )
    return demo


if __name__ == "__main__":
    shell.launch(create_ui(), UI_KEY, argv=sys.argv[1:])
