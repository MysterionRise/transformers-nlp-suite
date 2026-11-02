"""
Sentiment Analysis - classify text as negative, neutral or positive.

Models come from the ``sentiment_analysis``, ``financial_sentiment`` and
``retail`` registry categories and load through the shared model cache, so the
same weights serve this page, the other tools and the API in one process.
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
except ModuleNotFoundError:  # running as `python ui/sentiment_playground.py`
    import common as shell  # type: ignore[no-redef]

logger = shell.get_logger(__name__)

UI_KEY = "sentiment"
MODELS = shell.ModelMenu(
    "sentiment_analysis",
    "financial_sentiment",
    "retail",
    default="twitter_roberta_multilingual",
)

# Colour by meaning, not by score, so a winning "negative" bar is never green.
LABEL_COLORS = {"negative": "#d1495b", "neutral": "#8d99ae", "positive": "#2a9d8f"}
STAR_COLORS = {1: "#d1495b", 2: "#ee8959", 3: "#8d99ae", 4: "#84b59f", 5: "#2a9d8f"}
BATCH_COLUMNS = ["#", "Text", "Sentiment", "Confidence"]


def _label_color(label: str) -> str:
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    stars = re.match(r"^(\d)\s*star", label)
    if stars:
        return STAR_COLORS.get(int(stars.group(1)), "#8d99ae")
    return "#6c757d"


def _scores(raw: Any, classifier: Any) -> List[Tuple[str, float]]:
    """Normalise pipeline output to [(label, score)] sorted by score, highest first."""
    if isinstance(raw, dict):
        raw = [raw]
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    num_labels = shell.num_labels_of(classifier)
    scores = [(shell.normalize_sentiment_label(item["label"], num_labels), float(item["score"])) for item in raw]
    return sorted(scores, key=lambda pair: pair[1], reverse=True)


def _chart(scores: List[Tuple[str, float]]) -> go.Figure:
    labels = [label for label, _ in reversed(scores)]
    values = [score for _, score in reversed(scores)]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=[_label_color(label) for label in labels],
            text=[f"{value:.1%}" for value in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1], title="Probability", tickformat=".0%"),
        height=80 + 45 * len(labels),
        margin=dict(l=10, r=10, t=10, b=40),
    )
    return fig


def analyze_sentiment(text: str, model_name: str) -> Tuple[Dict[str, Any], go.Figure]:
    """Classify one text; returns a result summary and a score chart."""
    text = shell.require_text(text, "some text to analyze")
    choice = MODELS.get(model_name)
    classifier = MODELS.load(model_name)
    used = shell.clip_for_model(text, classifier, label="Text")
    try:
        raw = classifier(used, top_k=None)
    except Exception as exc:
        shell.fail_from(exc, "Sentiment analysis")

    scores = _scores(raw, classifier)
    top_label, top_score = scores[0]
    result = {
        "prediction": top_label,
        "confidence": f"{top_score:.1%}",
        "scores": {label: round(score, 4) for label, score in scores},
        "model": choice.model_id,
        "truncated": used != text,
    }
    return result, _chart(scores)


def analyze_batch(text: str, model_name: str) -> Tuple[pd.DataFrame, str]:
    """Classify one text per line; a failing line is reported in its row, not for the whole batch."""
    items = shell.split_items(text, what="one line of text")
    classifier = MODELS.load(model_name)
    limit = shell.token_limit(classifier)

    rows, counts, truncated = [], Counter(), 0
    for index, item in enumerate(items, 1):
        used, _ = shell.truncate_to_tokens(item, classifier.tokenizer, limit)
        truncated += used != item
        try:
            raw = classifier(used, top_k=None)
            label, score = _scores(raw, classifier)[0]
            counts[label] += 1
            rows.append([index, item, label, f"{score:.1%}"])
        except Exception as exc:
            logger.warning(f"Batch item {index} failed: {exc}")
            rows.append([index, item, "error", str(exc)[:120]])

    if truncated:
        shell.warn(f"{truncated} line(s) exceeded the model's {limit}-token limit and were truncated.")
    summary = " · ".join(f"**{label}**: {count}" for label, count in counts.most_common()) or "No results"
    return pd.DataFrame(rows, columns=BATCH_COLUMNS), f"{len(items)} texts analyzed. {summary}"


EXAMPLES = [
    ["This movie was absolutely fantastic! I loved every minute of it.", "Twitter RoBERTa (Multilingual)"],
    ["I'm so disappointed with this product. Complete waste of money.", "Twitter RoBERTa (Multilingual)"],
    ["El servicio fue rápido y el personal muy amable.", "Twitter RoBERTa (Multilingual)"],
    [
        "Terrible experience with the delivery. Package was damaged and customer service was unhelpful.",
        "Twitter RoBERTa (English)",
    ],
    ["The running shoes are comfortable but overpriced. Decent quality overall.", "DistilBERT SST-2"],
    ["Another biased and misleading article. The author clearly didn't research the facts.", "BERT Base (SST-2)"],
    [
        "Quarterly revenue grew 12% and the company raised its full-year guidance.",
        "FinBERT",
    ],
    ["Battery died after two weeks and the seller ignored my emails.", "Amazon Review Sentiment"],
]

# fmt: off
BATCH_EXAMPLES = {
    "Retail reviews": (
        "I love my new laptop! The chip is incredibly fast and the battery lasts forever.\n"
        "The vacuum stopped working after just 2 months. Very disappointed with the quality.\n"
        "Decent headphones. Good sound quality but the ear cushions could be more comfortable.\n"
        "Best customer service experience! They replaced my defective item immediately.\n"
        "This furniture was a nightmare to assemble. Missing screws and confusing instructions."
    ),
    "Healthcare feedback": (
        "The doctors were exceptional. They explained my diagnosis thoroughly.\n"
        "Long wait times in the emergency room. Was in severe pain but had to wait 3 hours.\n"
        "Physical therapy sessions have been helpful but scheduling is difficult.\n"
        "Very satisfied with my telehealth consultation. Convenient and professional.\n"
        "Disappointed with the billing department. Received unexpected charges."
    ),
    "Financial services": (
        "Excellent mobile banking app. Easy to deposit checks and track spending.\n"
        "Hidden fees everywhere with this credit card. The annual fee wasn't clearly disclosed.\n"
        "My financial advisor helped me build a solid retirement portfolio.\n"
        "The loan approval process took forever. Submitted all documents but still waiting.\n"
        "Great interest rates on the savings account without a huge minimum balance."
    ),
}
# fmt: on

ABOUT = """
Scores are the model's probabilities over its own labels. They sum to 1 for one model but are not
comparable across models. Labels are normalised to **negative / neutral / positive**; binary SST-2
models have no neutral class, and the review model predicts **1 to 5 stars**.

Tips: try the same text with several models, and use the multilingual model for non-English text.
"""


def create_ui() -> gr.Blocks:
    """Build the sentiment page."""
    with shell.page(UI_KEY) as demo:
        with gr.Tab("Single text"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Text to analyze",
                        placeholder="Type or paste a review, comment or message...",
                        lines=5,
                    )
                    model_dropdown = MODELS.dropdown()
                    analyze_btn = gr.Button("Analyze sentiment", variant="primary")
                with gr.Column(scale=2):
                    output_json = gr.JSON(label="Result")
                    output_plot = gr.Plot(label="Probability per label")
            gr.Examples(examples=EXAMPLES, inputs=[text_input, model_dropdown], label="Try an example")

        with gr.Tab("Batch"):
            gr.Markdown(f"One text per line, up to {shell.MAX_BATCH_ITEMS} lines.")
            with gr.Row():
                with gr.Column(scale=2):
                    batch_input = gr.Textbox(
                        label="Texts (one per line)",
                        placeholder="First review\nSecond review\n...",
                        lines=10,
                    )
                    batch_model = MODELS.dropdown()
                    batch_btn = gr.Button("Analyze all", variant="primary")
                    with gr.Row():
                        for name, sample in BATCH_EXAMPLES.items():
                            gr.Button(f"Load: {name}", size="sm").click(lambda s=sample: s, outputs=batch_input)
                with gr.Column(scale=3):
                    batch_summary = gr.Markdown()
                    batch_output = gr.Dataframe(
                        headers=BATCH_COLUMNS,
                        label="Results",
                        wrap=True,
                        interactive=False,
                    )

        shell.about(ABOUT, MODELS)

        analyze_btn.click(analyze_sentiment, inputs=[text_input, model_dropdown], outputs=[output_json, output_plot])
        text_input.submit(analyze_sentiment, inputs=[text_input, model_dropdown], outputs=[output_json, output_plot])
        batch_btn.click(analyze_batch, inputs=[batch_input, batch_model], outputs=[batch_output, batch_summary])
    return demo


if __name__ == "__main__":
    shell.launch(create_ui(), UI_KEY, argv=sys.argv[1:])
