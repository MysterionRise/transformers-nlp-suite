"""
Catalog of the Gradio demo UIs.

Kept free of heavy imports (no gradio, torch or transformers) so the launcher can
list tools instantly. The UI modules, the combined app and launch_ui.py all read
titles, ports and descriptions from here instead of keeping their own copies.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class UIInfo:
    """Static description of one demo UI."""

    key: str
    title: str
    emoji: str
    module: str
    port: int
    description: str
    short_title: str = ""

    @property
    def tab_label(self) -> str:
        """Short name for the combined app's tab bar (falls back to the title)."""
        return self.short_title or self.title

    @property
    def file(self) -> str:
        """Path of the UI script relative to the repository root."""
        return f"ui/{self.module}.py"

    @property
    def import_path(self) -> str:
        """Dotted module path, for `python -m` and importlib."""
        return f"ui.{self.module}"


_ENTRIES = [
    UIInfo(
        "sentiment",
        "Sentiment Analysis",
        "🎭",
        "sentiment_playground",
        7860,
        "Classify text as positive, neutral or negative with general, financial and review models.",
        short_title="Sentiment",
    ),
    UIInfo(
        "similarity",
        "Semantic Similarity",
        "🔗",
        "similarity_explorer",
        7861,
        "Compare sentences with embeddings: pairwise scores, similarity matrices and 2-D projections.",
        short_title="Similarity",
    ),
    UIInfo(
        "ner",
        "Named Entity Recognition",
        "🏷️",
        "ner_visualizer",
        7862,
        "Extract people, organisations, places and custom entity types with spaCy or GLiNER.",
        short_title="Entities",
    ),
    UIInfo(
        "summarization",
        "Summarization",
        "📝",
        "summarization_studio",
        7863,
        "Generate abstractive summaries and compare models side by side.",
        short_title="Summaries",
    ),
    UIInfo(
        "performance",
        "Evaluation Metrics",
        "📊",
        "performance_dashboard",
        7864,
        "Score generated text against a reference with BLEU, ROUGE, METEOR and BERTScore.",
        short_title="Evaluation",
    ),
    UIInfo(
        "qa",
        "Question Answering",
        "❓",
        "qa_system",
        7865,
        "Extract the answer to a question from a passage of text.",
        short_title="Q&A",
    ),
    UIInfo(
        "generation",
        "Text Generation",
        "✍️",
        "generation_playground",
        7866,
        "Continue a prompt with GPT-2 family models and tune the sampling parameters.",
        short_title="Generation",
    ),
    UIInfo(
        "zero_shot",
        "Zero-Shot Classification",
        "🗂️",
        "zero_shot_classifier",
        7867,
        "Classify text into labels you define, without any training data.",
        short_title="Zero-shot",
    ),
    UIInfo(
        "translation",
        "Translation",
        "🌐",
        "translation_hub",
        7868,
        "Translate between languages with mBART-50 or language-pair OPUS-MT models.",
        short_title="Translation",
    ),
    UIInfo(
        "vision",
        "Vision-Language",
        "🖼️",
        "vision_language_explorer",
        7869,
        "Caption images and match images against text descriptions with CLIP.",
        short_title="Vision",
    ),
]

UIS: Dict[str, UIInfo] = {entry.key: entry for entry in _ENTRIES}
