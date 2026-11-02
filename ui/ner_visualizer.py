"""
Named Entity Recognition - find people, organisations, places and custom entity types.

Models come from the ``ner`` and ``medical_ner`` registry categories and load
through the shared model cache on first use (never while the page is built):

- spaCy pipelines tag the 18 OntoNotes entity types (``en_core_web_trf`` is
  offered only when that package is installed),
- GLiNER finds whatever entity types you name (zero-shot),
- BiomedNLP NER, a transformers token-classification model, tags diseases,
  medications, symptoms and other clinical entities.

Character offsets always index the analyzed text exactly
(``text[start:end] == entity text``), also when long input was clipped.
"""

import json
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

try:
    from ui import common as shell
except ModuleNotFoundError:  # running as `python ui/ner_visualizer.py`
    import common as shell  # type: ignore[no-redef]

logger = shell.get_logger(__name__)

UI_KEY = "ner"


def _uninstalled_spacy_models() -> List[str]:
    """Registry keys of optional spaCy pipelines whose package is not installed."""
    from spacy.util import is_package

    return [
        choice.key
        for choice in shell.ModelMenu("ner", frameworks=["spacy"])
        if choice.key != "spacy_sm" and not is_package(choice.model_id)
    ]


MODELS = shell.ModelMenu(
    "ner",
    "medical_ner",
    keys=["spacy_sm", "spacy_trf", "gliner_multi", "biomedical_ner"],
    exclude=_uninstalled_spacy_models(),
    default="spacy_sm",
)

# OntoNotes types tagged by the English spaCy pipelines.
SPACY_TYPES = [
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL",
]

DEFAULT_GLINER_LABELS = "person, organization, location, date, event, product"
MAX_GLINER_LABELS = 25
GLINER_MAX_WORDS = 384  # gliner_multi-v2.1 config.max_len, used when a model does not report its own
GLINER_FIRST_RUN = (
    "Loading {name}: the first GLiNER run downloads about 0.8 GB and can take a few minutes on CPU. "
    "Later runs reuse the cached model."
)
# GLiNER's own word splitter, used when a model object does not expose it.
_WORD_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")

GLINER_PRESETS = {
    "General": DEFAULT_GLINER_LABELS,
    "Retail: Products": "brand, product, price, feature, category, competitor",
    "Retail: Reviews": "product name, brand, feature, issue, sentiment phrase",
    "Retail: Customer service": "customer name, order number, product, issue type, resolution",
    "Publishing: News": "person, organization, location, date, event, quote",
    "Publishing: Books": "author, title, publisher, character, setting, genre",
    "Publishing: Academic": "author, institution, journal, methodology, finding, citation",
    "Medical: Clinical": "disease, symptom, medication, dosage, procedure, body part",
    "Medical: Research": "drug, gene, protein, cell type, organism, biomarker",
    "Medical: Adverse events": "medication, adverse effect, severity, onset time, outcome",
    "Finance: Reports": "company, executive, revenue, profit, stock price, fiscal period",
    "Finance: Transactions": "bank, account type, amount, currency, transaction type",
    "Legal: Contracts": "party, date, obligation, term, jurisdiction, signature",
    "Legal: Cases": "plaintiff, defendant, court, judge, ruling, statute",
}

ENTITY_COLUMNS = ["Text", "Type", "Start", "End", "Score"]

# Model names that have been loaded in this process (for the first-run GLiNER notice).
_loaded: Set[str] = set()

Entity = Dict[str, Any]


def _framework(model_name: Optional[str]) -> str:
    return MODELS.get(model_name).config.framework or "transformers"


# --------------------------------------------------------------------------- per-framework extraction


def _clip_chars(text: str, limit: Any, label: str = "Text") -> str:
    if isinstance(limit, int) and 0 < limit < len(text):
        shell.warn(f"{label} is {len(text)} characters long; only the first {limit} were analyzed (model limit).")
        return text[:limit]
    return text


def _spacy_entities(nlp: Any, text: str, entity_types: Sequence[str]) -> Tuple[str, List[Entity]]:
    text = _clip_chars(text, getattr(nlp, "max_length", None))
    wanted = set(entity_types or [])
    doc = nlp(text)
    entities = [
        {"text": text[ent.start_char : ent.end_char], "type": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
        if not wanted or ent.label_ in wanted
    ]
    return text, entities


class _WordTokenizer:
    """
    GLiNER's word splitter behind the slice of the tokenizer API the shell's
    clipping helpers use, so long text is cut on a word boundary with a warning
    instead of being truncated silently inside GLiNER.
    """

    def __init__(self, model: Any):
        processor = getattr(model, "data_processor", None)
        self._split = getattr(processor, "words_splitter", None) or self._regex_split

    @staticmethod
    def _regex_split(text: str) -> Iterator[Tuple[str, int, int]]:
        for match in _WORD_PATTERN.finditer(text):
            yield match.group(), match.start(), match.end()

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 0

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_offsets_mapping: bool = False,
        **_: Any,
    ) -> Dict[str, List]:
        spans = [(int(start), int(end)) for _, start, end in self._split(text)]
        if truncation and max_length is not None:
            spans = spans[:max_length]
        encoded: Dict[str, List] = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            encoded["offset_mapping"] = spans
        return encoded


def parse_labels(labels: Optional[str]) -> List[str]:
    """Comma- or newline-separated GLiNER labels, de-duplicated; the general set when empty."""
    parsed: List[str] = []
    seen: Set[str] = set()
    for label in re.split(r"[,\n]", labels or ""):
        label = label.strip()
        if label and label.lower() not in seen:
            seen.add(label.lower())
            parsed.append(label)
    if not parsed:
        return parse_labels(DEFAULT_GLINER_LABELS)
    if len(parsed) > MAX_GLINER_LABELS:
        shell.warn(f"Only the first {MAX_GLINER_LABELS} of {len(parsed)} labels were used.")
        parsed = parsed[:MAX_GLINER_LABELS]
    return parsed


def _gliner_entities(model: Any, text: str, labels: str, threshold: float) -> Tuple[str, List[Entity]]:
    max_words = getattr(getattr(model, "config", None), "max_len", None)
    if not isinstance(max_words, int) or max_words <= 0:
        max_words = GLINER_MAX_WORDS
    text = shell.clip_to_tokens(text, _WordTokenizer(model), max_words, label="Text")
    raw = model.predict_entities(text, parse_labels(labels), threshold=float(threshold))
    entities = []
    for item in raw:
        start, end = int(item["start"]), int(item["end"])
        entities.append(
            {
                "text": text[start:end],
                "type": item["label"],
                "start": start,
                "end": end,
                "score": round(float(item["score"]), 3),
            }
        )
    return text, entities


def _pipeline_entities(pipe: Any, text: str, threshold: float) -> Tuple[str, List[Entity]]:
    text = shell.clip_for_model(text, pipe, label="Text")
    raw = pipe(text, aggregation_strategy="simple")
    entities = []
    for item in raw:
        if item.get("start") is None or item.get("end") is None:
            logger.warning(f"Skipping entity without character offsets: {item}")
            continue
        start, end = int(item["start"]), int(item["end"])
        # Trim whitespace the tokenizer folded into the span so offsets match the visible text.
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        score = round(float(item["score"]), 3)
        if start == end or score < float(threshold):
            continue
        label = item.get("entity_group") or item.get("entity", "")
        entities.append({"text": text[start:end], "type": label, "start": start, "end": end, "score": score})
    return text, entities


# --------------------------------------------------------------------------- output builders


def render_entities(text: str, entities: Sequence[Entity]) -> str:
    """Highlight entities in the analyzed text; all text is HTML-escaped."""
    parts, last = [], 0
    for ent in sorted(entities, key=lambda e: (e["start"], -e["end"])):
        start, end = ent["start"], ent["end"]
        if start < last:  # overlaps a span already highlighted
            continue
        parts.append(shell.esc(text[last:start]))
        parts.append(
            f'<mark class="tns-mark">{shell.esc(text[start:end])}'
            f'<span class="tns-label">{shell.esc(ent["type"])}</span></mark>'
        )
        last = end
    parts.append(shell.esc(text[last:]))
    return f'<div class="tns-text">{"".join(parts)}</div>'


def _table(entities: Sequence[Entity], scored: bool) -> pd.DataFrame:
    columns = ENTITY_COLUMNS if scored else ENTITY_COLUMNS[:-1]
    rows = [[e["text"], e["type"], e["start"], e["end"]] + ([e["score"]] if scored else []) for e in entities]
    return pd.DataFrame(rows, columns=columns)


def _chart(entities: Sequence[Entity]) -> Optional[go.Figure]:
    if not entities:
        return None
    counts = Counter(e["type"] for e in entities).most_common()
    labels = [label for label, _ in counts]
    values = [count for _, count in counts]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            text=values,
            textposition="auto",
            marker=dict(color=values, colorscale="Viridis", showscale=False),
        )
    )
    fig.update_layout(
        xaxis_title="Entity type",
        yaxis_title="Count",
        height=320,
        margin=dict(l=10, r=10, t=10, b=40),
    )
    return fig


# --------------------------------------------------------------------------- handlers


def extract_entities(
    text: str,
    model_name: str,
    entity_types: Optional[Sequence[str]] = None,
    labels: str = "",
    threshold: float = 0.5,
) -> Tuple[str, pd.DataFrame, Optional[go.Figure], str]:
    """Find entities; returns highlighted HTML, an entity table, a per-type chart and a JSON export."""
    text = shell.require_text(text, "some text to analyze")
    choice = MODELS.get(model_name)
    framework = choice.config.framework or "transformers"

    if framework == "spacy":
        from spacy.util import is_package

        if not is_package(choice.model_id):
            shell.fail(
                f"The spaCy pipeline {choice.model_id} is not installed. "
                f"Install it with: python -m spacy download {choice.model_id}"
            )
    if framework == "gliner" and choice.name not in _loaded:
        shell.info(GLINER_FIRST_RUN.format(name=choice.name))

    model = MODELS.load(model_name)
    _loaded.add(choice.name)

    try:
        if framework == "spacy":
            analyzed, entities = _spacy_entities(model, text, entity_types or [])
        elif framework == "gliner":
            analyzed, entities = _gliner_entities(model, text, labels, threshold)
        else:
            analyzed, entities = _pipeline_entities(model, text, threshold)
    except Exception as exc:
        shell.fail_from(exc, "Entity extraction")

    if not entities:
        shell.info("No entities found. Try another model, other labels or a lower threshold.")
    scored = framework != "spacy"
    return (
        render_entities(analyzed, entities),
        _table(entities, scored),
        _chart(entities),
        json.dumps(entities, indent=2, ensure_ascii=False),
    )


def option_visibility(model_name: str) -> Tuple[Dict[str, Any], ...]:
    """Show the options that apply to the selected model's framework."""
    framework = _framework(model_name)
    return (
        gr.update(visible=framework == "spacy"),  # spaCy entity-type filter
        gr.update(visible=framework == "gliner"),  # GLiNER labels
        gr.update(visible=framework == "gliner"),  # GLiNER presets
        gr.update(visible=framework != "spacy"),  # confidence threshold
    )


def apply_preset(preset: Optional[str]) -> Any:
    """Labels of a GLiNER preset (leaves the labels unchanged when the preset is cleared)."""
    if preset in GLINER_PRESETS:
        return GLINER_PRESETS[preset]
    return gr.skip()


# --------------------------------------------------------------------------- page

SPACY_SM = MODELS.get("spacy_sm").name
GLINER = MODELS.get("gliner_multi").name

EXAMPLES = [
    [
        "West Germany is the colloquial English term used to indicate the Federal Republic of Germany "
        "between its formation on 23 May 1949 and the German reunification on 3 October 1990. "
        "The FRG's provisional capital was the city of Bonn.",
        SPACY_SM,
        [],
        DEFAULT_GLINER_LABELS,
        0.5,
    ],
    [
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 in "
        "Cupertino, California. The company's first product was the Apple I computer.",
        GLINER,
        [],
        "person, organization, location, date, product",
        0.5,
    ],
    [
        "I bought the Samsung Galaxy S24 Ultra from Best Buy for $1,199. The camera quality is "
        "excellent but battery life could be better. Compared to the iPhone 15 Pro Max, I think "
        "Samsung offers better value.",
        GLINER,
        [],
        GLINER_PRESETS["Retail: Products"],
        0.5,
    ],
    [
        "Patient presents with type 2 diabetes mellitus and hypertension. Currently taking "
        "Metformin 1000mg twice daily and Lisinopril 20mg once daily. Reports occasional dizziness "
        "and fatigue.",
        GLINER,
        [],
        GLINER_PRESETS["Medical: Clinical"],
        0.4,
    ],
    [
        "Goldman Sachs reported Q4 2023 revenue of $11.3 billion, exceeding analyst expectations. "
        "CEO David Solomon highlighted strong performance in investment banking.",
        SPACY_SM,
        ["ORG", "PERSON", "MONEY", "DATE"],
        DEFAULT_GLINER_LABELS,
        0.5,
    ],
]

ABOUT = """
**spaCy** tags a fixed set of 18 entity types and is fast. **GLiNER** finds any entity types you
name in the labels box (zero-shot); lower the threshold to see more, less certain entities.
**BiomedNLP NER** tags clinical entities such as diseases, medications, symptoms and dosages.

| spaCy type | Meaning |
| --- | --- |
| PERSON | People, including fictional |
| NORP | Nationalities, religious or political groups |
| FAC | Buildings, airports, highways, bridges |
| ORG | Companies, agencies, institutions |
| GPE | Countries, cities, states |
| LOC | Non-GPE locations, mountain ranges, bodies of water |
| PRODUCT | Objects, vehicles, foods (not services) |
| EVENT | Named hurricanes, battles, wars, sports events |
| WORK_OF_ART | Titles of books, songs, films |
| LAW | Named documents made into laws |
| LANGUAGE | Any named language |
| DATE, TIME | Dates, periods and times of day |
| PERCENT, MONEY, QUANTITY | Percentages, monetary values, measurements |
| ORDINAL, CARDINAL | "first", "second"; other numerals |

Start and end in the table are character offsets into the analyzed text.
"""


def create_ui() -> gr.Blocks:
    """Build the NER page."""
    default_framework = _framework(MODELS.default)
    with shell.page(UI_KEY) as demo:
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text to analyze",
                    placeholder="Paste a news story, review, report or clinical note...",
                    lines=8,
                )
                model_dropdown = MODELS.dropdown()
                extract_btn = gr.Button("Extract entities", variant="primary")
                with gr.Accordion("Entity types, labels and threshold", open=False):
                    entity_filter = gr.CheckboxGroup(
                        choices=SPACY_TYPES,
                        value=[],
                        label="Show only these types (spaCy)",
                        info="Leave empty to show every type.",
                        visible=default_framework == "spacy",
                    )
                    labels_input = gr.Textbox(
                        label="Entity labels (GLiNER)",
                        value=DEFAULT_GLINER_LABELS,
                        info="Comma-separated; any entity types you want GLiNER to find.",
                        visible=default_framework == "gliner",
                    )
                    preset_dropdown = gr.Dropdown(
                        choices=list(GLINER_PRESETS),
                        value=None,
                        label="Label preset (GLiNER)",
                        info="Fills the labels box with an industry-specific set.",
                        visible=default_framework == "gliner",
                    )
                    threshold_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.95,
                        value=0.5,
                        step=0.05,
                        label="Confidence threshold",
                        visible=default_framework != "spacy",
                    )
            with gr.Column(scale=4):
                html_output = gr.HTML(label="Entities in context")
                stats_plot = gr.Plot(label="Entities per type")

        with gr.Tab("Entity table"):
            entities_table = gr.Dataframe(headers=ENTITY_COLUMNS, label="Entities", wrap=True, interactive=False)
        with gr.Tab("JSON"):
            json_output = gr.Code(label="Entities as JSON", language="json", lines=12)

        inputs = [text_input, model_dropdown, entity_filter, labels_input, threshold_slider]
        gr.Examples(examples=EXAMPLES, inputs=inputs, label="Try an example")
        shell.about(ABOUT, MODELS)

        outputs = [html_output, entities_table, stats_plot, json_output]
        extract_btn.click(extract_entities, inputs=inputs, outputs=outputs)
        model_dropdown.change(
            option_visibility,
            inputs=model_dropdown,
            outputs=[entity_filter, labels_input, preset_dropdown, threshold_slider],
            api_visibility="undocumented",
        )
        preset_dropdown.input(apply_preset, inputs=preset_dropdown, outputs=labels_input, api_visibility="undocumented")
    return demo


if __name__ == "__main__":
    shell.launch(create_ui(), UI_KEY, argv=sys.argv[1:])
