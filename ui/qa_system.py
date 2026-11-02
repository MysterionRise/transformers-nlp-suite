"""
Question Answering - extract the answer to a question from a passage of text.

Models come from the ``question_answering`` registry category and load through the
shared model cache. The question-answering model reads long passages in
overlapping windows, so the whole passage is searched (up to MAX_CONTEXT_CHARS
characters); only the question is clipped to a token budget.
"""

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

try:
    from ui import common as shell
except ModuleNotFoundError:  # running as `python ui/qa_system.py`
    import common as shell  # type: ignore[no-redef]

logger = shell.get_logger(__name__)

UI_KEY = "qa"
MODELS = shell.ModelMenu("question_answering", default="distilbert_squad")

MAX_CONTEXT_CHARS = 20_000
MAX_QUESTION_TOKENS = 64  # leaves room for the context in every 384-token window
MAX_ANSWER_TOKENS = 30
CONFIDENCE_THRESHOLD = 0.10
# Models trained on SQuAD 2.0 can say "the passage does not answer this".
NO_ANSWER_KEYS = {"roberta_squad"}
EXCERPT_CHARS = 600  # passage shown on each side of the answer for long contexts
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "qa_contexts.txt"

ANSWER, LOW_CONFIDENCE, NO_ANSWER = "answer", "low_confidence", "no_answer"
STATUS_TEXT = {
    ANSWER: "Answer found",
    LOW_CONFIDENCE: "No confident answer found",
    NO_ANSWER: "No answer in the context",
}
BATCH_STATUS = {ANSWER: "answer", LOW_CONFIDENCE: "low confidence", NO_ANSWER: "no answer"}
BATCH_COLUMNS = ["#", "Question", "Answer", "Confidence", "Status", "Expected"]

_LABELLED_LINE = re.compile(r"^\s*(context|question|answer)\s*\d*\s*:\s*(.*)$", re.IGNORECASE)
_QUESTION_LINE = re.compile(r"^\s*question\s*\d*\s*:", re.IGNORECASE | re.MULTILINE)
_SEPARATOR_LINE = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)


@dataclass
class Found:
    """Normalised pipeline result: the verdict and the best span (empty when there is none)."""

    status: str
    text: str = ""
    score: float = 0.0
    start: int = 0
    end: int = 0


@dataclass
class QAItem:
    """One batch question with its context; `problem` is set when it cannot be asked."""

    question: str
    context: str
    expected: str = ""
    problem: str = ""


# --------------------------------------------------------------------------- inference


def _allows_no_answer(choice: shell.ModelChoice) -> bool:
    return choice.key in NO_ANSWER_KEYS or "squad2" in choice.model_id.lower()


def _cap_context(context: str) -> Tuple[str, bool]:
    """Cut very long passages at MAX_CONTEXT_CHARS, on whitespace when possible."""
    if len(context) <= MAX_CONTEXT_CHARS:
        return context, False
    cut = max(context.rfind(ch, 0, MAX_CONTEXT_CHARS) for ch in " \n\t")
    if cut < MAX_CONTEXT_CHARS // 2:
        cut = MAX_CONTEXT_CHARS
    return context[:cut], True


def _ask(qa: Any, choice: shell.ModelChoice, question: str, context: str) -> Found:
    """Run the model and classify the result as answer, low confidence or no answer."""
    kwargs: Dict[str, Any] = {"top_k": 1, "max_answer_len": MAX_ANSWER_TOKENS}
    if _allows_no_answer(choice):
        # top_k=2 keeps the best real span next to the "no answer" candidate.
        kwargs.update(handle_impossible_answer=True, top_k=2)
    raw = qa({"question": question, "context": context}, **kwargs)
    answers = raw if isinstance(raw, list) else [raw]
    spans = [a for a in answers if str(a.get("answer", "")).strip()]

    if not answers or not str(answers[0].get("answer", "")).strip():
        status = NO_ANSWER
    elif float(answers[0]["score"]) < CONFIDENCE_THRESHOLD:
        status = LOW_CONFIDENCE
    else:
        status = ANSWER
    if not spans:
        return Found(status)
    best = spans[0]
    # Spans found in several overlapping windows have their scores summed; keep it a probability.
    score = min(max(float(best["score"]), 0.0), 1.0)
    return Found(status, str(best["answer"]).strip(), score, int(best["start"]), int(best["end"]))


# --------------------------------------------------------------------------- display


def _snap_start(text: str, index: int) -> int:
    """Move an excerpt start forward to a word boundary (at most 40 characters)."""
    if index <= 0:
        return 0
    space = text.find(" ", index, index + 40)
    return space + 1 if space != -1 else index


def _snap_end(text: str, index: int) -> int:
    """Move an excerpt end back to a word boundary (at most 40 characters)."""
    if index >= len(text):
        return len(text)
    space = text.rfind(" ", max(0, index - 40), index)
    return space if space > 0 else index


def highlight_answer(context: str, start: int, end: int, label: str = "answer") -> str:
    """
    The context as escaped HTML with [start, end) marked. Long contexts show only
    the passage around the span; without a span, the start of the context.
    """
    has_span = 0 <= start < end <= len(context)
    if not has_span:
        start = end = 0
    lo = _snap_start(context, start - EXCERPT_CHARS) if has_span else 0
    hi = _snap_end(context, end + EXCERPT_CHARS if has_span else 2 * EXCERPT_CHARS)

    parts = ["… " if lo > 0 else "", shell.esc(context[lo:start])]
    if has_span:
        parts.append(
            f'<mark class="tns-mark">{shell.esc(context[start:end])}'
            f'<span class="tns-label">{shell.esc(label)}</span></mark>'
        )
    parts.append(shell.esc(context[end:hi]))
    parts.append(" …" if hi < len(context) else "")

    note = ""
    if lo > 0 or hi < len(context):
        note = f"<p>Showing characters {lo:,} to {hi:,} of {len(context):,}.</p>"
    if not has_span:
        note += "<p>No answer span to highlight.</p>"
    return f'{note}<div class="tns-text">{"".join(parts)}</div>'


def _gauge(score: float, title: str) -> go.Figure:
    threshold = CONFIDENCE_THRESHOLD * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score * 100, 1),
            number={"suffix": "%"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4c78a8"},
                "steps": [
                    {"range": [0, threshold], "color": "rgba(209, 73, 91, 0.35)"},
                    {"range": [threshold, 50], "color": "rgba(238, 137, 89, 0.25)"},
                    {"range": [50, 100], "color": "rgba(42, 157, 143, 0.25)"},
                ],
                "threshold": {"line": {"color": "#d1495b", "width": 3}, "thickness": 0.75, "value": threshold},
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=60, b=10))
    return fig


# --------------------------------------------------------------------------- handlers


def answer_question(context: str, question: str, model_name: str) -> Tuple[Dict[str, Any], str, go.Figure]:
    """Answer one question; returns a result summary, the highlighted context and a confidence gauge."""
    context = shell.require_text(context, "a context passage to search")
    question = shell.require_text(question, "a question")
    choice = MODELS.get(model_name)
    used_context, context_cut = _cap_context(context)
    if context_cut:
        shell.warn(
            f"The context is {len(context):,} characters long; only the first {len(used_context):,} were searched."
        )
    qa = MODELS.load(model_name)
    used_question = shell.clip_to_tokens(question, qa.tokenizer, MAX_QUESTION_TOKENS, label="Question")
    try:
        found = _ask(qa, choice, used_question, used_context)
    except Exception as exc:
        shell.fail_from(exc, "Question answering")

    result: Dict[str, Any] = {"status": found.status, "verdict": STATUS_TEXT[found.status]}
    if found.status == ANSWER:
        result["answer"] = found.text
        result["confidence"] = f"{found.score:.1%}"
        label, title = "answer", "Answer confidence"
    else:
        if found.status == LOW_CONFIDENCE:
            result["explanation"] = (
                f"The best span scored below {CONFIDENCE_THRESHOLD:.0%}, so the passage probably does not "
                "answer the question. It is shown below as a low-confidence guess."
            )
        else:
            result["explanation"] = "The model judged that this passage does not answer the question." + (
                " The best span it considered is shown as a low-confidence guess." if found.text else ""
            )
        result["best_guess"] = found.text or None
        result["confidence"] = f"{found.score:.1%} (low)" if found.text else None
        label, title = "low confidence", "Best guess confidence"
    result.update(
        {
            "start": found.start if found.text else None,
            "end": found.end if found.text else None,
            "model": choice.model_id,
            "context_truncated": context_cut,
            "question_truncated": used_question != question,
        }
    )
    html = highlight_answer(used_context, found.start, found.end, label)  # no span: plain context
    return result, html, _gauge(found.score, title if found.text else "No answer")


def _parse_labelled(text: str) -> List[QAItem]:
    """'Context N: ...' / 'Question: ...' / optional 'Answer: ...' blocks (data/qa_contexts.txt)."""
    items: List[QAItem] = []
    context: List[str] = []
    field: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _SEPARATOR_LINE.match(line):
            field = None
            continue
        match = _LABELLED_LINE.match(line)
        if match:
            kind, value = match.group(1).lower(), match.group(2).strip()
            if kind == "context":
                context, field = ([value] if value else []), "context"
            elif kind == "question":
                items.append(QAItem(question=value, context="\n".join(context)))
                field = "question"
            elif items:
                items[-1].expected = value
                field = "answer"
            continue
        if field == "context":
            context.append(line)
        else:  # an unlabelled line after a question or answer starts the next context
            context, field = [line], "context"
    return items


def _parse_separated(text: str) -> List[QAItem]:
    """Blocks separated by '---' lines: the context, then the question on the last line."""
    items = []
    for block in _SEPARATOR_LINE.split(text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) < 2:
            items.append(QAItem(question=lines[0], context="", problem="needs a context line and a question line"))
        else:
            items.append(QAItem(question=lines[-1], context="\n".join(lines[:-1])))
    return items


def parse_batch(text: str) -> List[QAItem]:
    """Parse either batch format into questions with their contexts."""
    items = _parse_labelled(text) if _QUESTION_LINE.search(text) else _parse_separated(text)
    for item in items:
        if not item.problem and not item.context.strip():
            item.problem = "no context before this question"
        elif not item.problem and not item.question.strip():
            item.problem = "empty question"
    return items


def batch_answer(batch_text: str, model_name: str) -> Tuple[pd.DataFrame, str]:
    """Answer every question in the batch; a failing question is reported in its row."""
    text = shell.require_text(batch_text, "contexts and questions (or load the sample)")
    items = parse_batch(text)
    if not items:
        shell.fail("No questions found. Use one of the two formats described above the input box.")
    if len(items) > shell.MAX_BATCH_ITEMS:
        shell.warn(f"Only the first {shell.MAX_BATCH_ITEMS} of {len(items)} questions were processed.")
        items = items[: shell.MAX_BATCH_ITEMS]
    choice = MODELS.get(model_name)
    qa = MODELS.load(model_name)

    rows, counts, long_contexts, long_questions = [], Counter(), 0, 0
    for index, item in enumerate(items, 1):
        if item.problem:
            counts["error"] += 1
            rows.append([index, item.question, "", "", f"error: {item.problem}", item.expected])
            continue
        context, cut = _cap_context(item.context)
        question, _ = shell.truncate_to_tokens(item.question, qa.tokenizer, MAX_QUESTION_TOKENS)
        long_contexts += cut
        long_questions += question != item.question
        try:
            found = _ask(qa, choice, question, context)
        except Exception as exc:
            logger.warning(f"Batch question {index} failed: {exc}")
            counts["error"] += 1
            rows.append([index, item.question, "", "", f"error: {str(exc)[:120]}", item.expected])
            continue
        status = BATCH_STATUS[found.status]
        counts[status] += 1
        confidence = f"{found.score:.1%}" if found.text else ""
        rows.append([index, item.question, found.text, confidence, status, item.expected])

    if long_contexts:
        shell.warn(f"{long_contexts} context(s) were longer than {MAX_CONTEXT_CHARS:,} characters and were cut.")
    if long_questions:
        shell.warn(f"{long_questions} question(s) were longer than {MAX_QUESTION_TOKENS} tokens and were cut.")
    summary = " · ".join(f"**{status}**: {count}" for status, count in counts.most_common())
    return pd.DataFrame(rows, columns=BATCH_COLUMNS), f"{len(items)} questions answered with {choice.name}. {summary}"


def load_sample() -> str:
    """Contents of data/qa_contexts.txt for the batch tab."""
    try:
        return SAMPLE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        shell.fail_from(exc, "Loading the sample file")


# Short texts so the examples table stays readable on narrow screens.
EXAMPLES = [
    ["The Eiffel Tower in Paris was built in 1889 and is 330 metres tall.", "When was the Eiffel Tower built?"],
    ["Python was created by Guido van Rossum and first released in 1991.", "Who created Python?"],
    ["The Great Wall of China stretches over 13,000 miles.", "How long is the Great Wall?"],
    ["Marie Curie won Nobel Prizes in physics and in chemistry.", "What is the capital of Australia?"],
]

BATCH_HELP = f"""
Two formats are accepted (up to {shell.MAX_BATCH_ITEMS} questions):

- **Labelled** (as in the sample): `Context: ...` followed by one or more `Question: ...` lines, each optionally
  followed by `Answer: ...` (the expected answer, shown next to the model's).
- **Separated**: blocks separated by `---` lines; each block is the context followed by the question on its last line.
"""

ABOUT = f"""
The model **extracts** the answer: it picks the span of the passage most likely to answer the question, so it
cannot answer from outside knowledge. Long passages (up to {MAX_CONTEXT_CHARS:,} characters) are read in
overlapping windows, so the answer can be anywhere in the text; questions are limited to {MAX_QUESTION_TOKENS} tokens.

**Confidence** is the model's probability for the span. Below {CONFIDENCE_THRESHOLD:.0%} the tool reports
*no confident answer* and shows the span only as a low-confidence guess, because a weak span usually means the
passage does not contain the answer.

**RoBERTa SQuAD** was trained on SQuAD 2.0, which includes unanswerable questions, so it can say *no answer in
the context* outright. The other models always return some span; rely on the confidence for them.
"""


def create_ui() -> gr.Blocks:
    """Build the question answering page."""
    with shell.page(UI_KEY) as demo:
        with gr.Tab("Ask a question"):
            with gr.Row():
                with gr.Column(scale=3):
                    context_input = gr.Textbox(
                        label="Context",
                        placeholder="Paste the passage the answer should come from...",
                        lines=8,
                        max_lines=20,
                    )
                with gr.Column(scale=2):
                    question_input = gr.Textbox(label="Question", placeholder="What would you like to know?", lines=2)
                    model_dropdown = MODELS.dropdown()
                    answer_btn = gr.Button("Get answer", variant="primary")
            with gr.Row():
                with gr.Column(scale=3):
                    output_highlight = gr.HTML(label="Answer in context")
                with gr.Column(scale=2):
                    output_json = gr.JSON(label="Result")
                    output_gauge = gr.Plot(label="Confidence")
            gr.Examples(examples=EXAMPLES, inputs=[context_input, question_input], label="Try an example")

        with gr.Tab("Batch"):
            gr.Markdown(BATCH_HELP)
            with gr.Row():
                with gr.Column(scale=2):
                    batch_input = gr.Textbox(
                        label="Contexts and questions",
                        placeholder="Context: The Eiffel Tower is in Paris.\nQuestion: Where is the Eiffel Tower?",
                        lines=14,
                        max_lines=30,
                    )
                    batch_model = MODELS.dropdown()
                    with gr.Row():
                        batch_btn = gr.Button("Answer all", variant="primary")
                        sample_btn = gr.Button("Load sample", size="sm")
                with gr.Column(scale=3):
                    batch_summary = gr.Markdown()
                    batch_output = gr.Dataframe(headers=BATCH_COLUMNS, label="Results", wrap=True, interactive=False)

        shell.about(ABOUT, MODELS)

        inputs = [context_input, question_input, model_dropdown]
        outputs = [output_json, output_highlight, output_gauge]
        answer_btn.click(answer_question, inputs=inputs, outputs=outputs)
        question_input.submit(answer_question, inputs=inputs, outputs=outputs, api_visibility="private")
        batch_btn.click(batch_answer, inputs=[batch_input, batch_model], outputs=[batch_output, batch_summary])
        sample_btn.click(load_sample, outputs=batch_input, api_visibility="private")
    return demo


if __name__ == "__main__":
    shell.launch(create_ui(), UI_KEY, argv=sys.argv[1:])
