"""
Evaluation Metrics - score generated text against a reference.

BLEU, ROUGE, METEOR and a bag-of-words cosine run locally with nltk and
rouge-score. Their NLTK data is checked (and downloaded if possible) on the
first evaluation, never at import. BERTScore is opt-in because its first use
downloads roberta-large (~1.4 GB).
"""

import json
import math
import re
import sys
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

try:
    from ui import common as shell
except ModuleNotFoundError:  # running as `python ui/performance_dashboard.py`
    import common as shell  # type: ignore[no-redef]

logger = shell.get_logger(__name__)

UI_KEY = "performance"

NLTK_INSTALL_HINT = "python -m nltk.downloader punkt_tab wordnet"
BERTSCORE_LABEL = "Also compute BERTScore (first use downloads roberta-large, about 1.4 GB; slow on CPU)"
BERTSCORE_MAX_TOKENS = 510  # roberta-large: 512 positions minus <s> and </s>
MAX_CANDIDATES = 20

METRICS = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "Cosine (bag of words)"]
BERTSCORE_METRICS = ["BERTScore P", "BERTScore R", "BERTScore F1"]
CHART_METRICS = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "BERTScore F1"]
_CANDIDATE_SEPARATOR = re.compile(r"(?m)^[ \t]*-{3,}[ \t]*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_nltk_ready: Dict[str, bool] = {}
_nltk_lock = threading.Lock()
_scorer_lock = threading.Lock()
_bert_scorer: Optional[Any] = None


# --------------------------------------------------------------------------- NLTK data


def _download_nltk(package: str) -> bool:
    """Try to download one NLTK package; False when offline or refused (for example behind a proxy)."""
    import nltk

    try:
        return bool(nltk.download(package, quiet=True, raise_on_error=False))
    except Exception as exc:  # network and permission errors vary by platform
        logger.warning(f"NLTK download of {package} failed: {exc}")
        return False


def _check_punkt() -> None:
    import nltk

    nltk.data.find("tokenizers/punkt_tab/english/")


def _check_wordnet() -> None:
    from nltk.corpus import wordnet

    wordnet.synsets("test")  # loads the corpus (folder or zip) the way METEOR does


# NLTK package -> check that raises LookupError when the data is missing.
NLTK_CHECKS: Dict[str, Callable[[], None]] = {"punkt_tab": _check_punkt, "wordnet": _check_wordnet}


def _nltk_installed(package: str) -> bool:
    try:
        NLTK_CHECKS[package]()
        return True
    except LookupError:
        return False


def nltk_data_available(package: str) -> bool:
    """Whether NLTK data `package` is installed, downloading it on first use. Cached per process."""
    with _nltk_lock:
        if package not in _nltk_ready:
            found = _nltk_installed(package)
            if not found:
                logger.info(f"NLTK data '{package}' not found; trying to download it")
                found = _download_nltk(package) and _nltk_installed(package)
            _nltk_ready[package] = found
        return _nltk_ready[package]


# --------------------------------------------------------------------------- metrics


@dataclass
class MetricTools:
    """Metric callables prepared once per evaluation event."""

    tokenize: Callable[[str], List[str]]
    bleu: Callable[[List[str], List[str]], float]
    rouge: Any
    meteor: Optional[Callable[[List[str], List[str]], float]]


def _tokenizer(nltk: Any) -> Callable[[str], List[str]]:
    if nltk_data_available("punkt_tab"):
        return nltk.word_tokenize
    shell.warn(
        "NLTK 'punkt_tab' data is not installed and could not be downloaded, so a simpler sentence splitter "
        f"is used for BLEU and METEOR. Install it with: {NLTK_INSTALL_HINT}"
    )

    def tokenize(text: str) -> List[str]:
        sentences = _SENTENCE_END.split(text)
        return [token for sentence in sentences for token in nltk.word_tokenize(sentence, preserve_line=True)]

    return tokenize


def _bleu(nltk_bleu: Any) -> Callable[[List[str], List[str]], float]:
    smoothing = nltk_bleu.SmoothingFunction().method1

    def bleu(reference: List[str], candidate: List[str]) -> float:
        order = max(1, min(4, len(reference), len(candidate)))  # very short texts use fewer n-gram orders
        weights = tuple([1.0 / order] * order)
        return float(nltk_bleu.sentence_bleu([reference], candidate, weights=weights, smoothing_function=smoothing))

    return bleu


def prepare_metrics() -> MetricTools:
    """Import the metric libraries and check their NLTK data; warn about anything that has to be skipped."""
    try:
        import nltk
        from nltk.translate import bleu_score
        from rouge_score import rouge_scorer
    except ImportError as exc:
        shell.fail(f"Evaluation needs the nltk and rouge-score packages ({exc}). Install requirements.txt.")

    meteor = None
    if nltk_data_available("wordnet"):
        from nltk.translate.meteor_score import single_meteor_score

        meteor = single_meteor_score
    else:
        shell.warn(
            "METEOR was skipped: NLTK 'wordnet' data is not installed and could not be downloaded. "
            f"Install it with: {NLTK_INSTALL_HINT}"
        )
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return MetricTools(tokenize=_tokenizer(nltk), bleu=_bleu(bleu_score), rouge=rouge, meteor=meteor)


def _cosine(reference: List[str], candidate: List[str]) -> float:
    a = Counter(token.lower() for token in reference if token.isalnum())
    b = Counter(token.lower() for token in candidate if token.isalnum())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return sum(a[token] * b[token] for token in a) / norm if norm else 0.0


def _lexical_scores(reference: str, candidate: str, tools: MetricTools) -> Dict[str, float]:
    reference_tokens, candidate_tokens = tools.tokenize(reference), tools.tokenize(candidate)
    rouge = tools.rouge.score(reference, candidate)
    scores = {
        "BLEU": tools.bleu(reference_tokens, candidate_tokens),
        "ROUGE-1": rouge["rouge1"].fmeasure,
        "ROUGE-2": rouge["rouge2"].fmeasure,
        "ROUGE-L": rouge["rougeL"].fmeasure,
    }
    if tools.meteor is not None:
        scores["METEOR"] = float(tools.meteor(reference_tokens, candidate_tokens))
    scores["Cosine (bag of words)"] = _cosine(reference_tokens, candidate_tokens)
    return {name: round(float(value), 4) for name, value in scores.items()}


def _get_bert_scorer() -> Any:
    """roberta-large BERTScorer, loaded on the first BERTScore request and kept for the process."""
    global _bert_scorer
    with _scorer_lock:
        if _bert_scorer is None:
            try:
                from bert_score import BERTScorer
            except ImportError:
                shell.fail("BERTScore needs the bert-score package. Install requirements.txt.")
            shell.info("Loading BERTScore. Its first use downloads roberta-large (about 1.4 GB).")
            try:
                _bert_scorer = BERTScorer(lang="en", use_fast_tokenizer=True)
            except Exception as exc:
                shell.fail_from(exc, "Loading BERTScore (roberta-large)")
        return _bert_scorer


def bertscore(references: Sequence[str], candidates: Sequence[str]) -> List[Dict[str, float]]:
    """BERTScore precision, recall and F1 for each (reference, candidate) pair."""
    scorer = _get_bert_scorer()
    tokenizer = getattr(scorer, "_tokenizer", None)
    refs = [shell.clip_to_tokens(text, tokenizer, BERTSCORE_MAX_TOKENS, label="Reference") for text in references]
    cands = [shell.clip_to_tokens(text, tokenizer, BERTSCORE_MAX_TOKENS, label="Candidate") for text in candidates]
    try:
        precision, recall, f1 = scorer.score(cands, refs)
    except Exception as exc:
        shell.fail_from(exc, "BERTScore")
    return [
        {"BERTScore P": round(float(p), 4), "BERTScore R": round(float(r), 4), "BERTScore F1": round(float(f), 4)}
        for p, r, f in zip(precision.tolist(), recall.tolist(), f1.tolist())
    ]


def evaluate_text(reference: str, candidate: str, use_bertscore: bool = False) -> Dict[str, float]:
    """Score one candidate against a reference. BERTScore only when `use_bertscore` is set."""
    reference = shell.require_text(reference, "a reference text")
    candidate = shell.require_text(candidate, "a candidate text")
    tools = prepare_metrics()
    try:
        scores = _lexical_scores(reference, candidate, tools)
    except Exception as exc:
        shell.fail_from(exc, "Evaluation")
    if use_bertscore:
        scores.update(bertscore([reference], [candidate])[0])
    return scores


# --------------------------------------------------------------------------- charts


def _chart_metrics(results: List[Dict[str, Any]]) -> List[str]:
    return [metric for metric in CHART_METRICS if all(metric in result for result in results)]


def score_bars(scores: Dict[str, float]) -> go.Figure:
    names = list(reversed(scores))
    fig = go.Figure(
        go.Bar(
            x=[scores[name] for name in names],
            y=names,
            orientation="h",
            text=[f"{scores[name]:.3f}" for name in names],
            textposition="auto",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1], title="Score"), height=80 + 40 * len(names), margin=dict(l=10, r=10, t=10, b=40)
    )
    return fig


def create_comparison_chart(results: List[Dict[str, Any]]) -> go.Figure:
    """Radar chart of the metrics every candidate has."""
    metrics = _chart_metrics(results)
    fig = go.Figure()
    for result in results:
        values = [result[metric] for metric in metrics]
        fig.add_trace(go.Scatterpolar(r=values + values[:1], theta=metrics + metrics[:1], name=result["candidate"]))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=450)
    return fig


def create_metric_bars(results: List[Dict[str, Any]]) -> go.Figure:
    """Grouped bar chart: one group per candidate, one bar per metric."""
    fig = go.Figure()
    names = [result["candidate"] for result in results]
    for metric in _chart_metrics(results):
        fig.add_trace(go.Bar(name=metric, x=names, y=[result[metric] for result in results]))
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1], title="Score"), height=400)
    return fig


# --------------------------------------------------------------------------- handlers


def evaluate_single(reference: str, candidate: str, use_bertscore: bool) -> Tuple[Dict[str, float], go.Figure]:
    """Single comparison tab: scores and a bar chart."""
    scores = evaluate_text(reference, candidate, use_bertscore)
    return scores, score_bars(scores)


def split_candidates(text: Optional[str]) -> List[str]:
    """Candidates separated by lines containing only ``---``, capped at MAX_CANDIDATES with a warning."""
    marked = _CANDIDATE_SEPARATOR.sub("\x1e", (text or "").replace("\r\n", "\n"))
    return shell.split_items(marked, "\x1e", max_items=MAX_CANDIDATES, what="one candidate text")


def records_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Results table for a list of score records."""
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["text"] = frame["text"].map(lambda text: text if len(text) <= 100 else text[:97] + "...")
    metrics = [name for name in METRICS + BERTSCORE_METRICS if name in frame.columns]
    return frame[["candidate", "text", *metrics]]


def batch_evaluate(
    reference: str, candidates_text: str, use_bertscore: bool = False
) -> Tuple[pd.DataFrame, go.Figure, go.Figure, List[Dict[str, Any]]]:
    """Score several candidates; returns the table, radar chart, bar chart and records for export."""
    reference = shell.require_text(reference, "a reference text")
    candidates = split_candidates(candidates_text)
    tools = prepare_metrics()
    records: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates, 1):
        try:
            scores = _lexical_scores(reference, candidate, tools)
        except Exception as exc:
            shell.fail_from(exc, f"Evaluating candidate {index}")
        records.append({"candidate": f"Candidate {index}", "text": candidate, **scores})
    if use_bertscore:
        for record, extra in zip(records, bertscore([reference] * len(candidates), candidates)):
            record.update(extra)
    return records_frame(records), create_comparison_chart(records), create_metric_bars(records), records


def export_results(records: Optional[List[Dict[str, Any]]]) -> str:
    """JSON export of the latest batch comparison."""
    if not records:
        shell.fail("Run a batch comparison first; its results are exported here.")
    return json.dumps(records, indent=2, ensure_ascii=False)


EXAMPLE_REFERENCE = (
    "The investment strategy involves diversifying assets across various sectors to "
    "mitigate risks and maximize returns. This approach ensures a balanced portfolio "
    "that can withstand market volatility while providing steady growth opportunities."
)
EXAMPLE_CANDIDATES = [
    "The strategy for investments involves diversifying assets across multiple sectors "
    "to minimize risks and optimize returns. This method ensures a balanced portfolio "
    "that can handle market fluctuations while offering consistent growth opportunities.",
    "An investment approach focuses on spreading assets across different sectors to "
    "reduce risk and increase profits. This strategy creates a stable portfolio "
    "resistant to market changes and provides continuous growth potential.",
    "The financial plan includes asset diversification across various industries to "
    "lower risks and enhance returns. This ensures portfolio balance and resilience "
    "against market volatility with sustainable growth prospects.",
]

ABOUT = f"""
All scores run from 0 to 1; higher means the candidate is closer to the reference.

- **BLEU**: n-gram precision (up to 4-grams, smoothed) with a brevity penalty. Strict about exact wording.
- **ROUGE-1 / ROUGE-2 / ROUGE-L**: F1 of shared words, word pairs and the longest common subsequence
  (with stemming). The rouge-score tokenizer only keeps the letters a-z and digits, so use it for English.
- **METEOR**: word alignment that also accepts stems and WordNet synonyms. Needs NLTK's `wordnet` data and is
  skipped with a warning when that data is missing.
- **Cosine (bag of words)**: cosine similarity of word counts, ignoring order.
- **BERTScore** (opt-in): similarity of roberta-large contextual embeddings. Its first use downloads about
  1.4 GB. Values are not rescaled, so even unrelated English texts usually score above 0.8.

NLTK data (`punkt_tab`, `wordnet`) is checked on the first evaluation and downloaded when possible. Offline or
behind a proxy that NLTK refuses, install it once with `{NLTK_INSTALL_HINT}`.

**Batch:** put a line containing only `---` between candidates (up to {MAX_CANDIDATES}). The *Export* tab
holds the latest batch results as JSON.
"""


def create_ui() -> gr.Blocks:
    """Build the evaluation page."""
    with shell.page(UI_KEY) as demo:
        results_state = gr.State([])
        with gr.Tab("Single comparison"):
            with gr.Row():
                single_reference = gr.Textbox(label="Reference", lines=5, value=EXAMPLE_REFERENCE)
                single_candidate = gr.Textbox(label="Candidate", lines=5, value=EXAMPLE_CANDIDATES[0])
            single_bertscore = gr.Checkbox(label=BERTSCORE_LABEL, value=False)
            evaluate_btn = gr.Button("Evaluate", variant="primary")
            with gr.Row():
                scores_json = gr.JSON(label="Scores")
                scores_chart = gr.Plot(label="Scores")

        with gr.Tab("Batch comparison"):
            batch_reference = gr.Textbox(label="Reference", lines=4, value=EXAMPLE_REFERENCE)
            batch_candidates = gr.Textbox(
                label=f"Candidates: put a line containing only --- between them (up to {MAX_CANDIDATES})",
                lines=12,
                value="\n---\n".join(EXAMPLE_CANDIDATES),
                placeholder="First candidate\n---\nSecond candidate",
            )
            batch_bertscore = gr.Checkbox(label=BERTSCORE_LABEL, value=False)
            batch_btn = gr.Button("Compare all", variant="primary")
            batch_table = gr.Dataframe(label="Results", wrap=True, interactive=False)
            with gr.Row():
                radar_chart = gr.Plot(label="Radar chart")
                bar_chart = gr.Plot(label="Metrics by candidate")

        with gr.Tab("Export"):
            gr.Markdown("Results of the latest batch comparison. Run one in the *Batch comparison* tab first.")
            export_table = gr.Dataframe(label="Latest batch results", wrap=True, interactive=False)
            export_btn = gr.Button("Generate JSON export", variant="primary")
            export_output = gr.Code(label="JSON", language="json")

        shell.about(ABOUT)

        evaluate_btn.click(
            evaluate_single,
            inputs=[single_reference, single_candidate, single_bertscore],
            outputs=[scores_json, scores_chart],
        )
        batch_btn.click(
            batch_evaluate,
            inputs=[batch_reference, batch_candidates, batch_bertscore],
            outputs=[batch_table, radar_chart, bar_chart, results_state],
        ).success(records_frame, inputs=results_state, outputs=export_table)
        export_btn.click(export_results, inputs=results_state, outputs=export_output)
    return demo


if __name__ == "__main__":
    shell.launch(create_ui(), UI_KEY, argv=sys.argv[1:])
