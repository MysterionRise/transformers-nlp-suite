"""
Semantic Similarity - compare sentences with sentence-transformer embeddings.

Pairwise scores, semantic search, a similarity matrix and a 2-D map of the
embedding space (PCA, t-SNE or UMAP). Models come from the ``embeddings``
registry category and load through the shared model cache. Embeddings are
L2-normalised, so the dot product of two embeddings is their cosine similarity.
"""

import sys
from typing import List, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from ui import common as shell
except ModuleNotFoundError:  # running as `python ui/similarity_explorer.py`
    import common as shell  # type: ignore[no-redef]

logger = shell.get_logger(__name__)

UI_KEY = "similarity"
MODELS = shell.ModelMenu("embeddings", default="all_minilm_l6")

METHODS = ["PCA", "t-SNE", "UMAP"]
SEARCH_COLUMNS = ["Rank", "Score", "Sentence #", "Sentence"]
MIN_SENTENCES = 2
MIN_SENTENCES_UMAP = 3  # UMAP's neighbour graph is degenerate for two points
TOP_K_CHART = 10
MAX_MATRIX_LABELS = 15  # print scores inside heatmap cells up to this many sentences

# (lower bound, label, meaning), checked from the top.
BANDS = [
    (0.9, "Very high similarity", "nearly identical meaning"),
    (0.7, "High similarity", "closely related meaning"),
    (0.5, "Moderate similarity", "somewhat related"),
    (0.3, "Low similarity", "loosely related"),
    (-1.0, "Very low similarity", "different topics"),
]

_umap_used = False


# --------------------------------------------------------------------------- embeddings


def _token_budget(model) -> int:
    """
    Usable tokens per sentence. SentenceTransformer models cut input at
    ``max_seq_length`` (often below the tokenizer's limit), so take the smaller.
    """
    limit = shell.token_limit(model)
    max_seq_length = getattr(model, "max_seq_length", None)
    if isinstance(max_seq_length, int) and max_seq_length > 0:
        tokenizer = getattr(model, "tokenizer", None)
        special = tokenizer.num_special_tokens_to_add() if hasattr(tokenizer, "num_special_tokens_to_add") else 2
        limit = min(limit, max_seq_length - special)
    return max(1, limit)


def embed(sentences: List[str], model_name: str) -> np.ndarray:
    """L2-normalised embeddings, one row per sentence; long sentences are clipped with a warning."""
    model = MODELS.load(model_name)
    budget = _token_budget(model)
    tokenizer = getattr(model, "tokenizer", None)
    used, clipped = [], 0
    for sentence in sentences:
        cut, _ = shell.truncate_to_tokens(sentence, tokenizer, budget)
        clipped += cut != sentence
        used.append(cut)
    if clipped:
        shell.warn(
            f"{clipped} sentence(s) exceeded the model's {budget}-token limit; only their first {budget} were used."
        )
    try:
        vectors = model.encode(used, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    except Exception as exc:
        shell.fail_from(exc, "Computing embeddings")
    return np.asarray(vectors, dtype=float)


def _sentences(text: str, what: str) -> List[str]:
    """Non-empty lines (capped at MAX_BATCH_ITEMS); at least MIN_SENTENCES of them."""
    if sum(1 for line in (text or "").split("\n") if line.strip()) < MIN_SENTENCES:
        shell.fail(f"Please enter at least {MIN_SENTENCES} sentences, one per line, to {what}.")
    return shell.split_items(text, what="one sentence")


def _short(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def interpret(score: float) -> Tuple[str, str]:
    """(label, meaning) for a cosine similarity."""
    for lower, label, meaning in BANDS:
        if score >= lower:
            return label, meaning
    return BANDS[-1][1], BANDS[-1][2]


# --------------------------------------------------------------------------- handlers


def compare_sentences(text1: str, text2: str, model_name: str) -> Tuple[str, go.Figure]:
    """Cosine similarity of two sentences; returns an interpretation and a gauge."""
    first = shell.require_text(text1, "the first sentence")
    second = shell.require_text(text2, "the second sentence")
    vectors = embed([first, second], model_name)
    score = float(np.clip(vectors[0] @ vectors[1], -1.0, 1.0))
    label, meaning = interpret(score)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".3f"},
            gauge={
                "axis": {"range": [-1 if score < 0 else 0, 1]},
                "bar": {"color": "#3a86ff"},
                "steps": [
                    {"range": [0.0, 0.3], "color": "rgba(209, 73, 91, 0.35)"},
                    {"range": [0.3, 0.5], "color": "rgba(238, 137, 89, 0.35)"},
                    {"range": [0.5, 0.7], "color": "rgba(141, 153, 174, 0.35)"},
                    {"range": [0.7, 0.9], "color": "rgba(132, 181, 159, 0.35)"},
                    {"range": [0.9, 1.0], "color": "rgba(42, 157, 143, 0.45)"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=30, b=10))
    return f"{score:.3f} · {label}: {meaning}", fig


def semantic_search(sentences_text: str, query: str, model_name: str) -> Tuple[pd.DataFrame, go.Figure]:
    """Rank sentences by similarity to a query; returns a ranked table and a top-10 chart."""
    query = shell.require_text(query, "a query")
    sentences = shell.split_items(sentences_text, what="one sentence to search")
    vectors = embed(sentences + [query], model_name)
    scores = vectors[:-1] @ vectors[-1]
    order = [int(i) for i in np.argsort(-scores, kind="stable")]

    rows = [[rank, round(float(scores[i]), 4), i + 1, sentences[i]] for rank, i in enumerate(order, 1)]
    top = order[:TOP_K_CHART]
    fig = go.Figure(
        go.Bar(
            x=[float(scores[i]) for i in top],
            y=[f"#{i + 1}: {_short(sentences[i], 40)}" for i in top],
            orientation="h",
            text=[f"{scores[i]:.3f}" for i in top],
            textposition="auto",
            hovertext=[sentences[i] for i in top],
            marker=dict(color=[float(scores[i]) for i in top], colorscale="Viridis", cmin=0, cmax=1),
        )
    )
    fig.update_layout(
        xaxis=dict(title="Cosine similarity to the query"),
        yaxis=dict(autorange="reversed"),
        height=120 + 36 * len(top),
        margin=dict(l=10, r=10, t=10, b=40),
    )
    return pd.DataFrame(rows, columns=SEARCH_COLUMNS), fig


def similarity_matrix(text: str, model_name: str) -> go.Figure:
    """Heatmap of the cosine similarity between every pair of sentences."""
    sentences = _sentences(text, "build a similarity matrix")
    vectors = embed(sentences, model_name)
    matrix = np.clip(vectors @ vectors.T, -1.0, 1.0)
    labels = [f"S{i + 1}" for i in range(len(sentences))]
    hover = [[f"{_short(a)}<br>{_short(b)}" for b in sentences] for a in sentences]

    heatmap = dict(
        z=matrix,
        x=labels,
        y=labels,
        zmin=min(0.0, float(matrix.min())),
        zmax=1.0,
        colorscale="RdYlGn",
        customdata=hover,
        hovertemplate="%{customdata}<br>similarity %{z:.3f}<extra></extra>",
        colorbar=dict(title="Cosine"),
    )
    if len(sentences) <= MAX_MATRIX_LABELS:
        heatmap.update(text=np.round(matrix, 2), texttemplate="%{text}", textfont={"size": 11})
    fig = go.Figure(go.Heatmap(**heatmap))
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=min(800, 220 + 32 * len(sentences)),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def project(vectors: np.ndarray, method: str) -> np.ndarray:
    """2-D coordinates for the embeddings with PCA, t-SNE or UMAP."""
    n = len(vectors)
    if method == "PCA":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=42).fit_transform(vectors)
    if method == "t-SNE":
        from sklearn.manifold import TSNE

        # Perplexity must be below the number of samples; scale it down for small inputs.
        perplexity = min(30.0, max(1.0, (n - 1) / 3))
        return TSNE(n_components=2, perplexity=perplexity, random_state=42, init="random").fit_transform(vectors)
    if method == "UMAP":
        global _umap_used
        if n < MIN_SENTENCES_UMAP:
            shell.fail(f"UMAP needs at least {MIN_SENTENCES_UMAP} sentences; use PCA or t-SNE for {n}.")
        if not _umap_used:
            shell.info("UMAP is loading; its first run in a session takes 10 to 20 seconds.")
            _umap_used = True
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=max(2, min(15, n - 1)),
            metric="cosine",
            init="spectral" if n > 4 else "random",
            random_state=42,
            n_jobs=1,
        )
        return reducer.fit_transform(vectors)
    shell.fail(f"Unknown projection method {method!r}. Choose one of: {', '.join(METHODS)}.")


def visualize_embeddings(text: str, method: str, model_name: str) -> go.Figure:
    """Scatter plot of the sentences' embeddings projected to 2-D."""
    sentences = _sentences(text, "draw an embedding map")
    vectors = embed(sentences, model_name)
    try:
        coords = project(vectors, method)
    except Exception as exc:  # fail_from re-raises gr.Error unchanged
        shell.fail_from(exc, f"{method} projection")

    fig = go.Figure(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers+text",
            text=[str(i + 1) for i in range(len(sentences))],
            textposition="top center",
            hovertext=[f"{i + 1}: {_short(s, 80)}" for i, s in enumerate(sentences)],
            hoverinfo="text",
            marker=dict(size=11, color=list(range(len(sentences))), colorscale="Viridis"),
        )
    )
    fig.update_layout(
        xaxis_title=f"{method} 1",
        yaxis_title=f"{method} 2",
        height=500,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=40),
    )
    return fig


# --------------------------------------------------------------------------- page

EXAMPLE_PAIRS = [
    ["The cat sat on the mat.", "A feline rested on the rug."],
    ["Python is a programming language.", "Java is used for software development."],
    ["I love eating pizza.", "Pizza is my favorite food."],
    ["The weather is sunny today.", "It's raining heavily outside."],
    ["Machine learning is fascinating.", "AI and ML are interesting topics."],
]

EXAMPLE_SENTENCES = """The cat sat on the mat.
A kitten is sleeping on the sofa.
Dogs are loyal animals.
Python is a programming language.
JavaScript runs in the browser.
The weather is sunny today.
Heavy rain is expected tomorrow.
Coffee tastes great in the morning."""

ABOUT = """
Each sentence becomes a vector (an embedding) that captures its meaning; similar meanings get
nearby vectors. Scores are cosine similarities between -1 and 1:

| Score | Reading |
| --- | --- |
| 0.9 to 1.0 | Nearly identical meaning (paraphrases) |
| 0.7 to 0.9 | Closely related |
| 0.5 to 0.7 | Somewhat related |
| 0.3 to 0.5 | Loosely related |
| below 0.3 | Different topics |

Scores are comparable within one model, not across models. The embedding map projects the
vectors to 2-D: **PCA** keeps global structure, **t-SNE** and **UMAP** keep neighbourhoods (the
distances between clusters are not meaningful). Use the multilingual model for non-English text.
"""


def create_ui() -> gr.Blocks:
    """Build the similarity page."""
    with shell.page(UI_KEY) as demo:
        with gr.Tab("Compare two sentences"):
            with gr.Row():
                with gr.Column():
                    sent1_input = gr.Textbox(label="First sentence", placeholder="Enter a sentence...", lines=3)
                    sent2_input = gr.Textbox(label="Second sentence", placeholder="Enter another...", lines=3)
                    pair_model = MODELS.dropdown()
                    compare_btn = gr.Button("Compare", variant="primary")
                with gr.Column():
                    pair_result = gr.Textbox(label="Similarity", interactive=False)
                    pair_plot = gr.Plot(label="Cosine similarity")
            gr.Examples(examples=EXAMPLE_PAIRS, inputs=[sent1_input, sent2_input], label="Try an example")

        with gr.Tab("Semantic search"):
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(label="Query", placeholder="What are you looking for?", lines=2)
                    corpus_input = gr.Textbox(
                        label=f"Sentences to search (one per line, up to {shell.MAX_BATCH_ITEMS})",
                        value=EXAMPLE_SENTENCES,
                        lines=10,
                    )
                    search_model = MODELS.dropdown()
                    search_btn = gr.Button("Search", variant="primary")
                with gr.Column():
                    search_table = gr.Dataframe(
                        headers=SEARCH_COLUMNS, label="Ranked sentences", wrap=True, interactive=False
                    )
                    search_plot = gr.Plot(label=f"Top {TOP_K_CHART} matches")

        with gr.Tab("Similarity matrix"):
            with gr.Row():
                with gr.Column(scale=1):
                    matrix_input = gr.Textbox(label="Sentences (one per line)", value=EXAMPLE_SENTENCES, lines=12)
                    matrix_model = MODELS.dropdown()
                    matrix_btn = gr.Button("Build matrix", variant="primary")
                with gr.Column(scale=2):
                    matrix_plot = gr.Plot(label="Pairwise cosine similarity")

        with gr.Tab("Embedding map"):
            with gr.Row():
                with gr.Column(scale=1):
                    map_input = gr.Textbox(label="Sentences (one per line)", value=EXAMPLE_SENTENCES, lines=12)
                    map_method = gr.Radio(choices=METHODS, value="PCA", label="Projection")
                    map_model = MODELS.dropdown()
                    map_btn = gr.Button("Draw map", variant="primary")
                with gr.Column(scale=2):
                    map_plot = gr.Plot(label="Sentences in 2-D")

        shell.about(ABOUT, MODELS)

        compare_btn.click(
            compare_sentences, inputs=[sent1_input, sent2_input, pair_model], outputs=[pair_result, pair_plot]
        )
        search_btn.click(
            semantic_search, inputs=[corpus_input, query_input, search_model], outputs=[search_table, search_plot]
        )
        query_input.submit(
            semantic_search,
            inputs=[corpus_input, query_input, search_model],
            outputs=[search_table, search_plot],
            api_visibility="undocumented",  # same endpoint as the Search button
        )
        matrix_btn.click(similarity_matrix, inputs=[matrix_input, matrix_model], outputs=matrix_plot)
        map_btn.click(visualize_embeddings, inputs=[map_input, map_method, map_model], outputs=map_plot)
    return demo


if __name__ == "__main__":
    shell.launch(create_ui(), UI_KEY, argv=sys.argv[1:])
