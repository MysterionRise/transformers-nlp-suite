"""Tests for the similarity UI (ui/similarity_explorer.py)."""

import sys
import zlib
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import patch

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from tests.unit.ui.conftest import FakeTokenizer
from ui import similarity_explorer as ui

DEFAULT = "embeddings/all_minilm_l6"


class FakeEmbedder:
    """SentenceTransformer stand-in: fixed vectors for known sentences, deterministic ones otherwise."""

    def __init__(
        self,
        vectors: Optional[Dict[str, Sequence[float]]] = None,
        max_seq_length: int = 256,
        tokenizer: Optional[FakeTokenizer] = None,
        dim: int = 8,
    ):
        self.vectors = vectors or {}
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer or FakeTokenizer(model_max_length=512)
        self.dim = dim
        self.calls: List[Dict[str, Any]] = []

    def _vector(self, sentence: str) -> np.ndarray:
        if sentence in self.vectors:
            return np.asarray(self.vectors[sentence], dtype=float)
        return np.random.default_rng(zlib.crc32(sentence.encode())).normal(size=self.dim)

    def encode(self, sentences, normalize_embeddings: bool = False, **kwargs: Any) -> np.ndarray:
        self.calls.append({"sentences": list(sentences), "normalize_embeddings": normalize_embeddings, **kwargs})
        out = np.stack([self._vector(s) for s in sentences])
        if normalize_embeddings:
            out = out / np.linalg.norm(out, axis=1, keepdims=True)
        return out


SENTENCES = "The cat sat on the mat.\nDogs are loyal animals.\nPython is a programming language."


class TestCompareSentences:
    @pytest.mark.parametrize("first, second, missing", [("", "b", "first"), ("a", "  ", "second")])
    def test_empty_input_is_a_friendly_error(self, first, second, missing):
        with pytest.raises(gr.Error, match=f"Please enter the {missing} sentence"):
            ui.compare_sentences(first, second, ui.MODELS.default)

    def test_cosine_of_normalised_embeddings(self, fake_models):
        # Unnormalised vectors: the dot product is 6, the cosine 0.6.
        fake = fake_models[DEFAULT] = FakeEmbedder({"cat": [2.0, 0.0], "kitten": [3.0, 4.0]})
        text, fig = ui.compare_sentences("cat", "kitten", ui.MODELS.default)

        assert fake.calls[0]["normalize_embeddings"] is True
        assert fake.calls[0]["sentences"] == ["cat", "kitten"]
        assert text == "0.600 · Moderate similarity: somewhat related"
        assert isinstance(fig, go.Figure)
        assert fig.data[0].value == pytest.approx(0.6)

    def test_selected_model_is_used(self, fake_models):
        fake = fake_models["embeddings/all_mpnet_base"] = FakeEmbedder({"a": [1, 0], "b": [1, 0]})
        text, _ = ui.compare_sentences("a", "b", "All-MPNet-Base-v2")
        assert len(fake.calls) == 1
        assert text.startswith("1.000 · Very high similarity")

    def test_long_sentence_is_clipped_to_max_seq_length_with_warning(self, fake_models):
        # The tokenizer allows 510 tokens but the model only reads max_seq_length=7 (5 + 2 special).
        fake = fake_models[DEFAULT] = FakeEmbedder(max_seq_length=7)
        long_sentence = " ".join(f"w{i}" for i in range(20))
        with pytest.warns(UserWarning, match="1 sentence\\(s\\) exceeded the model's 5-token limit"):
            ui.compare_sentences(long_sentence, "short one", ui.MODELS.default)
        assert fake.calls[0]["sentences"] == ["w0 w1 w2 w3 w4", "short one"]

    def test_encoding_failure_is_reported(self, fake_models):
        class Broken(FakeEmbedder):
            def encode(self, *args, **kwargs):
                raise RuntimeError("out of memory")

        fake_models[DEFAULT] = Broken()
        with pytest.raises(gr.Error, match="Computing embeddings failed"):
            ui.compare_sentences("a", "b", ui.MODELS.default)


class TestSemanticSearch:
    def test_needs_query_and_sentences(self):
        with pytest.raises(gr.Error, match="Please enter a query"):
            ui.semantic_search(SENTENCES, " ", ui.MODELS.default)
        with pytest.raises(gr.Error, match="Please enter at least one sentence"):
            ui.semantic_search("\n\n", "pets", ui.MODELS.default)

    def test_ranked_table_and_chart(self, fake_models):
        fake_models[DEFAULT] = FakeEmbedder(
            {
                "The cat sat on the mat.": [0.8, 0.6],
                "Dogs are loyal animals.": [1.0, 0.0],
                "Python is a programming language.": [0.0, 1.0],
                "pets": [1.0, 0.0],
            }
        )
        df, fig = ui.semantic_search(SENTENCES, "pets", ui.MODELS.default)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ui.SEARCH_COLUMNS
        assert df["Sentence #"].tolist() == [2, 1, 3]
        assert df["Score"].tolist() == [1.0, 0.8, 0.0]
        assert df["Rank"].tolist() == [1, 2, 3]
        bar = fig.data[0]
        assert list(bar.y) == [
            "#2: Dogs are loyal animals.",
            "#1: The cat sat on the mat.",
            "#3: Python is a programming language.",
        ]
        assert list(bar.x) == pytest.approx([1.0, 0.8, 0.0])


class TestSimilarityMatrix:
    @pytest.mark.parametrize("text", ["", "only one sentence", "one\n\n   \n"])
    def test_fewer_than_two_sentences_is_an_error(self, text):
        with pytest.raises(gr.Error, match="at least 2 sentences"):
            ui.similarity_matrix(text, ui.MODELS.default)

    def test_symmetric_matrix_with_unit_diagonal(self, fake_models):
        fake_models[DEFAULT] = FakeEmbedder()
        fig = ui.similarity_matrix(SENTENCES, ui.MODELS.default)
        z = np.asarray(fig.data[0].z)
        assert z.shape == (3, 3)
        assert np.allclose(z, z.T)
        assert np.allclose(np.diag(z), 1.0)
        assert list(fig.data[0].x) == ["S1", "S2", "S3"]


class FakeUMAP:
    instances: List["FakeUMAP"] = []

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        FakeUMAP.instances.append(self)

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        return np.asarray(vectors)[:, :2]


@pytest.fixture
def fake_umap():
    module = ModuleType("umap")
    module.UMAP = FakeUMAP
    FakeUMAP.instances = []
    with patch.dict(sys.modules, {"umap": module}), patch("ui.common.info"):
        yield FakeUMAP


class TestEmbeddingMap:
    @pytest.mark.parametrize("text", ["", "just one", "one\n \n"])
    def test_fewer_than_two_sentences_is_an_error(self, text):
        with pytest.raises(gr.Error, match="at least 2 sentences"):
            ui.visualize_embeddings(text, "PCA", ui.MODELS.default)

    @pytest.mark.parametrize("method", ["PCA", "t-SNE"])
    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_small_inputs_project_to_2d(self, fake_models, method, n):
        fake_models[DEFAULT] = FakeEmbedder()
        text = "\n".join(f"sentence number {i}" for i in range(n))
        fig = ui.visualize_embeddings(text, method, ui.MODELS.default)
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].x) == n
        assert np.isfinite(np.asarray(fig.data[0].x, dtype=float)).all()

    @pytest.mark.parametrize("n", [2, 3, 10, 200])
    def test_tsne_perplexity_is_below_sample_count(self, n):
        from sklearn.manifold import TSNE

        vectors = np.random.default_rng(0).normal(size=(n, 8))
        with patch("sklearn.manifold.TSNE", wraps=TSNE) as spy:
            ui.project(vectors, "t-SNE")
        perplexity = spy.call_args.kwargs["perplexity"]
        assert 0 < perplexity < n
        assert perplexity <= 30

    def test_umap_is_real_and_lazy(self, fake_models, fake_umap):
        fake_models[DEFAULT] = FakeEmbedder()
        fig = ui.visualize_embeddings(SENTENCES, "UMAP", ui.MODELS.default)
        assert len(fig.data[0].x) == 3
        (reducer,) = fake_umap.instances
        assert reducer.kwargs["n_components"] == 2
        assert 2 <= reducer.kwargs["n_neighbors"] < 3
        assert fig.layout.xaxis.title.text == "UMAP 1"

    def test_umap_needs_three_sentences(self, fake_models, fake_umap):
        fake_models[DEFAULT] = FakeEmbedder()
        with pytest.raises(gr.Error, match="UMAP needs at least 3 sentences"):
            ui.visualize_embeddings("a\nb", "UMAP", ui.MODELS.default)
        assert fake_umap.instances == []

    def test_unknown_method_is_not_silently_replaced(self, fake_models):
        fake_models[DEFAULT] = FakeEmbedder()
        with pytest.raises(gr.Error, match="Unknown projection method 'LDA'"):
            ui.visualize_embeddings(SENTENCES, "LDA", ui.MODELS.default)


class TestPage:
    def test_build_loads_no_model(self):
        with patch("ui.common.load_registry_model", side_effect=AssertionError("eager load")) as loader:
            demo = ui.create_ui()
        loader.assert_not_called()
        blocks = list(demo.blocks.values())
        dropdowns = [b for b in blocks if isinstance(b, gr.Dropdown)]
        assert len(dropdowns) == 4  # one model selector per tab
        assert all([value for _, value in d.choices] == ui.MODELS.names for d in dropdowns)
        radios = [b for b in blocks if isinstance(b, gr.Radio)]
        assert [value for _, value in radios[0].choices] == ["PCA", "t-SNE", "UMAP"]

    def test_menu_is_the_embeddings_registry(self):
        assert ui.MODELS.default == "All-MiniLM-L6-v2"
        assert all(choice.config.framework == "sentence-transformers" for choice in ui.MODELS)
