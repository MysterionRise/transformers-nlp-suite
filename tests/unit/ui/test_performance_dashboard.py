"""Tests for the evaluation metrics UI (ui/performance_dashboard.py)."""

import importlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import numpy as np
import pandas as pd
import pytest

from tests.unit.ui.conftest import FakeTokenizer
from ui import performance_dashboard as ui

REPO = Path(__file__).resolve().parents[3]
REFERENCE = "The cat sat on the mat. It was very happy there."
DIFFERENT = "A dog slept under the old wooden table all day."


@pytest.fixture(autouse=True)
def offline_nltk():
    """Never download NLTK data in unit tests; each test starts with no cached availability."""
    ui._nltk_ready.clear()
    with patch.object(ui, "_download_nltk", return_value=False) as download:
        yield download
    ui._nltk_ready.clear()


@pytest.fixture
def without_wordnet():
    ui._nltk_ready["wordnet"] = False


class FakeScorer:
    """Stands in for bert_score.BERTScorer: records inputs, returns fixed scores."""

    def __init__(self):
        self._tokenizer = FakeTokenizer(model_max_length=10_000)
        self.calls = []

    def score(self, cands, refs):
        self.calls.append((list(cands), list(refs)))
        n = len(cands)
        return np.full(n, 0.91), np.full(n, 0.87), np.full(n, 0.89)


def test_import_and_page_build_need_no_network():
    import nltk

    def no_network(*_args, **_kwargs):
        raise AssertionError("network access while importing or building the page")

    with (
        patch.object(nltk, "download", side_effect=no_network) as download,
        patch.object(socket.socket, "connect", side_effect=no_network),
    ):
        module = importlib.reload(ui)
        demo = module.create_ui()
    download.assert_not_called()
    assert demo.title.startswith("Evaluation Metrics")


def test_imports_in_script_mode_from_the_ui_directory():
    """`python ui/performance_dashboard.py` puts ui/ (not the repo root) on sys.path."""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    code = "import performance_dashboard as m; m.create_ui(); print('built', m.UI_KEY)"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO / "ui", env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "built performance" in result.stdout


class TestEvaluateText:
    def test_identical_text_scores_one(self):
        scores = ui.evaluate_text(REFERENCE, REFERENCE)
        assert scores["BLEU"] == 1.0
        assert scores["ROUGE-1"] == scores["ROUGE-2"] == scores["ROUGE-L"] == 1.0
        assert scores["Cosine (bag of words)"] == 1.0

    def test_different_text_scores_lower(self):
        same = ui.evaluate_text(REFERENCE, REFERENCE)
        scores = ui.evaluate_text(REFERENCE, DIFFERENT)
        assert 0.0 < scores["ROUGE-L"] < 1.0
        assert scores["ROUGE-L"] < same["ROUGE-L"]
        assert scores["BLEU"] < 0.2
        assert scores["Cosine (bag of words)"] < same["Cosine (bag of words)"]

    def test_meteor_is_skipped_with_a_warning_when_wordnet_is_missing(self, without_wordnet):
        with pytest.warns(UserWarning, match="METEOR was skipped"):
            scores = ui.evaluate_text(REFERENCE, REFERENCE)
        assert "METEOR" not in scores
        assert scores["BLEU"] == 1.0 and scores["ROUGE-L"] == 1.0

    def test_meteor_is_computed_when_wordnet_is_available(self):
        ui._nltk_ready["wordnet"] = True
        with patch("nltk.translate.meteor_score.single_meteor_score", return_value=0.4242) as meteor:
            scores = ui.evaluate_text("The cat sat.", "A cat sat.")
        assert scores["METEOR"] == 0.4242
        reference_tokens, candidate_tokens = meteor.call_args.args
        assert reference_tokens == ["The", "cat", "sat", "."]
        assert candidate_tokens == ["A", "cat", "sat", "."]

    def test_missing_punkt_falls_back_to_a_simple_tokenizer_with_a_warning(self, without_wordnet):
        ui._nltk_ready["punkt_tab"] = False
        with pytest.warns(UserWarning, match="punkt_tab"):
            tools = ui.prepare_metrics()
        assert tools.tokenize("The cat sat. It slept!") == ["The", "cat", "sat", ".", "It", "slept", "!"]
        assert tools.meteor is None

    def test_nltk_download_is_attempted_once_per_process(self, offline_nltk):
        with patch.object(ui, "_nltk_installed", return_value=False):
            ui.evaluate_text(REFERENCE, DIFFERENT)
            ui.evaluate_text(REFERENCE, REFERENCE)
        assert sorted(call.args[0] for call in offline_nltk.call_args_list) == ["punkt_tab", "wordnet"]

    def test_bertscore_only_when_requested(self):
        with patch.object(ui, "bertscore", return_value=[{"BERTScore F1": 0.88}]) as bertscore:
            scores = ui.evaluate_text(REFERENCE, DIFFERENT)
            assert not any(name.startswith("BERTScore") for name in scores)
            bertscore.assert_not_called()

            scores = ui.evaluate_text(REFERENCE, DIFFERENT, use_bertscore=True)
        bertscore.assert_called_once_with([REFERENCE], [DIFFERENT])
        assert scores["BERTScore F1"] == 0.88

    def test_empty_inputs_are_friendly_errors(self):
        with pytest.raises(gr.Error, match="reference"):
            ui.evaluate_text("  ", "text")
        with pytest.raises(gr.Error, match="candidate"):
            ui.evaluate_text("text", "")


class TestBertScore:
    def test_scores_and_clipping_to_the_model_limit(self):
        scorer = FakeScorer()
        long_candidate = " ".join(f"w{i}" for i in range(ui.BERTSCORE_MAX_TOKENS + 40))
        with patch.object(ui, "_get_bert_scorer", return_value=scorer):
            with pytest.warns(UserWarning, match=f"only the first {ui.BERTSCORE_MAX_TOKENS} were used"):
                results = ui.bertscore(["short reference"], [long_candidate])
        assert results == [{"BERTScore P": 0.91, "BERTScore R": 0.87, "BERTScore F1": 0.89}]
        cands, refs = scorer.calls[0]
        assert refs == ["short reference"]
        assert len(cands[0].split()) == ui.BERTSCORE_MAX_TOKENS

    def test_checkboxes_are_off_by_default_and_state_the_download(self):
        demo = ui.create_ui()
        boxes = [block for block in demo.blocks.values() if isinstance(block, gr.Checkbox)]
        assert len(boxes) == 2
        assert all(box.value is False and "1.4 GB" in box.label for box in boxes)


class TestBatch:
    def test_separator_lines(self):
        text = "first\n---\nsecond\n  ----  \r\nthird\n---\n\nnot --- a separator"
        assert ui.split_candidates(text) == ["first", "second", "third", "not --- a separator"]

    def test_candidate_count_is_capped(self):
        text = "\n---\n".join(f"candidate {i}" for i in range(ui.MAX_CANDIDATES + 5))
        with pytest.warns(UserWarning, match=f"Only the first {ui.MAX_CANDIDATES}"):
            assert len(ui.split_candidates(text)) == ui.MAX_CANDIDATES

    def test_results_charts_and_export_records(self, without_wordnet):
        df, radar, bars, records = ui.batch_evaluate(REFERENCE, f"{REFERENCE}\n---\n{DIFFERENT}")
        assert isinstance(df, pd.DataFrame)
        assert df["candidate"].tolist() == ["Candidate 1", "Candidate 2"]
        assert df["BLEU"].tolist()[0] == 1.0 and df["BLEU"].tolist()[1] < 0.2
        assert "METEOR" not in df.columns  # skipped metrics are not shown as zeros
        assert [record["text"] for record in records] == [REFERENCE, DIFFERENT]
        assert list(radar.data[0].theta[:-1]) == ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
        assert [trace.name for trace in bars.data] == ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"]

        exported = json.loads(ui.export_results(records))
        assert exported == records

    def test_batch_bertscore_scores_every_candidate_in_one_call(self, without_wordnet):
        scorer = FakeScorer()
        with patch.object(ui, "_get_bert_scorer", return_value=scorer):
            df, _, bars, _ = ui.batch_evaluate(REFERENCE, "one\n---\ntwo", use_bertscore=True)
        assert scorer.calls == [(["one", "two"], [REFERENCE, REFERENCE])]
        assert df["BERTScore F1"].tolist() == [0.89, 0.89]
        assert "BERTScore F1" in [trace.name for trace in bars.data]

    def test_export_needs_a_batch_run(self):
        with pytest.raises(gr.Error, match="Run a batch comparison first"):
            ui.export_results([])

    def test_export_tab_is_linked_to_batch_results(self):
        demo = ui.create_ui()
        events = list(demo.fns.values())
        batch = next(event for event in events if event.fn is ui.batch_evaluate)
        export = next(event for event in events if event.fn is ui.export_results)
        preview = next(event for event in events if event.fn is ui.records_frame)
        state = export.inputs[0]
        assert isinstance(state, gr.State)
        assert any(output is state for output in batch.outputs)
        assert preview.inputs[0] is state and isinstance(preview.outputs[0], gr.Dataframe)


class TestModelsAndCache:
    def test_lists_registry_and_cache_state(self):
        from unittest.mock import patch

        from ui import common
        from ui import performance_dashboard as ui

        fake_info = {"cached_models": 1, "max_size": 3, "device": "cpu", "models": ["spacy::ner::spacy_sm"]}
        with patch("ui.common.cache_info", return_value=fake_info):
            info, table = ui.models_and_cache()
        assert info == fake_info
        assert list(table.columns) == ui.REGISTRY_COLUMNS
        assert len(table) == len(common.registry_rows())
        assert ("ner", "gliner_multi", "gliner") in set(zip(table["Category"], table["Key"], table["Framework"]))
