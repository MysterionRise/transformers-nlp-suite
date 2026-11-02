"""Tests for the NER UI (ui/ner_visualizer.py)."""

import json
import re
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import pytest

from tests.unit.ui.conftest import FakePipeline, FakeTokenizer
from ui import ner_visualizer as ui

SPACY = "ner/spacy_sm"
GLINER = "ner/gliner_multi"
BIOMED = "medical_ner/biomedical_ner"

TEXT = "Apple Inc. was founded by Steve Jobs in Cupertino in April 1976."


class FakeNlp:
    """spaCy Language stand-in: tags every occurrence of the given phrases."""

    def __init__(self, phrases: Dict[str, str], max_length: int = 1_000_000):
        self.phrases = phrases
        self.max_length = max_length
        self.calls: List[str] = []

    def __call__(self, text: str) -> SimpleNamespace:
        self.calls.append(text)
        ents = [
            SimpleNamespace(text=m.group(), label_=label, start_char=m.start(), end_char=m.end())
            for phrase, label in self.phrases.items()
            for m in re.finditer(re.escape(phrase), text)
        ]
        return SimpleNamespace(ents=sorted(ents, key=lambda e: e.start_char))


class FakeGliner:
    """GLiNER stand-in: finds the given phrases, labelled with the first requested label they map to."""

    def __init__(self, phrases: Dict[str, str], max_len: int = 384, score: float = 0.9):
        self.phrases = phrases
        self.score = score
        self.config = SimpleNamespace(max_len=max_len)
        self.calls: List[Dict[str, Any]] = []

    def predict_entities(self, text: str, labels: List[str], threshold: float = 0.5) -> List[Dict[str, Any]]:
        self.calls.append({"text": text, "labels": labels, "threshold": threshold})
        found = []
        for phrase, label in self.phrases.items():
            if label not in labels:
                continue
            for m in re.finditer(re.escape(phrase), text):
                found.append(
                    {"start": m.start(), "end": m.end(), "text": m.group(), "label": label, "score": self.score}
                )
        return [e for e in found if e["score"] >= threshold]


@pytest.fixture(autouse=True)
def spacy_installed():
    """Unit tests must not depend on which spaCy pipelines the machine has installed."""
    with patch("spacy.util.is_package", return_value=True):
        yield


@pytest.fixture(autouse=True)
def fresh_session():
    ui._loaded.clear()
    yield
    ui._loaded.clear()


def assert_offsets_exact(text: str, df: pd.DataFrame) -> None:
    assert len(df) > 0
    for row in df.itertuples():
        assert text[row.Start : row.End] == row.Text


class TestInputValidation:
    def test_empty_input_is_a_friendly_error(self):
        with pytest.raises(gr.Error, match="Please enter some text"):
            ui.extract_entities("  \n ", ui.MODELS.default)

    def test_unknown_model(self):
        with pytest.raises(gr.Error, match="Unknown model"):
            ui.extract_entities(TEXT, "Not A Model")

    def test_missing_spacy_package_gives_install_hint(self, fake_models):
        with patch("spacy.util.is_package", return_value=False):
            with pytest.raises(gr.Error, match="python -m spacy download en_core_web_sm"):
                ui.extract_entities(TEXT, ui.SPACY_SM)


class TestSpacy:
    def test_outputs_and_types(self, fake_models):
        nlp = fake_models[SPACY] = FakeNlp({"Apple Inc.": "ORG", "Steve Jobs": "PERSON", "Cupertino": "GPE"})
        html, df, fig, js = ui.extract_entities(TEXT, ui.SPACY_SM, [], "", 0.5)

        assert nlp.calls == [TEXT]
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Text", "Type", "Start", "End"]  # spaCy has no scores
        assert df["Text"].tolist() == ["Apple Inc.", "Steve Jobs", "Cupertino"]
        assert df["Type"].tolist() == ["ORG", "PERSON", "GPE"]
        assert_offsets_exact(TEXT, df)

        assert isinstance(fig, go.Figure)
        assert dict(zip(fig.data[0].x, fig.data[0].y)) == {"ORG": 1, "PERSON": 1, "GPE": 1}
        assert json.loads(js) == [
            {"text": "Apple Inc.", "type": "ORG", "start": 0, "end": 10},
            {"text": "Steve Jobs", "type": "PERSON", "start": 26, "end": 36},
            {"text": "Cupertino", "type": "GPE", "start": 40, "end": 49},
        ]
        assert html.startswith('<div class="tns-text">')
        assert html.count('<mark class="tns-mark">') == 3
        assert '<span class="tns-label">PERSON</span>' in html
        assert "style=" not in html  # theme classes, no inline light colours

    def test_entity_type_filter(self, fake_models):
        fake_models[SPACY] = FakeNlp({"Apple Inc.": "ORG", "Steve Jobs": "PERSON", "Cupertino": "GPE"})
        _, df, fig, _ = ui.extract_entities(TEXT, ui.SPACY_SM, ["PERSON", "GPE"], "", 0.5)
        assert df["Type"].tolist() == ["PERSON", "GPE"]
        assert list(fig.data[0].x) == ["PERSON", "GPE"]

    def test_table_does_not_need_tabulate(self, fake_models):
        """Regression: the old UI used DataFrame.to_markdown(), which needs the uninstalled tabulate."""
        fake_models[SPACY] = FakeNlp({"Steve Jobs": "PERSON"})
        with patch.object(
            pd.DataFrame, "to_markdown", side_effect=ImportError("Missing optional dependency 'tabulate'")
        ):
            _, df, _, _ = ui.extract_entities(TEXT, ui.SPACY_SM)
        assert df["Text"].tolist() == ["Steve Jobs"]

    def test_no_entities(self, fake_models):
        fake_models[SPACY] = FakeNlp({})
        with patch("ui.common.info") as info:
            html, df, fig, js = ui.extract_entities("<b>nothing here</b>", ui.SPACY_SM)
        assert html == '<div class="tns-text">&lt;b&gt;nothing here&lt;/b&gt;</div>'
        assert df.empty and list(df.columns) == ["Text", "Type", "Start", "End"]
        assert fig is None  # None, not {} or "", for an empty gr.Plot
        assert js == "[]"
        info.assert_called_once()
        assert "No entities found" in info.call_args.args[0]

    def test_overlong_text_is_clipped_with_warning(self, fake_models):
        fake_models[SPACY] = FakeNlp({"Steve Jobs": "PERSON"}, max_length=40)
        with pytest.warns(UserWarning, match="only the first 40 were analyzed"):
            html, df, _, _ = ui.extract_entities(TEXT, ui.SPACY_SM)
        assert_offsets_exact(TEXT, df)
        assert "Cupertino" not in html


class TestGliner:
    PHRASES = {"Apple Inc.": "organization", "Steve Jobs": "person", "<script>alert(1)</script>": "person"}

    def test_escapes_user_text_and_labels(self, fake_models):
        text = "Steve Jobs wrote <script>alert(1)</script> & left."
        fake_models[GLINER] = FakeGliner(
            {"Steve Jobs": '<img src=x onerror="alert(2)">', "<script>alert(1)</script>": "person"}
        )
        html, df, _, _ = ui.extract_entities(text, ui.GLINER, [], 'person, <img src=x onerror="alert(2)">', 0.5)

        assert "<script>" not in html and "<img" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        assert "&lt;img src=x onerror=&quot;alert(2)&quot;&gt;" in html
        assert "&amp; left." in html
        assert_offsets_exact(text, df)

    def test_scores_labels_and_threshold(self, fake_models):
        fake = fake_models[GLINER] = FakeGliner(self.PHRASES, score=0.8)
        _, df, fig, js = ui.extract_entities(TEXT, ui.GLINER, ["PERSON"], "person, organization, person", 0.3)

        assert fake.calls[0]["labels"] == ["person", "organization"]  # de-duplicated, spaCy filter ignored
        assert fake.calls[0]["threshold"] == pytest.approx(0.3)
        assert list(df.columns) == ui.ENTITY_COLUMNS
        assert df["Type"].tolist() == ["organization", "person"]
        assert df["Score"].tolist() == [0.8, 0.8]
        assert_offsets_exact(TEXT, df)
        assert isinstance(fig, go.Figure)
        assert json.loads(js)[1] == {"text": "Steve Jobs", "type": "person", "start": 26, "end": 36, "score": 0.8}

    def test_empty_labels_use_general_set(self, fake_models):
        fake = fake_models[GLINER] = FakeGliner(self.PHRASES)
        ui.extract_entities(TEXT, ui.GLINER, [], " , ", 0.5)
        assert fake.calls[0]["labels"] == ["person", "organization", "location", "date", "event", "product"]

    def test_first_run_tells_user_about_download(self, fake_models):
        fake_models[GLINER] = FakeGliner(self.PHRASES)
        with patch("ui.common.info") as info:
            ui.extract_entities(TEXT, ui.GLINER)
            ui.extract_entities(TEXT, ui.GLINER)
        messages = [call.args[0] for call in info.call_args_list]
        assert len([m for m in messages if "0.8 GB" in m]) == 1

    def test_no_download_notice_for_spacy(self, fake_models):
        fake_models[SPACY] = FakeNlp({"Steve Jobs": "PERSON"})
        with patch("ui.common.info") as info:
            ui.extract_entities(TEXT, ui.SPACY_SM)
        info.assert_not_called()

    def test_long_text_is_clipped_on_a_word_boundary_and_offsets_stay_exact(self, fake_models):
        fake = fake_models[GLINER] = FakeGliner({"Steve Jobs": "person"}, max_len=13)
        text = " ".join(["Then Steve Jobs spoke."] * 10)
        with pytest.warns(UserWarning, match="only the first 13 were used"):
            _, df, _, _ = ui.extract_entities(text, ui.GLINER, [], "person", 0.5)

        sent = fake.calls[0]["text"]
        assert text.startswith(sent)
        assert sent == "Then Steve Jobs spoke. Then Steve Jobs spoke. Then Steve Jobs"  # 13 words, "." counts as one
        assert len(df) == 3
        assert_offsets_exact(text, df)

    def test_inference_failure_is_reported(self, fake_models):
        class Broken(FakeGliner):
            def predict_entities(self, *args, **kwargs):
                raise RuntimeError("CUDA out of memory")

        fake_models[GLINER] = Broken({})
        with pytest.raises(gr.Error, match="Entity extraction failed"):
            ui.extract_entities(TEXT, ui.GLINER)

    def test_load_failure_is_reported(self):
        with patch("ui.common.load_registry_model", side_effect=OSError("hub unreachable")):
            with pytest.raises(gr.Error, match="Loading GLiNER Multi v2.1 failed"):
                ui.extract_entities(TEXT, ui.GLINER)


class TestTransformersPipeline:
    NAME = "BiomedNLP NER"

    @staticmethod
    def tagger(text, **_kwargs):
        out = []
        for phrase, group, score in [(" Metformin", "Medication", 0.99), ("diabetes", "Disease_disorder", 0.4)]:
            for m in re.finditer(re.escape(phrase), text):
                out.append(
                    {"entity_group": group, "score": score, "word": phrase.lower(), "start": m.start(), "end": m.end()}
                )
        return out

    def test_aggregation_offsets_and_threshold(self, fake_models):
        fake = fake_models[BIOMED] = FakePipeline(self.tagger)
        text = "Started Metformin for type 2 diabetes."
        _, df, _, _ = ui.extract_entities(text, self.NAME, [], "", 0.5)

        assert fake.calls[0]["aggregation_strategy"] == "simple"
        assert df["Text"].tolist() == ["Metformin"]  # original casing, leading space trimmed
        assert df["Type"].tolist() == ["Medication"]
        assert_offsets_exact(text, df)  # "diabetes" (0.4) is below the threshold

    def test_long_text_is_clipped_with_warning(self, fake_models):
        fake = fake_models[BIOMED] = FakePipeline(self.tagger, FakeTokenizer(model_max_length=12))
        text = " ".join(["Given Metformin today."] * 10)
        with pytest.warns(UserWarning, match="only the first 10 were used"):
            _, df, _, _ = ui.extract_entities(text, self.NAME, [], "", 0.5)
        assert fake.calls[0]["inputs"] == " ".join(text.split()[:10])
        assert len(df) == 3
        assert_offsets_exact(text, df)


class TestPageWiring:
    def test_no_model_is_loaded_while_building(self):
        with (
            patch("ui.common.load_registry_model", side_effect=AssertionError("eager load")) as loader,
            patch("spacy.load", side_effect=AssertionError("eager spaCy load")) as spacy_load,
        ):
            demo = ui.create_ui()
        assert isinstance(demo, gr.Blocks)
        loader.assert_not_called()
        spacy_load.assert_not_called()

    def test_results_table_is_a_dataframe_and_options_sit_in_an_accordion(self):
        demo = ui.create_ui()
        blocks = list(demo.blocks.values())
        tables = [b for b in blocks if isinstance(b, gr.Dataframe)]
        assert len(tables) == 1 and tables[0].headers == ui.ENTITY_COLUMNS

        filters = [b for b in blocks if isinstance(b, gr.CheckboxGroup)]
        assert len(filters) == 1

        def ancestors(block):
            while block is not None:
                yield block
                block = block.parent

        assert any(isinstance(a, gr.Accordion) for a in ancestors(filters[0]))
        presets = [b for b in blocks if isinstance(b, gr.Dropdown) and b.label == "Label preset (GLiNER)"]
        assert len(presets) == 1 and any(isinstance(a, gr.Accordion) for a in ancestors(presets[0]))

    def test_submit_follows_input_and_model(self):
        """Keyboard users reach the button right after the text and model fields."""
        demo = ui.create_ui()
        button = next(b for b in demo.blocks.values() if isinstance(b, gr.Button) and b.value == "Extract entities")

        def components(block):
            for child in getattr(block, "children", []):
                if isinstance(child, gr.components.Component):
                    yield child
                yield from components(child)

        in_tab_order = [type(c) for c in components(button.parent)]
        assert in_tab_order[:3] == [gr.Textbox, gr.Dropdown, gr.Button]

    def test_spacy_trf_offered_only_when_installed(self):
        with patch("spacy.util.is_package", return_value=False):
            assert ui._uninstalled_spacy_models() == ["spacy_trf"]
        with patch("spacy.util.is_package", return_value=True):
            assert ui._uninstalled_spacy_models() == []
        assert {"spaCy Small", "GLiNER Multi v2.1"} <= set(ui.MODELS.names)

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("spaCy Small", [True, False, False, False]),
            ("GLiNER Multi v2.1", [False, True, True, True]),
            ("BiomedNLP NER", [False, False, False, True]),
        ],
    )
    def test_option_visibility(self, model, expected):
        assert [update["visible"] for update in ui.option_visibility(model)] == expected

    def test_presets(self):
        assert ui.apply_preset("Medical: Clinical") == ui.GLINER_PRESETS["Medical: Clinical"]
        assert ui.apply_preset(None) == gr.skip()

    def test_examples_reference_menu_models(self):
        for _, model, types, _, _ in ui.EXAMPLES:
            assert model in ui.MODELS
            assert set(types) <= set(ui.SPACY_TYPES)
