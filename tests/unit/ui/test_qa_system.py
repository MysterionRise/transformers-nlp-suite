"""Tests for the question answering UI (ui/qa_system.py)."""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import pytest

from tests.unit.ui.conftest import FakePipeline
from ui import qa_system as ui

DISTIL = "question_answering/distilbert_squad"
ROBERTA = "question_answering/roberta_squad"
CONTEXT = "The Eiffel Tower is located in Paris, France. It was built in 1889."


def span(context, text, score):
    start = context.index(text)
    return {"answer": text, "score": score, "start": start, "end": start + len(text)}


def finds(text, score=0.9):
    """Fake extractor that returns `text` located in whatever context it receives."""
    return lambda inputs, **_kwargs: span(inputs["context"], text, score)


def wiring(demo, fn):
    return next(dep for dep in demo.fns.values() if dep.fn is fn)


class TestAnswerQuestion:
    def test_empty_question_raises_before_loading(self, fake_models):
        # fake_models has no entries: any model load would fail the test with AssertionError.
        with pytest.raises(gr.Error, match="Please enter a question"):
            ui.answer_question(CONTEXT, "   ", ui.MODELS.default)

    def test_empty_context_raises(self, fake_models):
        with pytest.raises(gr.Error, match="context passage"):
            ui.answer_question("", "Where is it?", ui.MODELS.default)

    def test_confident_answer(self, fake_models):
        fake = fake_models[DISTIL] = FakePipeline(finds("Paris"))
        outputs = ui.answer_question(CONTEXT, "Where is the Eiffel Tower?", ui.MODELS.default)

        assert len(outputs) == len(wiring(ui.create_ui(), ui.answer_question).outputs) == 3
        result, html, fig = outputs
        assert result["status"] == ui.ANSWER
        assert result["verdict"] == "Answer found"
        assert result["answer"] == "Paris"
        assert result["confidence"] == "90.0%"
        assert (result["start"], result["end"]) == (31, 36)
        assert result["model"] == "distilbert-base-cased-distilled-squad"
        assert '<mark class="tns-mark">Paris<span class="tns-label">answer</span></mark>' in html
        assert 'class="tns-text"' in html and "background-color" not in html
        assert isinstance(fig, go.Figure) and fig.data[0].value == 90.0
        call = fake.calls[0]
        assert call["inputs"] == {"question": "Where is the Eiffel Tower?", "context": CONTEXT}
        assert call["top_k"] == 1 and "handle_impossible_answer" not in call

    def test_user_text_is_escaped(self, fake_models):
        context = 'Hi <script>alert("x")</script> the <b>capital</b> is Paris & more.'
        fake_models[DISTIL] = FakePipeline(finds("<b>capital</b>"))
        result, html, _ = ui.answer_question(context, "What?", ui.MODELS.default)
        assert result["answer"] == "<b>capital</b>"
        assert "<script>" not in html and "<b>" not in html
        assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in html
        assert '<mark class="tns-mark">&lt;b&gt;capital&lt;/b&gt;' in html
        assert "Paris &amp; more." in html

    def test_low_confidence_span_is_not_presented_as_the_answer(self, fake_models):
        fake_models[DISTIL] = FakePipeline(finds("1889", score=0.028))
        result, html, fig = ui.answer_question(CONTEXT, "Who painted the Mona Lisa?", ui.MODELS.default)
        assert result["status"] == ui.LOW_CONFIDENCE
        assert result["verdict"] == "No confident answer found"
        assert "answer" not in result
        assert result["best_guess"] == "1889"
        assert result["confidence"] == "2.8% (low)"
        assert "below 10%" in result["explanation"]
        assert '<span class="tns-label">low confidence</span>' in html
        assert fig.data[0].title.text == "Best guess confidence"

    def test_squad2_model_can_say_no_answer(self, fake_models):
        def unanswerable(inputs, **_kwargs):
            return [{"answer": "", "score": 0.7, "start": 0, "end": 0}, span(inputs["context"], "Paris", 0.2)]

        fake = fake_models[ROBERTA] = FakePipeline(unanswerable)
        result, html, _ = ui.answer_question(CONTEXT, "Who painted the Mona Lisa?", "RoBERTa SQuAD")
        assert fake.calls[0]["handle_impossible_answer"] is True
        assert fake.calls[0]["top_k"] == 2
        assert result["status"] == ui.NO_ANSWER
        assert result["verdict"] == "No answer in the context"
        assert "answer" not in result
        assert result["best_guess"] == "Paris"
        assert result["model"] == "deepset/roberta-base-squad2"
        assert "low confidence" in html

    def test_no_answer_without_any_span(self, fake_models):
        fake_models[ROBERTA] = FakePipeline(lambda *_a, **_k: {"answer": "", "score": 0.9, "start": 0, "end": 0})
        result, html, fig = ui.answer_question(CONTEXT, "Who painted the Mona Lisa?", "RoBERTa SQuAD")
        assert result["status"] == ui.NO_ANSWER
        assert result["best_guess"] is None and result["confidence"] is None and result["start"] is None
        assert "No answer span to highlight" in html and "<mark" not in html
        assert fig.data[0].value == 0

    def test_long_context_is_passed_whole(self, fake_models, recwarn):
        context = ("Filler sentence about nothing in particular. " * 340) + "The treasure is in Lisbon."
        assert 15_000 < len(context) < ui.MAX_CONTEXT_CHARS
        fake = fake_models[DISTIL] = FakePipeline(finds("Lisbon"))
        result, html, _ = ui.answer_question(context, "Where is the treasure?", ui.MODELS.default)
        assert fake.calls[0]["inputs"]["context"] == context
        assert result["answer"] == "Lisbon" and result["context_truncated"] is False
        assert not [w for w in recwarn if "characters long" in str(w.message)]
        assert f"of {len(context):,}" in html and len(html) < 3_000  # excerpt around the answer

    def test_very_long_context_is_capped_with_warning(self, fake_models):
        context = "word " * 6_000
        fake = fake_models[DISTIL] = FakePipeline(finds("word"))
        with pytest.warns(UserWarning, match="only the first 19,99"):
            result, _, _ = ui.answer_question(context, "Which word?", ui.MODELS.default)
        sent = fake.calls[0]["inputs"]["context"]
        assert len(sent) <= ui.MAX_CONTEXT_CHARS and context.startswith(sent)
        assert result["context_truncated"] is True

    def test_long_question_is_clipped_with_warning(self, fake_models):
        fake = fake_models[DISTIL] = FakePipeline(finds("Paris"))
        question = " ".join(f"q{i}" for i in range(100))
        with pytest.warns(UserWarning, match="only the first 64 were used"):
            result, _, _ = ui.answer_question(CONTEXT, question, ui.MODELS.default)
        assert fake.calls[0]["inputs"]["question"] == " ".join(f"q{i}" for i in range(64))
        assert result["question_truncated"] is True

    def test_summed_window_scores_are_capped_at_100_percent(self, fake_models):
        fake_models[DISTIL] = FakePipeline(finds("Paris", score=1.6))
        result, _, fig = ui.answer_question(CONTEXT, "Where?", ui.MODELS.default)
        assert result["confidence"] == "100.0%" and fig.data[0].value == 100.0

    def test_inference_failure_is_reported(self, fake_models):
        def boom(*_a, **_k):
            raise RuntimeError("index out of range in self")

        fake_models[DISTIL] = FakePipeline(boom)
        with pytest.raises(gr.Error, match="Question answering failed"):
            ui.answer_question(CONTEXT, "Where?", ui.MODELS.default)


class TestBatchParsing:
    def test_sample_file_format(self):
        items = ui.parse_batch(ui.load_sample())
        assert len(items) == 6
        assert not [item for item in items if item.problem]
        assert items[0].question == "Where is the Eiffel Tower located?"
        assert items[0].context.startswith("The Eiffel Tower is located in Paris")
        assert items[0].expected == "Paris, France"
        assert items[2].expected == "1991"
        assert "Context 2" not in items[0].context

    def test_labelled_format_with_several_questions_per_context(self):
        text = (
            "Context: Ada wrote notes.\nShe lived in London.\n"
            "Question: Who wrote notes?\nQuestion: Where?\nAnswer: London"
        )
        items = ui.parse_batch(text)
        assert [item.question for item in items] == ["Who wrote notes?", "Where?"]
        assert items[1].context == "Ada wrote notes.\nShe lived in London."
        assert (items[0].expected, items[1].expected) == ("", "London")

    def test_separated_format(self):
        text = "Context A.\nQuestion A?\n---\nContext B1.\nContext B2.\nQuestion B?\n---\nOnly one line\n---\n"
        items = ui.parse_batch(text)
        assert [(item.context, item.question) for item in items[:2]] == [
            ("Context A.", "Question A?"),
            ("Context B1.\nContext B2.", "Question B?"),
        ]
        assert items[2].problem == "needs a context line and a question line"

    def test_question_without_context_is_flagged(self):
        [item] = ui.parse_batch("Question: Where is it?")
        assert item.problem == "no context before this question"

    def test_load_sample_reads_the_repo_data_file(self):
        assert ui.SAMPLE_PATH.parts[-2:] == ("data", "qa_contexts.txt")
        assert ui.load_sample() == ui.SAMPLE_PATH.read_text(encoding="utf-8")

    def test_missing_sample_is_a_friendly_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ui, "SAMPLE_PATH", tmp_path / "missing.txt")
        with pytest.raises(gr.Error, match="Loading the sample file failed"):
            ui.load_sample()


class TestBatchAnswer:
    def test_sample_batch_rows_and_summary(self, fake_models):
        def fn(inputs, **_kwargs):
            score = 0.05 if "Python" in inputs["question"] else 0.8
            return span(inputs["context"], inputs["context"].split()[1], score)

        fake_models[DISTIL] = FakePipeline(fn)
        df, summary = ui.batch_answer(ui.load_sample(), ui.MODELS.default)
        assert isinstance(df, pd.DataFrame) and list(df.columns) == ui.BATCH_COLUMNS
        assert len(df) == 6
        assert df["Answer"].tolist()[0] == "Eiffel"
        assert df["Status"].tolist().count("low confidence") == 1
        assert df["Expected"].tolist()[2] == "1991"
        assert "6 questions answered" in summary and "**answer**: 5" in summary

    def test_separated_batch_with_bad_block_and_failure(self, fake_models):
        def fn(inputs, **_kwargs):
            if inputs["question"] == "Crash?":
                raise RuntimeError("broken")
            return span(inputs["context"], "Paris", 0.9)

        fake_models[DISTIL] = FakePipeline(fn)
        text = "Paris is in France.\nWhere?\n---\nlonely line\n---\nParis again.\nCrash?"
        df, summary = ui.batch_answer(text, ui.MODELS.default)
        assert df["Status"].tolist() == ["answer", "error: needs a context line and a question line", "error: broken"]
        assert "**error**: 2" in summary

    def test_squad2_empty_answer_in_batch(self, fake_models):
        fake_models[ROBERTA] = FakePipeline(lambda *_a, **_k: [{"answer": "", "score": 0.9, "start": 0, "end": 0}])
        df, _ = ui.batch_answer("Context: Cats sleep a lot.\nQuestion: What is GDP?", "RoBERTa SQuAD")
        assert df["Status"].tolist() == ["no answer"] and df["Answer"].tolist() == [""]

    def test_empty_batch(self, fake_models):
        with pytest.raises(gr.Error, match="Please enter contexts and questions"):
            ui.batch_answer("  \n ", ui.MODELS.default)

    def test_batch_without_questions(self, fake_models):
        with pytest.raises(gr.Error, match="No questions found"):
            ui.batch_answer("---\n---", ui.MODELS.default)


def test_highlight_shows_excerpt_of_long_context():
    context = "lorem " * 1000 + "Lisbon" + " ipsum" * 1000
    start = context.index("Lisbon")
    html = ui.highlight_answer(context, start, start + 6)
    assert html.count("…") == 2
    assert "Showing characters" in html and len(html) < 2 * ui.EXCERPT_CHARS + 500


def test_examples_are_short_question_context_pairs():
    for row in ui.EXAMPLES:
        assert len(row) == 2 and all(len(cell) <= 80 for cell in row)


def test_load_sample_button_fills_the_batch_box():
    dep = wiring(ui.create_ui(), ui.load_sample)
    assert [component.label for component in dep.outputs] == ["Contexts and questions"]
    assert dep.api_visibility == "private"
