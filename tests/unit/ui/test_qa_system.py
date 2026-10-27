"""Tests for Question Answering (qa_system.py). Rewrite against ui/common.py when migrating."""

from unittest.mock import MagicMock, patch

import pytest


class TestQASystem:
    """Test the Question Answering System UI"""

    @patch("ui.qa_system.pipeline")
    def test_load_model_success(self, mock_pipeline):
        """Test successful model loading for QA"""
        from ui.qa_system import load_model, model_cache

        mock_pipeline.return_value = MagicMock()

        # Clear cache to force fresh load
        model_cache.clear()

        model = load_model("DistilBERT SQuAD")
        assert model is not None
        mock_pipeline.assert_called_once()

    @patch("ui.qa_system.pipeline")
    def test_load_model_caching(self, mock_pipeline):
        """Test that models are properly cached"""
        from ui.qa_system import load_model, model_cache

        mock_pipeline.return_value = MagicMock()

        # Clear cache
        model_cache.clear()

        # Load twice, should only call pipeline once
        load_model("DistilBERT SQuAD")
        load_model("DistilBERT SQuAD")

        assert mock_pipeline.call_count == 1
        assert "DistilBERT SQuAD" in model_cache

    def test_answer_question_empty_context(self):
        """Test handling of empty context"""
        from ui.qa_system import answer_question

        result, _ = answer_question("", "What is AI?", "DistilBERT SQuAD")
        assert "error" in result

    def test_answer_question_empty_question(self):
        """Test handling of empty question"""
        from ui.qa_system import answer_question

        context = "Artificial intelligence is a field of computer science."
        result, _ = answer_question(context, "", "DistilBERT SQuAD")
        assert "error" in result

    @patch("ui.qa_system.load_model")
    def test_answer_question_success(self, mock_load):
        """Test successful question answering"""
        from ui.qa_system import answer_question

        mock_model = MagicMock()
        mock_model.return_value = {
            "answer": "Paris",
            "score": 0.95,
            "start": 20,
            "end": 25,
        }
        mock_load.return_value = mock_model

        context = "The Eiffel Tower is located in Paris, France."
        question = "Where is the Eiffel Tower?"

        result, html, fig = answer_question(context, question, "DistilBERT SQuAD")

        assert "Answer" in result
        assert result["Answer"] == "Paris"
        assert "95.00%" in result["Confidence"]
        assert html is not None

    def test_highlight_answer(self):
        """Test answer highlighting HTML generation"""
        from ui.qa_system import highlight_answer

        context = "The Eiffel Tower is in Paris"
        answer = "Paris"
        start = context.index(answer)
        end = start + len(answer)
        html = highlight_answer(context, answer, start, end)

        assert "mark" in html
        assert "Paris" in html
        assert "FFD700" in html  # Gold color
