"""Tests for Zero-Shot Classification (zero_shot_classifier.py). Rewrite against ui/common.py when migrating."""

from unittest.mock import MagicMock, patch

import pytest


class TestZeroShotClassifier:
    """Test the Zero-Shot Classification UI"""

    @patch("ui.zero_shot_classifier.pipeline")
    def test_load_model_success(self, mock_pipeline):
        """Test successful zero-shot model loading"""
        from ui.zero_shot_classifier import load_model, model_cache

        mock_pipeline.return_value = MagicMock()

        # Clear cache to force fresh load
        model_cache.clear()

        model = load_model("BART Large MNLI")
        assert model is not None
        mock_pipeline.assert_called_once()

    def test_classify_text_empty_text(self):
        """Test handling of empty text"""
        from ui.zero_shot_classifier import classify_text

        result, _ = classify_text("", "positive, negative", "BART Large MNLI")
        assert "error" in result

    def test_classify_text_empty_labels(self):
        """Test handling of empty labels"""
        from ui.zero_shot_classifier import classify_text

        result, _ = classify_text("This is great!", "", "BART Large MNLI")
        assert "error" in result

    @patch("ui.zero_shot_classifier.load_model")
    def test_classify_text_success(self, mock_load):
        """Test successful text classification"""
        from ui.zero_shot_classifier import classify_text

        mock_model = MagicMock()
        mock_model.return_value = {
            "labels": ["positive", "negative", "neutral"],
            "scores": [0.9, 0.08, 0.02],
        }
        mock_load.return_value = mock_model

        result, fig = classify_text(
            "This movie was great!",
            "positive, negative, neutral",
            "BART Large MNLI",
        )

        assert "Top Classification" in result
        assert result["Top Classification"] == "positive"
        assert "90.00%" in result["Confidence"]
        assert "All Scores" in result

    @patch("ui.zero_shot_classifier.load_model")
    def test_batch_classify(self, mock_load):
        """Test batch classification"""
        from ui.zero_shot_classifier import batch_classify

        mock_model = MagicMock()
        mock_model.return_value = {
            "labels": ["positive", "negative"],
            "scores": [0.85, 0.15],
        }
        mock_load.return_value = mock_model

        texts = "Great!\nTerrible."
        result = batch_classify(texts, "positive, negative", "BART Large MNLI")

        assert "1. Text:" in result
        assert "2. Text:" in result
