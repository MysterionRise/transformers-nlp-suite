"""Tests for Text Generation (generation_playground.py). Rewrite against ui/common.py when migrating."""

from unittest.mock import MagicMock, patch

import pytest


class TestGenerationPlayground:
    """Test the Text Generation Playground UI"""

    @patch("ui.generation_playground.pipeline")
    def test_load_model_success(self, mock_pipeline):
        """Test successful generation model loading"""
        from ui.generation_playground import load_model, model_cache

        mock_pipeline.return_value = MagicMock()

        # Clear cache to force fresh load
        model_cache.clear()

        model = load_model("GPT-2")
        assert model is not None
        mock_pipeline.assert_called_once()

    def test_generate_text_empty_prompt(self):
        """Test handling of empty prompt"""
        from ui.generation_playground import generate_text

        result = generate_text("", "GPT-2")
        assert "error" in result

    @patch("ui.generation_playground.load_model")
    def test_generate_text_success(self, mock_load):
        """Test successful text generation"""
        from ui.generation_playground import generate_text

        mock_model = MagicMock()
        mock_model.return_value = [{"generated_text": "The future of AI is bright and promising"}]
        mock_load.return_value = mock_model

        result = generate_text("The future of AI", "GPT-2", max_length=50, temperature=0.7)

        assert "error" not in result
        assert "Prompt" in result
        assert "Samples" in result
        assert len(result["Samples"]) > 0

    @patch("ui.generation_playground.load_model")
    def test_batch_generate(self, mock_load):
        """Test batch text generation"""
        from ui.generation_playground import batch_generate

        mock_model = MagicMock()
        mock_model.return_value = [{"generated_text": "Generated text here"}]
        mock_load.return_value = mock_model

        prompts = "Once upon a time\nIn a galaxy far away"
        result = batch_generate(prompts, "GPT-2")

        assert "1. Prompt:" in result
        assert "2. Prompt:" in result
        assert "Generated:" in result
