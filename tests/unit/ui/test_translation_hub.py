"""Tests for Translation (translation_hub.py). Rewrite against ui/common.py when migrating."""

from unittest.mock import MagicMock, patch

import pytest


class TestTranslationHub:
    """Test the Translation Hub UI"""

    def test_translate_empty_text(self):
        """Test handling of empty text"""
        from ui.translation_hub import translate_mbart

        result = translate_mbart("", "English", "Spanish")
        assert "error" in result

    @patch("ui.translation_hub.load_mbart_model")
    def test_translate_mbart_success(self, mock_load):
        """Test successful mBART translation"""
        from ui.translation_hub import translate_mbart

        mock_model = MagicMock()
        mock_model.return_value = [{"translation_text": "Hola, ¿cómo estás?"}]
        mock_model.tokenizer = MagicMock()
        mock_load.return_value = mock_model

        result = translate_mbart("Hello, how are you?", "English", "Spanish")

        assert "error" not in result
        assert "Original" in result
        assert "Translation" in result
        assert "Model" in result

    def test_translate_helsinki_empty_text(self):
        """Test Helsinki translation with empty text"""
        from ui.translation_hub import translate_helsinki

        result = translate_helsinki("", "en-es")
        assert "error" in result

    @patch("ui.translation_hub.load_helsinki_model")
    def test_translate_helsinki_success(self, mock_load):
        """Test successful Helsinki translation"""
        from ui.translation_hub import translate_helsinki

        mock_model = MagicMock()
        mock_model.return_value = [{"translation_text": "Hola mundo"}]
        mock_load.return_value = mock_model

        result = translate_helsinki("Hello world", "en-es")

        assert "error" not in result
        assert "Translation" in result
        assert "Language Pair" in result

    @patch("ui.translation_hub.load_mbart_model")
    def test_batch_translate(self, mock_load):
        """Test batch translation"""
        from ui.translation_hub import batch_translate

        mock_model = MagicMock()
        mock_model.return_value = [{"translation_text": "Translated text"}]
        mock_model.tokenizer = MagicMock()
        mock_load.return_value = mock_model

        texts = "Hello\nWorld"
        result = batch_translate(texts, "mbart", "English", "Spanish")

        assert "1. Original:" in result
        assert "2. Original:" in result
