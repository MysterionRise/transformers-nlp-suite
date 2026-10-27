"""Integration tests for UI components - testing imports and basic functionality"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestTransformersPipeline:
    """Test basic transformers pipeline functionality"""

    @pytest.mark.slow
    def test_sentiment_pipeline_creation(self):
        """Test creating a sentiment analysis pipeline (slow test - downloads model)"""
        pytest.importorskip("transformers")
        from transformers import pipeline

        # Use a tiny model for testing
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        assert classifier is not None

        # Test with simple input
        result = classifier("This is great!")
        assert len(result) > 0
        assert "label" in result[0]
        assert "score" in result[0]


class TestDependencies:
    """Test that required dependencies are available"""

    def test_transformers_import(self):
        """Test that transformers library is available"""
        try:
            import transformers

            assert transformers is not None
        except ImportError:
            pytest.fail("transformers library not available")

    def test_torch_import(self):
        """Test that PyTorch is available"""
        try:
            import torch

            assert torch is not None
        except ImportError:
            pytest.fail("torch library not available")

    def test_gradio_import(self):
        """Test that Gradio is available"""
        try:
            import gradio

            assert gradio is not None
        except ImportError:
            pytest.fail("gradio library not available")

    def test_numpy_import(self):
        """Test that numpy is available"""
        try:
            import numpy

            assert numpy is not None
        except ImportError:
            pytest.fail("numpy library not available")

    def test_spacy_import(self):
        """Test that spacy is available"""
        try:
            import spacy

            assert spacy is not None
        except ImportError:
            pytest.fail("spacy library not available")
