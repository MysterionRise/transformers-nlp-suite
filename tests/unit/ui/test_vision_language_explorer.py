"""Tests for Vision-Language (vision_language_explorer.py). Rewrite against ui/common.py when migrating."""

from unittest.mock import MagicMock, patch

import pytest


class TestVisionLanguageExplorer:
    """Test the Vision-Language Explorer UI"""

    def test_generate_caption_no_image(self):
        """Test caption generation with no image"""
        from ui.vision_language_explorer import generate_caption

        result = generate_caption(None)
        assert "error" in result

    @patch("ui.vision_language_explorer.load_image_captioning_model")
    def test_generate_caption_success(self, mock_load):
        """Test successful image caption generation"""
        from PIL import Image

        from ui.vision_language_explorer import generate_caption

        # Create a dummy image
        img = Image.new("RGB", (100, 100), color="red")

        mock_model = MagicMock()
        mock_processor = MagicMock()
        # Return an object with .pixel_values attribute (not a plain dict)
        mock_processor_output = MagicMock()
        mock_processor_output.pixel_values = MagicMock()
        mock_processor.return_value = mock_processor_output
        mock_processor.batch_decode = MagicMock(return_value=["a red square on white background"])
        mock_model.generate = MagicMock(return_value=MagicMock())

        mock_load.return_value = (mock_model, mock_processor)

        result = generate_caption(img)

        assert "error" not in result
        assert "Caption" in result
        assert "Model" in result
        assert "100x100" in result["Image Size"]

    def test_calculate_clip_similarity_no_image(self):
        """Test similarity calculation with no image"""
        from ui.vision_language_explorer import calculate_clip_similarity

        result, html = calculate_clip_similarity(None, ["a cat"])
        assert "error" in result

    def test_calculate_clip_similarity_no_texts(self):
        """Test similarity calculation with no texts"""
        from PIL import Image

        from ui.vision_language_explorer import calculate_clip_similarity

        img = Image.new("RGB", (100, 100))
        result, html = calculate_clip_similarity(img, [])
        assert "error" in result

    def test_batch_caption_no_images(self):
        """Test batch captioning with no images"""
        from ui.vision_language_explorer import batch_caption_images

        result = batch_caption_images([])
        assert "Please upload" in result or "No" in result


# Import verification tests
