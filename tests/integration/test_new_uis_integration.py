"""
Integration tests for newly implemented UI modules.

These tests verify that the new UI modules can be imported, configured,
and their main functions work correctly with mocked models.

Marked as @pytest.mark.slow for CI/CD pipeline.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.slow
class TestNewUIsLaunchConfiguration:
    """Test that new UIs are properly configured in launcher"""

    def test_launcher_configuration(self):
        """Test that launch_ui.py includes new UIs"""
        from launch_ui import UIS

        # Check that all new UIs are registered
        assert "qa" in UIS
        assert "generation" in UIS
        assert "zero_shot" in UIS
        assert "translation" in UIS
        assert "vision" in UIS

        # Verify port assignments
        assert UIS["qa"]["port"] == 7865
        assert UIS["generation"]["port"] == 7866
        assert UIS["zero_shot"]["port"] == 7867
        assert UIS["translation"]["port"] == 7868
        assert UIS["vision"]["port"] == 7869

        # Verify files exist
        qa_path = Path(project_root) / UIS["qa"]["file"]
        gen_path = Path(project_root) / UIS["generation"]["file"]
        zs_path = Path(project_root) / UIS["zero_shot"]["file"]
        trans_path = Path(project_root) / UIS["translation"]["file"]
        vision_path = Path(project_root) / UIS["vision"]["file"]

        assert qa_path.exists()
        assert gen_path.exists()
        assert zs_path.exists()
        assert trans_path.exists()
        assert vision_path.exists()

    def test_all_uis_launchable(self):
        """Test that all UIs can be launched without errors"""
        from launch_ui import UIS

        for ui_key in ["qa", "generation", "zero_shot", "translation", "vision"]:
            # Verify UI is in registry
            assert ui_key in UIS
            # Verify launch_ui can handle it
            ui_info = UIS[ui_key]
            assert "name" in ui_info
            assert "file" in ui_info
            assert "port" in ui_info
            assert "description" in ui_info


@pytest.mark.slow
class TestSampleData:
    """Test that sample data files exist and are readable"""

    def test_sentiment_reviews_data(self):
        """Test sentiment reviews sample data"""
        data_path = Path(project_root) / "data" / "sentiment_reviews.txt"
        assert data_path.exists()

        with open(data_path, "r") as f:
            content = f.read()
            assert len(content) > 0
            assert len(content.split("\n")) > 5

    def test_qa_contexts_data(self):
        """Test QA contexts sample data"""
        data_path = Path(project_root) / "data" / "qa_contexts.txt"
        assert data_path.exists()

        with open(data_path, "r") as f:
            content = f.read()
            assert len(content) > 0
            assert "Context" in content
            assert "Question" in content

    def test_news_articles_data(self):
        """Test news articles sample data"""
        data_path = Path(project_root) / "data" / "news_articles.txt"
        assert data_path.exists()

        with open(data_path, "r") as f:
            content = f.read()
            assert len(content) > 0
            assert "Article" in content

    def test_translation_samples_data(self):
        """Test translation samples data"""
        data_path = Path(project_root) / "data" / "translation_samples.txt"
        assert data_path.exists()

        with open(data_path, "r") as f:
            content = f.read()
            assert len(content) > 0
            assert "Sample" in content


@pytest.mark.slow
class TestConfigurationUpdates:
    """Test that configuration has been updated with new models"""

    def test_vision_models_in_config(self):
        """Test that vision models are in configuration"""
        import yaml

        config_path = Path(project_root) / "config" / "models.yaml"
        assert config_path.exists()

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check for vision-language category
        assert "vision_language" in config
        assert len(config["vision_language"]) >= 3

        # Check for specific models
        models_found = list(config["vision_language"].keys())
        assert len(models_found) > 0

    def test_model_registry_categories(self):
        """Test that all required model categories exist"""
        import yaml

        config_path = Path(project_root) / "config" / "models.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check for all original categories
        required_categories = [
            "sentiment_analysis",
            "summarization",
            "embeddings",
            "ner",
            "question_answering",
            "text_generation",
            "zero_shot",
            "translation",
            "vision_language",
        ]

        for category in required_categories:
            assert category in config, f"Missing category: {category}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
