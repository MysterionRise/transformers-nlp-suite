"""
Unit tests for configuration management
"""

import pytest

from config import ModelConfig, ModelRegistry, Settings, get_model_registry, get_settings


class TestSettings:
    """Test settings functionality"""

    def test_get_settings(self):
        """Test getting settings instance"""
        settings = get_settings()
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_settings_singleton(self):
        """Test that get_settings returns the same instance"""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_default_values(self):
        """Test default setting values"""
        settings = get_settings()
        assert settings.app_name == "Transformers NLP Suite"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.max_cached_models == 3

    def test_nested_api_environment_values(self, monkeypatch):
        """Test nested API settings load from NLP_API__ environment variables"""
        import json

        from config.settings import get_settings as cached_get_settings

        cached_get_settings.cache_clear()
        monkeypatch.setenv("NLP_API__JWT_SECRET", "env-secret")
        monkeypatch.setenv(
            "NLP_API__API_KEYS",
            json.dumps(
                {
                    "env-api-key": {
                        "name": "Env User",
                        "role": "admin",
                        "rate_limit": 250,
                        "enabled": True,
                    }
                }
            ),
        )
        monkeypatch.setenv("NLP_REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("NLP_OTEL_ENABLED", "true")

        settings = cached_get_settings()

        assert settings.api.jwt_secret == "env-secret"
        assert settings.api.api_keys["env-api-key"]["name"] == "Env User"
        assert settings.api.api_keys["env-api-key"]["rate_limit"] == 250
        assert settings.redis_url == "redis://localhost:6379/0"
        assert settings.otel_enabled is True

        cached_get_settings.cache_clear()


class TestModelRegistry:
    """Test model registry functionality"""

    def test_get_model_registry(self):
        """Test getting model registry instance"""
        registry = get_model_registry()
        assert registry is not None
        assert isinstance(registry, ModelRegistry)

    def test_registry_singleton(self):
        """Test that get_model_registry returns the same instance"""
        registry1 = get_model_registry()
        registry2 = get_model_registry()
        assert registry1 is registry2

    def test_list_categories(self):
        """Test listing model categories"""
        registry = get_model_registry()
        categories = registry.list_categories()

        assert isinstance(categories, list)
        assert len(categories) > 0
        assert "sentiment_analysis" in categories
        assert "summarization" in categories
        assert "embeddings" in categories
        assert "ner" in categories

    def test_get_category(self):
        """Test getting models in a category"""
        registry = get_model_registry()
        sentiment_models = registry.get_category("sentiment_analysis")

        assert isinstance(sentiment_models, dict)
        assert len(sentiment_models) > 0
        assert "twitter_roberta_multilingual" in sentiment_models

    def test_get_model(self):
        """Test getting a specific model"""
        registry = get_model_registry()
        model_config = registry.get_model("sentiment_analysis", "twitter_roberta_multilingual")

        assert isinstance(model_config, ModelConfig)
        assert model_config.name == "Twitter RoBERTa (Multilingual)"
        assert model_config.model_id == "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        assert model_config.task == "sentiment-analysis"

    def test_get_model_names(self):
        """Test getting model name mappings"""
        registry = get_model_registry()
        names = registry.get_model_names("sentiment_analysis")

        assert isinstance(names, dict)
        assert len(names) > 0
        assert "Twitter RoBERTa (Multilingual)" in names
        assert names["Twitter RoBERTa (Multilingual)"] == "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    def test_search_models(self):
        """Test searching for models"""
        registry = get_model_registry()
        results = registry.search_models("roberta")

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 3 for r in results)

    def test_invalid_category(self):
        """Test handling of invalid category"""
        registry = get_model_registry()
        with pytest.raises(KeyError):
            registry.get_category("nonexistent_category")

    def test_invalid_model(self):
        """Test handling of invalid model"""
        registry = get_model_registry()
        with pytest.raises(KeyError):
            registry.get_model("sentiment_analysis", "nonexistent_model")


class TestModelConfig:
    """Test ModelConfig functionality"""

    def test_model_config_creation(self):
        """Test creating a ModelConfig"""
        config = ModelConfig(
            name="Test Model",
            model_id="test/model",
            task="text-classification",
            description="A test model",
        )

        assert config.name == "Test Model"
        assert config.model_id == "test/model"
        assert config.task == "text-classification"
        assert config.description == "A test model"

    def test_model_config_optional_fields(self):
        """Test ModelConfig with optional fields"""
        config = ModelConfig(
            name="Test Model",
            model_id="test/model",
            task="text-classification",
            max_length=512,
            min_length=10,
            languages=["en", "es"],
        )

        assert config.max_length == 512
        assert config.min_length == 10
        assert config.languages == ["en", "es"]
