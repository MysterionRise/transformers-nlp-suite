"""
Settings and configuration management for NLP Transformers Examples

Uses Pydantic for settings validation and YAML for model configurations.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class APIKeyConfig(BaseModel):
    """Configuration for a single API key"""

    name: str
    role: str = "user"  # admin, user, readonly
    rate_limit: int = 100  # requests per minute
    enabled: bool = True


class APISettings(BaseModel):
    """
    API-specific settings for authentication and rate limiting
    """

    # JWT settings
    jwt_secret: str = Field(
        default="change-this-secret-in-production",
        description="Secret key for JWT token signing. MUST be changed in production.",
    )
    jwt_algorithm: str = Field(default="HS256", description="Algorithm for JWT encoding")
    jwt_expiration_hours: int = Field(default=24, description="JWT token expiration time in hours")

    # API Keys - dict of api_key -> config
    api_keys: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "dev-api-key": {
                "name": "Development",
                "role": "admin",
                "rate_limit": 1000,
                "enabled": True,
            },
            "demo-api-key": {
                "name": "Demo User",
                "role": "user",
                "rate_limit": 100,
                "enabled": True,
            },
        },
        description="API keys with their configurations",
    )

    # Rate limiting defaults
    default_rate_limit: int = Field(default=100, description="Default requests per minute")
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window in seconds")

    # CORS settings
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins",
    )


class ModelConfig(BaseModel):
    """Configuration for a single model"""

    name: str
    model_id: str
    task: str
    description: str = ""
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    dimensions: Optional[int] = None
    languages: Optional[List[str]] = None
    framework: Optional[str] = "transformers"
    prefix: Optional[str] = None
    template: Optional[bool] = False


class Settings(BaseSettings):
    """
    Application settings with environment variable support

    Environment variables can override these settings with the prefix NLP_
    Example: NLP_DEBUG=true, NLP_CACHE_DIR=/custom/path
    """

    # Application settings
    app_name: str = "NLP Transformers Examples"
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Model settings
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "huggingface",
        description="Directory for caching models",
    )
    max_cached_models: int = Field(default=3, description="Maximum number of models to keep in memory")
    device: str = Field(default="auto", description="Device to run models on (auto, cpu, cuda, mps)")

    # UI settings
    ui_theme: str = Field(default="default", description="Gradio theme")
    share: bool = Field(default=False, description="Create public Gradio links")
    server_name: str = Field(default="127.0.0.1", description="Server host")
    server_port: int = Field(default=7860, description="Starting port for UIs")

    # Performance settings
    max_workers: int = Field(default=4, description="Maximum number of workers for parallel processing")
    batch_size: int = Field(default=8, description="Default batch size for processing")

    # Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent, description="Project root directory"
    )
    data_dir: Path = Field(default_factory=lambda: Path("data"), description="Data directory")
    logs_dir: Path = Field(default_factory=lambda: Path("logs"), description="Logs directory")

    # API settings
    api: APISettings = Field(default_factory=APISettings, description="API-specific settings")

    # API server settings
    api_host: str = Field(default="127.0.0.1", description="API server host")
    api_port: int = Field(default=8000, description="API server port")

    class Config:
        env_prefix = "NLP_"
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


class ModelRegistry:
    """
    Registry for managing model configurations

    Loads model definitions from YAML and provides easy access.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize model registry

        Args:
            config_path: Path to models.yaml file. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "models.yaml"

        self.config_path = config_path
        self._models: Dict[str, Dict[str, ModelConfig]] = {}
        self._load_models()

    def _load_models(self):
        """Load model configurations from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Model configuration not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        # Parse each category
        for category, models in data.items():
            self._models[category] = {}
            for model_key, model_data in models.items():
                self._models[category][model_key] = ModelConfig(**model_data)

    def get_model(self, category: str, model_key: str) -> ModelConfig:
        """
        Get a specific model configuration

        Args:
            category: Model category (e.g., 'sentiment_analysis', 'summarization')
            model_key: Model key within category

        Returns:
            ModelConfig object

        Raises:
            KeyError: If category or model not found
        """
        if category not in self._models:
            raise KeyError(f"Unknown model category: {category}. Available: {list(self._models.keys())}")

        if model_key not in self._models[category]:
            raise KeyError(
                f"Unknown model key '{model_key}' in category '{category}'. "
                f"Available: {list(self._models[category].keys())}"
            )

        return self._models[category][model_key]

    def get_category(self, category: str) -> Dict[str, ModelConfig]:
        """
        Get all models in a category

        Args:
            category: Model category

        Returns:
            Dictionary of model_key -> ModelConfig
        """
        if category not in self._models:
            raise KeyError(f"Unknown model category: {category}")

        return self._models[category]

    def list_categories(self) -> List[str]:
        """List all available model categories"""
        return list(self._models.keys())

    def list_models(self, category: str) -> List[str]:
        """
        List all model keys in a category

        Args:
            category: Model category

        Returns:
            List of model keys
        """
        if category not in self._models:
            raise KeyError(f"Unknown model category: {category}")

        return list(self._models[category].keys())

    def get_model_names(self, category: str) -> Dict[str, str]:
        """
        Get mapping of display names to model IDs for a category

        Args:
            category: Model category

        Returns:
            Dictionary of {display_name: model_id}
        """
        if category not in self._models:
            raise KeyError(f"Unknown model category: {category}")

        return {config.name: config.model_id for config in self._models[category].values()}

    def search_models(self, query: str) -> List[tuple]:
        """
        Search for models by name or description

        Args:
            query: Search query

        Returns:
            List of (category, model_key, ModelConfig) tuples
        """
        query = query.lower()
        results = []

        for category, models in self._models.items():
            for model_key, config in models.items():
                if query in config.name.lower() or query in config.description.lower():
                    results.append((category, model_key, config))

        return results


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance

    Returns:
        Settings singleton
    """
    return Settings()


@lru_cache()
def get_model_registry() -> ModelRegistry:
    """
    Get cached model registry instance

    Returns:
        ModelRegistry singleton
    """
    return ModelRegistry()


# Convenience functions
def get_model_config(category: str, model_key: str) -> ModelConfig:
    """Get a specific model configuration"""
    return get_model_registry().get_model(category, model_key)


def get_category_models(category: str) -> Dict[str, ModelConfig]:
    """Get all models in a category"""
    return get_model_registry().get_category(category)
