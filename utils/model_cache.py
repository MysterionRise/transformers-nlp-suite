"""
Model Cache and Optimization utilities

Provides singleton pattern for model loading, caching, and memory management.
Supports automatic device detection (CPU, CUDA, MPS) and lazy loading.
"""

import gc
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

import torch
from transformers import Pipeline, pipeline

from config.settings import ModelConfig, get_model_registry, get_settings
from utils.error_handler import ErrorContext, ModelLoadError, ModelNotFoundError, retry_on_error
from utils.logger import PerformanceLogger, get_logger
from utils.metrics import record_cache_eviction, record_cache_hit, record_cache_miss, update_cache_size

logger = get_logger(__name__)


class ModelCache:
    """
    Singleton cache for managing transformer models

    Features:
    - Lazy loading (models loaded on first use)
    - LRU eviction when cache is full
    - Automatic device detection
    - Thread-safe: cache bookkeeping is guarded by a re-entrant lock and each
      cache key has its own load lock, so concurrent requests for the same cold
      model load it once while requests for other models proceed in parallel
    - Memory cleanup
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the cache"""
        if self._initialized:
            return

        self.settings = get_settings()
        self.model_registry = get_model_registry()

        # OrderedDict for LRU cache
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = self.settings.max_cached_models
        self._mutex = threading.RLock()
        self._load_locks: Dict[str, threading.Lock] = {}
        self._device = self._detect_device()

        logger.info(f"ModelCache initialized with max_size={self._max_size}, device={self._device}")
        self._initialized = True

    def _detect_device(self) -> str:
        """
        Detect the best available device for model inference

        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if self.settings.device != "auto":
            logger.info(f"Using configured device: {self.settings.device}")
            return self.settings.device

        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available - using GPU: {gpu_name}")
            return device

        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple MPS available - using Metal GPU")
            return device

        # Fallback to CPU
        device = "cpu"
        logger.info("No GPU available - using CPU")
        return device

    def _generate_cache_key(self, category: str, model_key: str, task_override: Optional[str] = None) -> str:
        """Generate unique cache key for a model"""
        task = task_override or f"{category}_{model_key}"
        return f"{category}::{model_key}::{task}"

    @retry_on_error(max_retries=2, delay=1.0)
    def _load_model(self, model_config: ModelConfig, task: Optional[str] = None) -> Pipeline:
        """
        Load a model from HuggingFace

        Args:
            model_config: Model configuration
            task: Optional task override

        Returns:
            Loaded pipeline

        Raises:
            ModelLoadError: If model fails to load
        """
        task = task or model_config.task
        model_id = model_config.model_id

        logger.info(f"Loading model: {model_config.name} ({model_id}) for task: {task}")

        try:
            with PerformanceLogger(f"load_model_{model_id}", logger=logger):
                pipeline_kwargs: Dict[str, Any] = {"task": task, "model": model_id, "device": self._pipeline_device()}
                if model_config.revision:
                    pipeline_kwargs["revision"] = model_config.revision
                if self.settings.cache_dir:
                    pipeline_kwargs["model_kwargs"] = {"cache_dir": str(self.settings.cache_dir)}

                model = pipeline(**pipeline_kwargs)

                logger.info(f"Successfully loaded: {model_config.name}")
                return model

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise ModelLoadError(model_id, original_error=e)

    @retry_on_error(max_retries=2, delay=1.0)
    def _load_sentence_transformer(self, model_config: ModelConfig) -> Any:
        """Load a sentence-transformers model."""
        model_id = model_config.model_id
        logger.info(f"Loading sentence transformer: {model_config.name} ({model_id})")

        try:
            with PerformanceLogger(f"load_sentence_transformer_{model_id}", logger=logger):
                from sentence_transformers import SentenceTransformer

                cache_folder = str(self.settings.cache_dir) if self.settings.cache_dir else None
                model = SentenceTransformer(
                    model_id,
                    cache_folder=cache_folder,
                    revision=model_config.revision,
                    device=self._device,
                )
                logger.info(f"Successfully loaded sentence transformer: {model_config.name}")
                return model
        except Exception as e:
            logger.error(f"Failed to load sentence transformer {model_id}: {str(e)}")
            raise ModelLoadError(model_id, original_error=e)

    @retry_on_error(max_retries=1, delay=1.0)
    def _load_spacy_model(self, model_config: ModelConfig) -> Any:
        """Load a spaCy model from the registry."""
        model_id = model_config.model_id
        logger.info(f"Loading spaCy model: {model_config.name} ({model_id})")

        try:
            with PerformanceLogger(f"load_spacy_{model_id}", logger=logger):
                import spacy

                model = spacy.load(model_id)
                logger.info(f"Successfully loaded spaCy model: {model_config.name}")
                return model
        except Exception as e:
            logger.error(f"Failed to load spaCy model {model_id}: {str(e)}")
            raise ModelLoadError(model_id, original_error=e)

    @retry_on_error(max_retries=1, delay=1.0)
    def _load_gliner_model(self, model_config: ModelConfig) -> Any:
        """Load a GLiNER zero-shot NER model."""
        model_id = model_config.model_id
        logger.info(f"Loading GLiNER model: {model_config.name} ({model_id})")

        try:
            with PerformanceLogger(f"load_gliner_{model_id}", logger=logger):
                from gliner import GLiNER

                kwargs: Dict[str, Any] = {}
                if self.settings.cache_dir:
                    kwargs["cache_dir"] = str(self.settings.cache_dir)
                model = GLiNER.from_pretrained(model_id, **kwargs)
                logger.info(f"Successfully loaded GLiNER model: {model_config.name}")
                return model
        except Exception as e:
            logger.error(f"Failed to load GLiNER model {model_id}: {str(e)}")
            raise ModelLoadError(model_id, original_error=e)

    def _pipeline_device(self) -> Optional[str]:
        """Device argument for transformers pipelines (None means CPU)."""
        return None if self._device == "cpu" else self._device

    def _key_lock(self, cache_key: str) -> threading.Lock:
        with self._mutex:
            lock = self._load_locks.get(cache_key)
            if lock is None:
                lock = self._load_locks[cache_key] = threading.Lock()
            return lock

    def _lookup(self, cache_key: str) -> Optional[Any]:
        """Return a cached model and mark it most recently used, or None."""
        with self._mutex:
            if cache_key not in self._cache:
                return None
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

    def _get_or_load(
        self,
        cache_key: str,
        resolve_config: Callable[[], ModelConfig],
        loader: Callable[[ModelConfig], Any],
        context: Dict[str, Any],
        force_reload: bool = False,
    ) -> Any:
        """
        Shared lookup/load path for every model family.

        Evicts before loading so peak memory stays near the configured cache size,
        and holds a per-key lock while loading so concurrent cold requests for the
        same model wait for one load instead of each loading their own copy.
        """
        if not force_reload:
            model = self._lookup(cache_key)
            if model is not None:
                logger.debug(f"Cache hit: {cache_key}")
                record_cache_hit(cache_key)
                return model

        with self._key_lock(cache_key):
            if not force_reload:
                model = self._lookup(cache_key)
                if model is not None:
                    logger.debug(f"Cache hit after waiting for load: {cache_key}")
                    record_cache_hit(cache_key)
                    return model

            logger.debug(f"Cache miss: {cache_key}")
            record_cache_miss(cache_key)
            model_config = resolve_config()

            with self._mutex:
                self._cache.pop(cache_key, None)
                while self._cache and len(self._cache) >= self._max_size:
                    self._evict_lru()

            with ErrorContext(context.pop("operation", "model_loading"), **context):
                model = loader(model_config)

            with self._mutex:
                while self._cache and len(self._cache) >= self._max_size:
                    self._evict_lru()
                self._cache[cache_key] = model
                size = len(self._cache)
            logger.info(f"Added to cache: {cache_key} (cache size: {size}/{self._max_size})")
            update_cache_size(size)
            return model

    def _registry_config(self, category: str, model_key: str) -> Callable[[], ModelConfig]:
        def resolve() -> ModelConfig:
            try:
                return self.model_registry.get_model(category, model_key)
            except KeyError as e:
                raise ModelNotFoundError(model_key, category) from e

        return resolve

    def _evict_lru(self):
        """Evict the least recently used model from cache"""
        if not self._cache:
            return

        # Remove oldest item (first in OrderedDict)
        cache_key, model = self._cache.popitem(last=False)
        logger.info(f"Evicting model from cache: {cache_key}")
        record_cache_eviction()

        # Clean up memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        update_cache_size(len(self._cache))

    def get_model(
        self,
        category: str,
        model_key: str,
        task_override: Optional[str] = None,
        force_reload: bool = False,
    ) -> Pipeline:
        """
        Get a model from cache or load it

        Args:
            category: Model category (e.g., 'sentiment_analysis')
            model_key: Model key within category
            task_override: Optional task type override
            force_reload: Force reload even if cached

        Returns:
            Loaded model pipeline

        Raises:
            ModelNotFoundError: If model not found in registry
            ModelLoadError: If model fails to load
        """
        cache_key = self._generate_cache_key(category, model_key, task_override)
        return self._get_or_load(
            cache_key,
            self._registry_config(category, model_key),
            lambda config: self._load_model(config, task=task_override),
            {"operation": "model_loading", "category": category, "model_key": model_key},
            force_reload=force_reload,
        )

    def get_sentence_transformer(
        self,
        category: str,
        model_key: str,
        force_reload: bool = False,
    ) -> Any:
        """
        Get a sentence-transformers model from cache or load it.

        Args:
            category: Registry category, usually 'embeddings'
            model_key: Model key within category
            force_reload: Force reload even if cached

        Returns:
            Loaded SentenceTransformer model
        """
        cache_key = f"sentence-transformers::{category}::{model_key}"
        return self._get_or_load(
            cache_key,
            self._registry_config(category, model_key),
            self._load_sentence_transformer,
            {"operation": "sentence_transformer_loading", "category": category, "model_key": model_key},
            force_reload=force_reload,
        )

    def get_spacy_model(
        self,
        model_key: str,
        force_reload: bool = False,
    ) -> Any:
        """
        Get a spaCy model from cache or load it from the NER registry.

        Args:
            model_key: Model key within the 'ner' category
            force_reload: Force reload even if cached

        Returns:
            Loaded spaCy language pipeline
        """
        cache_key = f"spacy::ner::{model_key}"
        return self._get_or_load(
            cache_key,
            self._registry_config("ner", model_key),
            self._load_spacy_model,
            {"operation": "spacy_model_loading", "category": "ner", "model_key": model_key},
            force_reload=force_reload,
        )

    def get_registry_model(self, category: str, model_key: str, force_reload: bool = False) -> Any:
        """
        Load any registry model, dispatching on its ``framework`` field.

        ``transformers`` (default) returns a pipeline for the configured task,
        ``sentence-transformers`` a SentenceTransformer, ``spacy`` a spaCy
        Language and ``gliner`` a GLiNER model. This is the entry point the
        Gradio UIs use, so every tool shares one LRU cache.
        """
        try:
            framework = self.model_registry.get_model(category, model_key).framework or "transformers"
        except KeyError as e:
            raise ModelNotFoundError(model_key, category) from e

        if framework == "sentence-transformers":
            return self.get_sentence_transformer(category, model_key, force_reload=force_reload)
        if framework == "spacy":
            cache_key = f"spacy::{category}::{model_key}"
            loader = self._load_spacy_model
        elif framework == "gliner":
            cache_key = f"gliner::{category}::{model_key}"
            loader = self._load_gliner_model
        else:
            return self.get_model(category, model_key, force_reload=force_reload)

        return self._get_or_load(
            cache_key,
            self._registry_config(category, model_key),
            loader,
            {"operation": f"{framework}_model_loading", "category": category, "model_key": model_key},
            force_reload=force_reload,
        )

    def get_model_by_id(self, model_id: str, task: str, force_reload: bool = False) -> Pipeline:
        """
        Get a model by HuggingFace model ID (bypass registry)

        Args:
            model_id: HuggingFace model ID
            task: Task type
            force_reload: Force reload even if cached

        Returns:
            Loaded model pipeline
        """
        cache_key = f"direct::{model_id}::{task}"

        def resolve() -> ModelConfig:
            return ModelConfig(name=model_id, model_id=model_id, task=task)

        def loader(config: ModelConfig) -> Any:
            logger.info(f"Loading model by ID: {model_id} (task: {task})")
            try:
                return pipeline(task=task, model=model_id, device=self._pipeline_device())
            except Exception as e:
                raise ModelLoadError(model_id, original_error=e)

        return self._get_or_load(
            cache_key,
            resolve,
            loader,
            {"operation": "model_loading", "model_id": model_id},
            force_reload=force_reload,
        )

    def clear_cache(self):
        """Clear all cached models"""
        with self._mutex:
            logger.info(f"Clearing model cache ({len(self._cache)} models)")
            self._cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        update_cache_size(0)

    def remove_model(self, category: str, model_key: str):
        """Remove a specific model from cache"""
        cache_key = self._generate_cache_key(category, model_key)
        with self._mutex:
            removed = self._cache.pop(cache_key, None) is not None
        if removed:
            logger.info(f"Removed from cache: {cache_key}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            update_cache_size(len(self._cache))

    def warmup(self, models: list[tuple[str, str]]):
        """
        Pre-load models into cache

        Args:
            models: List of (category, model_key) tuples to preload
        """
        logger.info(f"Warming up cache with {len(models)} models")
        for category, model_key in models:
            try:
                self.get_model(category, model_key)
            except Exception as e:
                logger.warning(f"Failed to warmup {category}::{model_key}: {str(e)}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics
        """
        with self._mutex:
            return {
                "cached_models": len(self._cache),
                "max_size": self._max_size,
                "device": self._device,
                "models": list(self._cache.keys()),
            }

    @property
    def device(self) -> str:
        """Get current device"""
        return self._device


# Singleton instance getter
_model_cache_instance: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """
    Get the global model cache instance

    Returns:
        ModelCache singleton
    """
    global _model_cache_instance
    if _model_cache_instance is None:
        _model_cache_instance = ModelCache()
    return _model_cache_instance


# Convenience functions
def load_model(category: str, model_key: str, **kwargs) -> Pipeline:
    """
    Load a model from the cache

    Args:
        category: Model category
        model_key: Model key
        **kwargs: Additional arguments passed to get_model

    Returns:
        Loaded model pipeline
    """
    cache = get_model_cache()
    return cache.get_model(category, model_key, **kwargs)


def load_model_by_id(model_id: str, task: str, **kwargs) -> Pipeline:
    """
    Load a model by HuggingFace ID

    Args:
        model_id: HuggingFace model ID
        task: Task type
        **kwargs: Additional arguments

    Returns:
        Loaded model pipeline
    """
    cache = get_model_cache()
    return cache.get_model_by_id(model_id, task, **kwargs)


def load_sentence_transformer(category: str, model_key: str, **kwargs) -> Any:
    """
    Load a sentence-transformers model from the cache.

    Args:
        category: Registry category
        model_key: Model key
        **kwargs: Additional arguments passed to get_sentence_transformer

    Returns:
        Loaded SentenceTransformer model
    """
    cache = get_model_cache()
    return cache.get_sentence_transformer(category, model_key, **kwargs)


def load_registry_model(category: str, model_key: str, **kwargs) -> Any:
    """Load any registry model through the shared cache (see ModelCache.get_registry_model)."""
    return get_model_cache().get_registry_model(category, model_key, **kwargs)


def load_spacy_model(model_key: str, **kwargs) -> Any:
    """
    Load a spaCy model from the cache.

    Args:
        model_key: NER registry model key
        **kwargs: Additional arguments passed to get_spacy_model

    Returns:
        Loaded spaCy pipeline
    """
    cache = get_model_cache()
    return cache.get_spacy_model(model_key, **kwargs)


def clear_model_cache():
    """Clear all cached models"""
    cache = get_model_cache()
    cache.clear_cache()


def get_cache_info() -> Dict[str, Any]:
    """Get cache statistics"""
    cache = get_model_cache()
    return cache.get_cache_stats()
