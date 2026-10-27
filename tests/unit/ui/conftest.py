"""Fakes shared by the UI tests: a whitespace tokenizer and pipeline doubles."""

import re
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch

import pytest


class FakeTokenizer:
    """Whitespace tokenizer with the slice of the HF tokenizer API the UIs use."""

    def __init__(self, model_max_length: int = 512, special_tokens: int = 2):
        self.model_max_length = model_max_length
        self._special = special_tokens

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return self._special

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_offsets_mapping: bool = False,
        **_: Any,
    ) -> Dict[str, List]:
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        if truncation and max_length is not None:
            spans = spans[:max_length]
        encoded: Dict[str, List] = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            encoded["offset_mapping"] = spans
        return encoded


class FakePipeline:
    """Callable stand-in for a transformers pipeline."""

    def __init__(
        self,
        fn: Callable[..., Any],
        tokenizer: Optional[FakeTokenizer] = None,
        num_labels: int = 3,
        max_position_embeddings: int = 514,
    ):
        self._fn = fn
        self.calls: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer or FakeTokenizer()
        self.model = SimpleNamespace(
            config=SimpleNamespace(num_labels=num_labels, max_position_embeddings=max_position_embeddings)
        )

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        self.calls.append({"inputs": inputs, **kwargs})
        return self._fn(inputs, **kwargs)


@pytest.fixture
def fake_models():
    """
    Route ``ui.common.load_registry_model`` to fakes.

    Usage: ``fake_models["sentiment_analysis/twitter_roberta_multilingual"] = FakePipeline(...)``.
    Unregistered keys raise, so a test fails loudly if the UI loads an unexpected model.
    """
    registry: Dict[str, Any] = {}

    def loader(category: str, model_key: str) -> Any:
        key = f"{category}/{model_key}"
        if key not in registry:
            raise AssertionError(f"Unexpected model load: {key}")
        return registry[key]

    with patch("ui.common.load_registry_model", side_effect=loader):
        yield registry
