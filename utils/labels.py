"""Label normalisation shared by the API and the UIs."""

import re
from typing import Optional

_GENERIC = re.compile(r"^label_(\d+)$")
_NAMED = {
    "neg": "negative",
    "negative": "negative",
    "neu": "neutral",
    "neutral": "neutral",
    "pos": "positive",
    "positive": "positive",
}


def normalize_sentiment_label(label: Optional[str], num_labels: Optional[int] = None) -> str:
    """
    Map model-specific sentiment labels onto negative / neutral / positive.

    Generic ``LABEL_n`` names are interpreted by the size of the label set:
    two classes are negative/positive (SST-2 style), three are
    negative/neutral/positive (Cardiff style). Named labels are lower-cased and
    abbreviations expanded. Anything else, such as ``"4 stars"``, is returned
    lower-cased so callers can still display it.
    """
    text = (label or "unknown").strip().lower()
    match = _GENERIC.match(text)
    if match:
        index = int(match.group(1))
        if num_labels == 2:
            return {0: "negative", 1: "positive"}.get(index, text)
        if num_labels in (None, 3):
            return {0: "negative", 1: "neutral", 2: "positive"}.get(index, text)
        return text
    return _NAMED.get(text, text)


def num_labels_of(model) -> Optional[int]:
    """Number of output labels of a pipeline's model, if it can be determined."""
    config = getattr(getattr(model, "model", None), "config", None)
    value = getattr(config, "num_labels", None)
    return value if isinstance(value, int) else None
