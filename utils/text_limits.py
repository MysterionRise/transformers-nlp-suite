"""
Token-limit helpers shared by the API and the UIs.

Many checkpoints ship a tokenizer without ``model_max_length`` (Hugging Face
then reports a huge sentinel) so ``truncation=True`` silently does nothing and
long inputs crash inside the model with "index out of range in self". These
helpers derive the real limit from the model config and cut text on a token
boundary before it reaches the model.
"""

from typing import Any, Tuple

# RoBERTa-style models offset position ids by padding_idx + 1, so two of their
# max_position_embeddings slots (514 for base models) are not usable.
_PADDING_OFFSET_MODEL_TYPES = {
    "roberta",
    "xlm-roberta",
    "xlm-roberta-xl",
    "camembert",
    "mpnet",
    "data2vec-text",
    "longformer",
}
_SENTINEL = 100_000


def token_limit(model: Any, default: int = 512) -> int:
    """
    Usable input tokens (excluding special tokens) for a pipeline-like object
    exposing ``.tokenizer`` and ``.model.config``.
    """
    tokenizer = getattr(model, "tokenizer", None)
    limits = []
    max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_length, int) and 0 < max_length < _SENTINEL:
        limits.append(max_length)
    config = getattr(getattr(model, "model", None), "config", None)
    positions = getattr(config, "max_position_embeddings", None)
    if isinstance(positions, int) and positions > 0:
        if getattr(config, "model_type", None) in _PADDING_OFFSET_MODEL_TYPES:
            pad = getattr(config, "pad_token_id", 1)
            positions -= (pad if isinstance(pad, int) else 1) + 1
        limits.append(positions)
    max_seq_length = getattr(model, "max_seq_length", None)  # SentenceTransformer models
    if isinstance(max_seq_length, int) and max_seq_length > 0:
        limits.append(max_seq_length)
    limit = min(limits) if limits else default
    special = 2
    if tokenizer is not None and hasattr(tokenizer, "num_special_tokens_to_add"):
        try:
            special = int(tokenizer.num_special_tokens_to_add())
        except Exception:
            special = 2
    return max(1, limit - special)


def count_tokens(tokenizer: Any, text: str) -> int:
    """Number of tokens in `text`, without special tokens."""
    return len(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])


def truncate_to_tokens(text: str, tokenizer: Any, max_tokens: int) -> Tuple[str, int]:
    """
    Longest prefix of `text` that fits in `max_tokens` tokens, plus the original count.

    Cuts on a token boundary using the offset mapping, so character offsets into
    the returned prefix stay valid. Falls back to a proportional character cut
    for tokenizers without offsets.
    """
    if tokenizer is None or not text:
        return text, 0
    count = count_tokens(tokenizer, text)
    if count <= max_tokens:
        return text, count
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
            return_offsets_mapping=True,
        )
        end = int(encoded["offset_mapping"][-1][1])
    except Exception:
        end = int(len(text) * max_tokens / count)
    return text[:end], count
