"""Tests for utils/text_limits.py and utils/labels.py."""

from types import SimpleNamespace

import pytest

from tests.unit.ui.conftest import FakeTokenizer
from utils.labels import normalize_sentiment_label, num_labels_of
from utils.text_limits import count_tokens, token_limit, truncate_to_tokens


def _model(model_type="bert", max_positions=512, model_max_length=512, pad=0):
    return SimpleNamespace(
        tokenizer=FakeTokenizer(model_max_length=model_max_length),
        model=SimpleNamespace(
            config=SimpleNamespace(model_type=model_type, max_position_embeddings=max_positions, pad_token_id=pad)
        ),
    )


class TestTokenLimit:
    def test_bert_style(self):
        assert token_limit(_model()) == 510

    def test_roberta_padding_offset_with_sentinel_tokenizer(self):
        # cardiffnlp/twitter-xlm-roberta-base-sentiment: no model_max_length, 514 positions, pad id 1
        model = _model("xlm-roberta", 514, int(1e30), pad=1)
        assert token_limit(model) == 510

    def test_bart_style_uses_positions(self):
        assert token_limit(_model("bart", 1024, int(1e30), pad=1)) == 1022

    def test_default_when_unknown(self):
        assert token_limit(SimpleNamespace(), default=100) == 98


class TestTruncate:
    def test_counts_and_noop(self):
        tok = FakeTokenizer()
        assert count_tokens(tok, "a b c") == 3
        assert truncate_to_tokens("a b c", tok, 5) == ("a b c", 3)

    def test_cut_on_token_boundary(self):
        assert truncate_to_tokens("alpha  beta gamma delta", FakeTokenizer(), 2) == ("alpha  beta", 4)

    def test_empty_and_missing_tokenizer(self):
        assert truncate_to_tokens("", FakeTokenizer(), 2) == ("", 0)
        assert truncate_to_tokens("x y", None, 1) == ("x y", 0)


class TestSentimentLabels:
    @pytest.mark.parametrize(
        "label,num_labels,expected",
        [
            ("LABEL_0", 2, "negative"),
            ("LABEL_1", 2, "positive"),
            ("LABEL_1", 3, "neutral"),
            ("LABEL_2", None, "positive"),
            ("NEGATIVE", 2, "negative"),
            ("Neutral", 3, "neutral"),
            ("pos", None, "positive"),
            ("4 stars", 5, "4 stars"),
            ("LABEL_4", 5, "label_4"),
            (None, None, "unknown"),
        ],
    )
    def test_normalize(self, label, num_labels, expected):
        assert normalize_sentiment_label(label, num_labels) == expected

    def test_num_labels_of(self):
        assert num_labels_of(SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(num_labels=2)))) == 2
        assert num_labels_of(SimpleNamespace()) is None
