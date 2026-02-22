"""
Unit tests for the tokenizer.

NOTE: These tests require a trained tokenizer model at data/tokenizer.model.
Run `python scripts/train_tokenizer.py` first, or skip these tests if
the tokenizer hasn't been trained yet.

Tests verify:
  1. Encode/decode roundtrip preserves text
  2. Byte-fallback handles non-ASCII characters
  3. Special tokens (BOS, EOS) are correctly handled
  4. Vocabulary size matches configuration
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOKENIZER_PATH = "data/tokenizer.model"

# Skip all tests if tokenizer hasn't been trained
pytestmark = pytest.mark.skipif(
    not os.path.exists(TOKENIZER_PATH),
    reason=f"Tokenizer model not found at {TOKENIZER_PATH}. "
           f"Run 'python scripts/train_tokenizer.py' first.",
)


@pytest.fixture
def tokenizer():
    from llama_vc.tokenizer import Tokenizer
    return Tokenizer(TOKENIZER_PATH)


class TestRoundtrip:
    """Encode/decode should preserve text."""

    def test_simple_text(self, tokenizer):
        text = "Once upon a time, there was a little cat."
        tokens = tokenizer.encode(text, bos=False, eos=False)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_multiple_sentences(self, tokenizer):
        text = "Hello world. How are you? I am fine."
        tokens = tokenizer.encode(text, bos=False, eos=False)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_numbers(self, tokenizer):
        text = "The answer is 42."
        tokens = tokenizer.encode(text, bos=False, eos=False)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_empty_string(self, tokenizer):
        tokens = tokenizer.encode("", bos=False, eos=False)
        decoded = tokenizer.decode(tokens)
        assert decoded == ""


class TestByteFallback:
    """Byte-fallback should handle non-ASCII characters."""

    def test_no_unk_for_ascii(self, tokenizer):
        text = "Hello, world!"
        tokens = tokenizer.encode(text, bos=False, eos=False)
        assert tokenizer.unk_id not in tokens

    def test_accented_characters(self, tokenizer):
        """Characters like é, ñ should not produce UNK."""
        text = "café"
        tokens = tokenizer.encode(text, bos=False, eos=False)
        # With byte-fallback, these should be encoded (possibly as byte tokens)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_unicode_roundtrip(self, tokenizer):
        """Unicode text should survive encode/decode."""
        texts = ["naïve", "über", "日本"]
        for text in texts:
            tokens = tokenizer.encode(text, bos=False, eos=False)
            decoded = tokenizer.decode(tokens)
            assert decoded == text, f"Failed for '{text}': got '{decoded}'"


class TestSpecialTokens:
    """Tests for BOS/EOS handling."""

    def test_bos_prepended(self, tokenizer):
        tokens = tokenizer.encode("Hello", bos=True, eos=False)
        assert tokens[0] == tokenizer.bos_id

    def test_eos_appended(self, tokenizer):
        tokens = tokenizer.encode("Hello", bos=False, eos=True)
        assert tokens[-1] == tokenizer.eos_id

    def test_both_special(self, tokenizer):
        tokens = tokenizer.encode("Hello", bos=True, eos=True)
        assert tokens[0] == tokenizer.bos_id
        assert tokens[-1] == tokenizer.eos_id

    def test_no_special(self, tokenizer):
        tokens = tokenizer.encode("Hello", bos=False, eos=False)
        assert tokens[0] != tokenizer.bos_id or len(tokens) == 0
        if len(tokens) > 0:
            assert tokens[-1] != tokenizer.eos_id

    def test_special_token_ids(self, tokenizer):
        """Special token IDs should be standard SentencePiece values."""
        assert tokenizer.unk_id == 0
        assert tokenizer.bos_id == 1
        assert tokenizer.eos_id == 2


class TestVocabulary:
    """Tests for vocabulary properties."""

    def test_vocab_size_positive(self, tokenizer):
        assert tokenizer.vocab_size > 0

    def test_len_matches_vocab_size(self, tokenizer):
        assert len(tokenizer) == tokenizer.vocab_size

    def test_id_to_piece(self, tokenizer):
        """Should be able to look up token strings."""
        piece = tokenizer.id_to_piece(tokenizer.bos_id)
        assert isinstance(piece, str)
        assert len(piece) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
