"""
SentencePiece BPE Tokenizer with Byte-Fallback.

This module wraps Google's SentencePiece library to provide a tokenizer
that matches the approach used by LLaMA (Meta AI).

TOKENIZATION OVERVIEW:
  A tokenizer converts raw text (strings) into sequences of integers (token IDs)
  that the model can process. The reverse operation (decode) converts IDs back
  to text.

  Example:
    "Once upon a time" → [1, 432, 876, 12, 345]  (encode)
    [1, 432, 876, 12, 345] → "Once upon a time"  (decode)

  The token vocabulary is a fixed set of subword units learned from training
  data. Common words become single tokens, while rare words are split into
  multiple subword pieces.

BPE (BYTE-PAIR ENCODING):
  The most popular tokenization algorithm for LLMs. It works by:
    1. Start with individual characters as the vocabulary
    2. Count all adjacent character pairs in the training data
    3. Merge the most frequent pair into a new token
    4. Repeat until vocabulary reaches desired size

  Example of BPE merges:
    "t h e" → "th e" → "the"  (if "th" and "the" are frequent)

  This creates a vocabulary of common subwords that balances:
    - Coverage: Can represent any text (falls back to characters)
    - Efficiency: Common words are single tokens (fewer tokens per sentence)

BYTE-FALLBACK:
  LLaMA's tokenizer uses "byte-fallback" mode. When the tokenizer encounters
  a character not in its vocabulary (e.g., a rare Unicode character), instead
  of producing <unk>, it encodes the character's UTF-8 bytes as individual
  byte tokens (e.g., <0xE2>, <0x80>, <0x99>).

  This means the tokenizer can handle ANY input text without loss of
  information. The model never sees <unk> tokens, which is critical for
  robustness in production systems.

SPECIAL TOKENS:
  - <unk> (id=0): Unknown token. Should never appear with byte-fallback.
  - <s> (id=1): Beginning Of Sequence (BOS). Marks the start of input.
  - </s> (id=2): End Of Sequence (EOS). Marks the end of generation.

WHY SENTENCEPIECE (not tiktoken, HuggingFace tokenizers, etc.):
  SentencePiece is what LLaMA actually uses. It's a standalone C++ library
  with Python bindings that:
    - Trains BPE/Unigram models from raw text
    - Supports byte-fallback natively
    - Produces a single .model file (easy to distribute)
    - Is fast and dependency-free
"""

import os
from typing import Optional

import sentencepiece as spm


def train_tokenizer(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 4096,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    byte_fallback: bool = True,
    num_threads: int = 4,
    max_sentence_length: int = 4192,
) -> str:
    """
    Train a SentencePiece BPE tokenizer from raw text data.

    This creates two files:
      - {model_prefix}.model: The trained model (used for encode/decode)
      - {model_prefix}.vocab: Human-readable vocabulary (for inspection)

    TRAINING PROCESS:
      1. SentencePiece reads the input file and counts character/byte frequencies
      2. It iteratively merges the most frequent byte/character pairs (BPE)
      3. It stops when the vocabulary reaches vocab_size
      4. Special tokens (<unk>, <s>, </s>) are added automatically

    PARAMETER CHOICES:
      vocab_size=4096: Small for our tiny model. This means:
        - Smaller embedding matrix (4096 × 384 = 1.57M params)
        - More tokens per sentence (less compression)
        - Faster tokenizer training
        Real LLaMA: 32000 (v1/v2), 128256 (v3)

      character_coverage=1.0: Cover 100% of characters in training data.
        For languages with large character sets (Chinese, Japanese), you
        might use 0.9995. For English TinyStories, 1.0 is fine.

      byte_fallback=True: Enable byte-level fallback for unknown characters.
        This is what makes LLaMA's tokenizer robust to any input.

    Args:
        input_file: Path to plain text file for training.
        model_prefix: Output path prefix (without extension).
        vocab_size: Target vocabulary size (including special tokens).
        model_type: "bpe" (Byte-Pair Encoding) or "unigram".
        character_coverage: Fraction of characters to cover (1.0 = all).
        byte_fallback: If True, encode unknown chars as UTF-8 bytes.
        num_threads: Parallel threads for training speed.
        max_sentence_length: Maximum sentence length in bytes (longer = truncated).

    Returns:
        Path to the trained .model file.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)

    # SentencePiece training arguments
    # See: https://github.com/google/sentencepiece/blob/master/doc/options.md
    train_args = {
        "input": input_file,
        "model_prefix": model_prefix,
        "model_type": model_type,
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "byte_fallback": byte_fallback,
        "num_threads": num_threads,
        "max_sentence_length": max_sentence_length,
        # Split digits into individual characters (like LLaMA)
        # e.g., "2023" → "2", "0", "2", "3" instead of "2023"
        # This helps the model learn arithmetic patterns.
        "split_digits": True,
        # Allow whitespace-only pieces (spaces are meaningful in text)
        "allow_whitespace_only_pieces": True,
        # Normalization: NFKC normalization for Unicode consistency
        # Maps equivalent Unicode forms to a single representation
        # e.g., "ﬁ" (U+FB01) → "fi" (two characters)
        "normalization_rule_name": "identity",
        # Remove extra whitespace during tokenization
        "remove_extra_whitespaces": False,
        # Special tokens are defined by their IDs:
        # 0 = <unk>, 1 = <s> (BOS), 2 = </s> (EOS)
        # We don't add a <pad> token since we use concatenated sequences
        # (no padding needed during training).
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": -1,  # -1 means no pad token
    }

    print(f"Training tokenizer:")
    print(f"  Input: {input_file}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model type: {model_type}")
    print(f"  Byte fallback: {byte_fallback}")
    print(f"  Output: {model_prefix}.model")

    spm.SentencePieceTrainer.Train(**train_args)

    model_path = f"{model_prefix}.model"
    print(f"Tokenizer trained successfully: {model_path}")

    # Quick verification
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    print(f"  Actual vocab size: {sp.GetPieceSize()}")
    print(f"  BOS id: {sp.bos_id()}")
    print(f"  EOS id: {sp.eos_id()}")
    print(f"  UNK id: {sp.unk_id()}")

    # Test encode/decode roundtrip
    test_text = "Once upon a time, there was a little cat."
    encoded = sp.Encode(test_text)
    decoded = sp.Decode(encoded)
    print(f"  Test encode: '{test_text}' → {encoded[:10]}... ({len(encoded)} tokens)")
    print(f"  Test decode: → '{decoded}'")
    assert decoded == test_text, f"Roundtrip failed: '{decoded}' != '{test_text}'"

    return model_path


class Tokenizer:
    """
    Wrapper around a trained SentencePiece model for encode/decode.

    This class provides a clean interface that the rest of the codebase
    uses for all tokenization. It handles:
      - Adding/removing BOS and EOS tokens
      - Encoding text → token IDs
      - Decoding token IDs → text

    USAGE:
      tokenizer = Tokenizer("data/tokenizer.model")

      # Encode text for training (with BOS and EOS)
      tokens = tokenizer.encode("Once upon a time", bos=True, eos=True)
      # → [1, 432, 876, 12, 345, 2]

      # Decode generated tokens back to text
      text = tokenizer.decode([432, 876, 12, 345])
      # → "Once upon a time"

    TOKEN ID LAYOUT:
      0        : <unk>  (should never appear with byte_fallback)
      1        : <s>    (BOS - beginning of sequence)
      2        : </s>   (EOS - end of sequence)
      3-258    : byte tokens <0x00>-<0xFF> (for byte_fallback)
      259-4095 : BPE-learned subword tokens
    """

    def __init__(self, model_path: str):
        """
        Load a trained SentencePiece model.

        Args:
            model_path: Path to the .model file created by train_tokenizer().

        Raises:
            FileNotFoundError: If the model file doesn't exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Tokenizer model not found: {model_path}\n"
                f"Train one first with: python scripts/train_tokenizer.py"
            )
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)

    def encode(
        self,
        text: str,
        bos: bool = True,
        eos: bool = True,
    ) -> list[int]:
        """
        Encode text into a list of token IDs.

        BOS/EOS CONVENTIONS:
          - Training: bos=False, eos=True (EOS separates concatenated stories)
          - Generation prompt: bos=True, eos=False (BOS signals start)
          - Full sequence: bos=True, eos=True (for standalone evaluation)

        Args:
            text: The input text string.
            bos: If True, prepend BOS token (id=1) to the sequence.
            eos: If True, append EOS token (id=2) to the sequence.

        Returns:
            List of integer token IDs.
        """
        # SentencePiece.Encode returns token IDs without special tokens
        tokens = self._sp.Encode(text)

        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]

        return tokens

    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of token IDs back into text.

        Handles special tokens gracefully:
          - BOS and EOS tokens are removed during decoding
          - Byte tokens are assembled back into UTF-8 characters
          - UNK tokens produce the UNK surface form (should not appear)

        Args:
            tokens: List of integer token IDs.

        Returns:
            The decoded text string.
        """
        return self._sp.Decode(tokens)

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary (including special tokens)."""
        return self._sp.GetPieceSize()

    @property
    def bos_id(self) -> int:
        """Token ID for Beginning Of Sequence (<s>)."""
        return self._sp.bos_id()

    @property
    def eos_id(self) -> int:
        """Token ID for End Of Sequence (</s>)."""
        return self._sp.eos_id()

    @property
    def unk_id(self) -> int:
        """Token ID for Unknown (<unk>). Should never appear with byte_fallback."""
        return self._sp.unk_id()

    @property
    def pad_id(self) -> int:
        """
        Token ID for padding. Returns -1 if no pad token is defined.

        We don't use padding in our training pipeline because we concatenate
        all stories into one long sequence and sample contiguous chunks.
        No padding = no wasted compute on meaningless tokens.
        """
        pad = self._sp.pad_id()
        return pad if pad >= 0 else -1

    def id_to_piece(self, token_id: int) -> str:
        """
        Convert a token ID to its string representation.

        Useful for debugging tokenization:
          tokenizer.id_to_piece(432) → "▁Once"
          tokenizer.id_to_piece(1) → "<s>"

        The "▁" (U+2581) character represents a space at the start of a word.
        This is SentencePiece's convention: spaces are part of the FOLLOWING
        token, not the preceding one. This allows lossless roundtrip.
        """
        return self._sp.IdToPiece(token_id)

    def piece_to_id(self, piece: str) -> int:
        """Convert a string token piece to its ID."""
        return self._sp.PieceToId(piece)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
