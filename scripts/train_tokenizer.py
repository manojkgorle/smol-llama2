"""
Script to download datasets and train a BPE tokenizer.

USAGE:
    # Train on TinyStories only (default, backward compatible)
    python scripts/train_tokenizer.py

    # Train on all datasets (TinyStories + math)
    python scripts/train_tokenizer.py --datasets all

    # Train on specific datasets
    python scripts/train_tokenizer.py --datasets tinystories gsm8k simplemath

This script:
  1. Downloads the selected datasets from HuggingFace
  2. Exports training text to flat files (one per dataset)
  3. Trains a SentencePiece BPE tokenizer on ALL selected text sources
  4. Verifies the tokenizer works correctly on examples from each dataset

The tokenizer model is saved to data/tokenizer.model and is used by
the training and inference pipelines.

AVAILABLE DATASETS:
  tinystories  ~2.1M short stories (Eldan & Li, 2023)
  gsm8k        8.5K grade school math word problems (OpenAI)
  simplemath   100K basic arithmetic problems
  aqua_rat     98K word problems with reasoning (DeepMind)
  all          All of the above
"""

import os
import sys
import argparse

# Add project root to path so we can import llama_vc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_vc.tokenizer import train_tokenizer, Tokenizer
from llama_vc.dataset import DATASETS, DATASET_NAMES, download_dataset


def verify_tokenizer(model_path: str) -> None:
    """
    Run verification tests on the trained tokenizer.

    Tests:
      1. Basic encode/decode roundtrip (stories + math)
      2. Byte-fallback for non-ASCII characters
      3. Special token handling
      4. Vocabulary size
    """
    print("\n" + "=" * 50)
    print("Verifying tokenizer...")
    print("=" * 50)

    tok = Tokenizer(model_path)

    # Test 1: Basic roundtrip — stories
    test_texts = [
        "Once upon a time, there was a little cat.",
        "The sun was shining bright in the blue sky.",
        "She smiled and said hello to her friend.",
    ]
    # Test 1b: Basic roundtrip — math
    math_texts = [
        "1 + 2 = 3",
        "Question: What is 48 / 2?\nAnswer: 24",
        "5159 + 4115 = 9274",
        "The answer is 42.",
    ]
    for text in test_texts + math_texts:
        tokens = tok.encode(text, bos=False, eos=False)
        decoded = tok.decode(tokens)
        status = "PASS" if decoded == text else "FAIL"
        print(f"  [{status}] Roundtrip: '{text[:60]}...' ({len(tokens)} tokens)"
              if len(text) > 60 else
              f"  [{status}] Roundtrip: '{text}' ({len(tokens)} tokens)")
        if decoded != text:
            print(f"         Got: '{decoded}'")

    # Test 2: Byte-fallback (non-ASCII)
    unicode_texts = [
        "café",       # French accent
        "naïve",      # diaeresis
        "日本語",     # Japanese
        "🎉🎊",      # Emoji
    ]
    for text in unicode_texts:
        tokens = tok.encode(text, bos=False, eos=False)
        decoded = tok.decode(tokens)
        has_unk = tok.unk_id in tokens
        status = "PASS" if not has_unk else "WARN"
        print(f"  [{status}] Byte-fallback: '{text}' → {len(tokens)} tokens, unk={has_unk}")

    # Test 3: Special tokens
    tokens_with_special = tok.encode("Hello", bos=True, eos=True)
    assert tokens_with_special[0] == tok.bos_id, "BOS should be first token"
    assert tokens_with_special[-1] == tok.eos_id, "EOS should be last token"
    print(f"  [PASS] Special tokens: BOS={tok.bos_id}, EOS={tok.eos_id}")

    # Test 4: Vocabulary info
    print(f"\n  Vocabulary size: {tok.vocab_size}")
    print(f"  Sample tokens:")
    for i in [0, 1, 2, 3, 100, 500, 1000, 2000]:
        if i < tok.vocab_size:
            piece = tok.id_to_piece(i)
            print(f"    {i:>5d}: '{piece}'")

    print("\nTokenizer verification complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on selected datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vocab-size", type=int, default=4096,
        help="Vocabulary size. Consider 8192 when training on math datasets "
             "for better number representation"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/",
        help="Directory for data files"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["tinystories"],
        help="Datasets to include in tokenizer training. "
             f"Choices: {', '.join(DATASET_NAMES)}, all"
    )
    parser.add_argument(
        "--max-examples", type=int, default=-1,
        help="Max examples per dataset (-1 = all). "
             "Use e.g. 100000 for faster dev iteration"
    )
    args = parser.parse_args()

    # Expand "all" to all dataset names
    if "all" in args.datasets:
        args.datasets = list(DATASET_NAMES)

    # Validate dataset names
    for name in args.datasets:
        if name not in DATASETS:
            parser.error(
                f"Unknown dataset '{name}'. "
                f"Available: {', '.join(DATASET_NAMES)}, all"
            )

    print("=" * 60)
    print("Tokenizer Training")
    print("=" * 60)
    print(f"Datasets:   {', '.join(args.datasets)}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Data dir:   {args.data_dir}")
    print("=" * 60)

    # Step 1: Download and export text files for each dataset
    text_files = []
    for name in args.datasets:
        print(f"\n--- Downloading {name} ---")
        train_txt, _ = download_dataset(name, args.data_dir, max_examples=args.max_examples)
        text_files.append(train_txt)

    # Step 2: Train tokenizer on all text sources
    # SentencePiece accepts comma-separated input files natively
    model_prefix = os.path.join(args.data_dir, "tokenizer")
    model_path = f"{model_prefix}.model"

    combined_input = ",".join(text_files)
    print(f"\nTraining tokenizer on {len(text_files)} source(s)...")
    for f in text_files:
        size_mb = os.path.getsize(f) / 1024**2
        print(f"  {f} ({size_mb:.1f} MB)")

    train_tokenizer(
        input_file=combined_input,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
    )

    # Step 3: Verify
    verify_tokenizer(model_path)


if __name__ == "__main__":
    main()
