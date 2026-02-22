"""
Data pipeline for training: download, tokenize, and serve TinyStories data.

This module handles the complete data lifecycle:
  1. Download TinyStories from HuggingFace
  2. Tokenize all stories using our trained BPE tokenizer
  3. Save tokenized data as memory-mapped binary files
  4. Serve random sequence chunks to the training loop via DataLoader

DATA FORMAT — MEMORY-MAPPED BINARY FILES:
  After tokenization, we save all token IDs as a flat numpy array in a
  binary file (.bin) using numpy's memmap format.

  WHY MEMORY-MAPPED FILES?
    Memory mapping lets the OS handle data loading:
      - The file is NOT loaded into RAM all at once
      - The OS pages in data on-demand as we access it
      - Multiple processes can share the same file
      - Works even when the dataset is larger than available RAM

    For TinyStories:
      - ~300M tokens × 2 bytes (uint16) = ~600MB binary file
      - With memmap, we use only a few MB of RAM regardless
      - Random access is fast because the OS caches recently used pages

  WHY uint16?
    Our vocab_size=4096 fits easily in uint16 (max value 65535).
    This halves the file size compared to int32.
    If vocab_size > 65535, you'd need uint32.

CONCATENATION STRATEGY:
  Instead of storing individual stories, we concatenate ALL stories into
  one continuous sequence, separated by EOS tokens:

    [story1_tok1, story1_tok2, ..., story1_tokN, EOS, story2_tok1, ..., EOS, ...]

  WHY CONCATENATION?
    - No padding needed: padding wastes compute on meaningless tokens
    - No variable-length batching complexity
    - Each training sample is a fixed-length window into this sequence
    - The model naturally learns to handle story boundaries (EOS tokens)
    - This is the standard approach used by GPT-2/3, LLaMA, etc.

RANDOM WINDOW SAMPLING:
  Each training sample is a random contiguous window of (seq_len + 1) tokens.
  The +1 is because we need (input, target) pairs where target is shifted by 1:
    input  = tokens[i : i+seq_len]      (what the model sees)
    target = tokens[i+1 : i+seq_len+1]  (what the model should predict)

  Random starting positions ensure the model sees diverse contexts and
  doesn't overfit to specific story alignments.
"""

import multiprocessing as mp
import os
from typing import Optional

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from llama_vc.tokenizer import Tokenizer


# --- Parallel tokenization helpers ---
# These must be module-level for multiprocessing pickling.

def _init_tokenize_worker(model_path: str):
    """Initialize a SentencePiece model in each worker process."""
    global _worker_sp
    _worker_sp = spm.SentencePieceProcessor()
    _worker_sp.Load(model_path)


def _tokenize_file_range(args: tuple[str, int, int]) -> np.ndarray:
    """
    Read and tokenize a byte range of a text file. Runs in a worker process.

    Each worker opens the file directly and seeks to its assigned byte range,
    avoiding the need to send text data through IPC pipes.
    """
    global _worker_sp
    text_file, start_byte, end_byte = args
    eos_id = _worker_sp.eos_id()

    tokens: list[int] = []

    with open(text_file, "rb") as f:
        if start_byte > 0:
            f.seek(start_byte)
            f.readline()  # align to next complete line

        batch: list[str] = []
        while f.tell() <= end_byte:
            line = f.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                batch.append(text)

            # Batch encode every 10K lines (one C++ call instead of 10K)
            if len(batch) >= 10000:
                for ids in _worker_sp.Encode(batch):
                    tokens.extend(ids)
                    tokens.append(eos_id)
                batch = []

        # Encode remaining
        if batch:
            for ids in _worker_sp.Encode(batch):
                tokens.extend(ids)
                tokens.append(eos_id)

    return np.array(tokens, dtype=np.uint16)


def download_tinystories(data_dir: str) -> str:
    """
    Download TinyStories dataset and return path to train split text.

    Uses HuggingFace datasets library to download. The first time this runs
    it will download ~1GB of data, which is cached for future runs.

    Args:
        data_dir: Directory to store downloaded data.

    Returns:
        Path to the exported training text file.
    """
    from datasets import load_dataset

    os.makedirs(data_dir, exist_ok=True)
    text_file = os.path.join(data_dir, "tinystories_train.txt")

    if os.path.exists(text_file):
        print(f"Training text already exists: {text_file}")
        return text_file

    print("Downloading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    print(f"Downloaded {len(dataset):,} stories")

    print("Exporting to text file...")
    with open(text_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(tqdm(dataset, desc="Exporting")):
            text = example["text"].strip()
            if text:
                f.write(text + "\n")

    file_size_mb = os.path.getsize(text_file) / 1024**2
    print(f"Saved: {text_file} ({file_size_mb:.1f} MB)")
    return text_file


def download_tinystories_val(data_dir: str) -> str:
    """Download TinyStories validation split."""
    from datasets import load_dataset

    os.makedirs(data_dir, exist_ok=True)
    text_file = os.path.join(data_dir, "tinystories_val.txt")

    if os.path.exists(text_file):
        print(f"Validation text already exists: {text_file}")
        return text_file

    print("Downloading TinyStories validation split...")
    dataset = load_dataset("roneneldan/TinyStories", split="validation")
    print(f"Downloaded {len(dataset):,} validation stories")

    with open(text_file, "w", encoding="utf-8") as f:
        for example in tqdm(dataset, desc="Exporting val"):
            text = example["text"].strip()
            if text:
                f.write(text + "\n")

    return text_file


def tokenize_and_save(
    text_file: str,
    output_bin: str,
    tokenizer: Tokenizer,
) -> int:
    """
    Tokenize a text file and save as a memory-mapped binary file.

    Uses multiprocessing for speed: the file is split into byte ranges
    and each worker reads/tokenizes its range directly from disk.
    No text data is sent through IPC pipes.

    Args:
        text_file: Path to the plain text file (one story per line).
        output_bin: Path for the output binary file.
        tokenizer: Trained tokenizer for encoding text.

    Returns:
        Total number of tokens written.
    """
    if os.path.exists(output_bin):
        # Load existing and return its size
        data = np.memmap(output_bin, dtype=np.uint16, mode="r")
        print(f"Tokenized data already exists: {output_bin} ({len(data):,} tokens)")
        return len(data)

    print(f"Tokenizing {text_file} → {output_bin}")

    # Verify no overflow (uint16 max = 65535)
    assert tokenizer.vocab_size <= 65535, (
        f"vocab_size {tokenizer.vocab_size} exceeds uint16 max (65535)"
    )

    # Split the file into byte ranges for parallel processing.
    # Workers read directly from the file (no data sent through IPC pipes).
    file_size = os.path.getsize(text_file)
    num_workers = os.cpu_count() or 4
    n_chunks = num_workers * 4  # more chunks than workers for progress granularity

    chunk_bytes = file_size // n_chunks
    ranges = []
    for i in range(n_chunks):
        start = i * chunk_bytes
        end = file_size if i == n_chunks - 1 else (i + 1) * chunk_bytes
        ranges.append((text_file, start, end))

    print(f"  File size: {file_size / 1024**2:.1f} MB")
    print(f"  Using {num_workers} workers, {n_chunks} chunks")

    os.makedirs(os.path.dirname(output_bin) or ".", exist_ok=True)
    total_tokens = 0

    with mp.Pool(num_workers, initializer=_init_tokenize_worker, initargs=(tokenizer.model_path,)) as pool:
        with open(output_bin, "wb") as f_out:
            for token_array in tqdm(
                pool.imap(_tokenize_file_range, ranges),
                total=n_chunks,
                desc="Tokenizing",
            ):
                token_array.tofile(f_out)
                total_tokens += len(token_array)

    file_size_mb = os.path.getsize(output_bin) / 1024**2
    print(f"  Saved: {output_bin} ({total_tokens:,} tokens, {file_size_mb:.1f} MB)")

    return total_tokens


def prepare_data(
    data_dir: str,
    tokenizer: Tokenizer,
) -> tuple[str, str]:
    """
    Complete data preparation pipeline.

    Downloads TinyStories (if needed), tokenizes train and validation splits,
    and saves as binary files.

    Args:
        data_dir: Base directory for all data files.
        tokenizer: Trained tokenizer.

    Returns:
        Tuple of (train_bin_path, val_bin_path).
    """
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")

    # Download and tokenize training data
    if not os.path.exists(train_bin):
        train_text = download_tinystories(data_dir)
        tokenize_and_save(train_text, train_bin, tokenizer)
    else:
        n_tokens = os.path.getsize(train_bin) // 2  # uint16 = 2 bytes
        print(f"Train data ready: {train_bin} ({n_tokens:,} tokens)")

    # Download and tokenize validation data
    if not os.path.exists(val_bin):
        val_text = download_tinystories_val(data_dir)
        tokenize_and_save(val_text, val_bin, tokenizer)
    else:
        n_tokens = os.path.getsize(val_bin) // 2
        print(f"Val data ready: {val_bin} ({n_tokens:,} tokens)")

    return train_bin, val_bin


class TokenDataset(Dataset):
    """
    PyTorch Dataset that reads from a pre-tokenized binary file.

    Each call to __getitem__ returns a random contiguous chunk of tokens,
    split into (input, target) pairs for next-token prediction.

    EXAMPLE:
      If the binary file contains: [45, 12, 88, 33, 7, 91, 56, ...]
      And seq_len = 4, a sample starting at position 2 would be:
        input  = [88, 33,  7, 91]   (positions 2-5)
        target = [33,  7, 91, 56]   (positions 3-6)

      The target is the input shifted right by 1 position.
      This teaches the model: given tokens up to position i, predict position i+1.

    MEMORY EFFICIENCY:
      We use np.memmap to access the binary file. This means:
        - Only the accessed pages are loaded into RAM
        - The OS manages the cache (recently used data stays in memory)
        - Multiple DataLoader workers can share the same memory-mapped file
        - Even a 600MB file uses only a few MB of RAM

    WHY RANDOM POSITIONS (not sequential):
      Sequential reading would mean the model sees the same stories in the
      same order every epoch. Random starting positions create diverse
      training contexts:
        - The model might start in the middle of a story (learns to continue)
        - Story boundaries (EOS tokens) appear at random positions in the window
        - Each epoch effectively creates new training examples

    Args:
        data_path: Path to the binary token file (.bin).
        seq_len: Sequence length for training (512).
    """

    def __init__(self, data_path: str, seq_len: int):
        """
        Args:
            data_path: Path to the tokenized binary file.
            seq_len: Training sequence length. Each sample returns
                    (seq_len,) input and (seq_len,) target tensors.
        """
        super().__init__()
        self.seq_len = seq_len

        # Memory-map the binary file as uint16
        # mode='r' = read-only (prevents accidental modification)
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)

        print(f"TokenDataset: {data_path}")
        print(f"  Tokens: {self.n_tokens:,}")
        print(f"  Seq length: {seq_len}")
        print(f"  Possible positions: {self.n_tokens - seq_len - 1:,}")

    def __len__(self) -> int:
        """
        Number of possible starting positions.

        We need seq_len + 1 consecutive tokens (input + 1 target token),
        so the last valid starting position is n_tokens - seq_len - 1.
        """
        return self.n_tokens - self.seq_len - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.

        Args:
            idx: Starting position in the token sequence.

        Returns:
            Tuple of (input_tokens, target_tokens), each of shape (seq_len,).
            Both are long tensors (int64) as expected by nn.Embedding.
        """
        # Read seq_len + 1 consecutive tokens starting at idx
        # The +1 gives us the target for the last input position
        chunk = self.data[idx: idx + self.seq_len + 1].astype(np.int64)

        # Split into input and target
        # input:  tokens[0 : seq_len]      → what the model sees
        # target: tokens[1 : seq_len + 1]  → what the model should predict
        x = torch.from_numpy(chunk[:-1])  # (seq_len,)
        y = torch.from_numpy(chunk[1:])   # (seq_len,)

        return x, y


def create_dataloader(
    data_path: str,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for training or evaluation.

    DataLoader wraps a Dataset and handles:
      - Batching: Combine multiple samples into a batch
      - Shuffling: Randomize the order of samples each epoch
      - Prefetching: Load next batch in background while GPU processes current
      - Multi-process loading: Use multiple CPU workers for data preparation

    PERFORMANCE TIPS:
      pin_memory=True: Allocates batch tensors in pinned (page-locked) CPU
        memory. This allows faster CPU→GPU transfer (DMA instead of copy).
        Only useful when training on GPU.

      num_workers: Number of parallel data loading processes.
        - 0: Load in main process (simplest, good for debugging)
        - 2-4: Good balance for our use case
        - Too many: Diminishing returns, wastes CPU/memory
        With memory-mapped files, even 2 workers is usually enough because
        the OS handles the actual I/O.

    Args:
        data_path: Path to tokenized binary file.
        seq_len: Sequence length for training.
        batch_size: Number of sequences per batch.
        shuffle: Whether to shuffle sample order (True for train, False for eval).
        num_workers: Number of data loading worker processes.
        pin_memory: Whether to use pinned memory for faster GPU transfer.

    Returns:
        Configured DataLoader ready for iteration.
    """
    dataset = TokenDataset(data_path, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete last batch (simplifies training loop)
    )
