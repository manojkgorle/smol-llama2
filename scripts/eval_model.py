"""
Evaluate a trained model on the validation set.

USAGE:
    python scripts/eval_model.py --checkpoint checkpoints/best.pt

Reports:
  - Validation loss (cross-entropy)
  - Perplexity (exp(loss))
  - Tokens processed and throughput
"""

import os
import sys
import argparse
import math
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_vc.config import ModelConfig, TrainConfig
from llama_vc.model import LLaMA
from llama_vc.tokenizer import Tokenizer
from llama_vc.dataset import create_dataloader
from llama_vc.device import get_device, get_dtype, get_autocast_context, device_info
from llama_vc.utils import count_parameters


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LLaMA model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--tokenizer-path", type=str, default="data/tokenizer.model")
    parser.add_argument("--val-data", type=str, default="data/val.bin", help="Validation data path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=-1, help="Max batches (-1=all)")
    parser.add_argument("--dtype", type=str, default="auto")
    args = parser.parse_args()

    device = get_device()
    print(device_info(device))

    dtype = get_dtype(args.dtype, device)
    autocast_ctx = get_autocast_context(device, dtype)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = ModelConfig.from_dict(checkpoint["model_config"])
    model = LLaMA(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n_params = count_parameters(model)
    print(f"Model: {n_params:,} parameters")
    print(f"Checkpoint step: {checkpoint.get('step', '?')}")

    # Create validation DataLoader
    val_loader = create_dataloader(
        args.val_data,
        seq_len=model_config.max_seq_len,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Evaluate
    print(f"\nEvaluating on {args.val_data}...")
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for x, y in val_loader:
            if args.max_batches > 0 and n_batches >= args.max_batches:
                break

            x = x.to(device)
            y = y.to(device)

            with autocast_ctx:
                _, loss = model(x, targets=y)

            total_loss += loss.item() * x.shape[0]  # Weight by batch size
            total_tokens += x.shape[0] * x.shape[1]
            n_batches += 1

            if n_batches % 10 == 0:
                print(f"  Batch {n_batches}...")

    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / (n_batches * args.batch_size)
    perplexity = math.exp(avg_loss)

    print(f"\n{'=' * 50}")
    print(f"Evaluation Results")
    print(f"{'=' * 50}")
    print(f"  Batches:     {n_batches}")
    print(f"  Tokens:      {total_tokens:,}")
    print(f"  Loss:        {avg_loss:.4f}")
    print(f"  Perplexity:  {perplexity:.2f}")
    print(f"  Time:        {elapsed:.2f}s")
    print(f"  Throughput:  {total_tokens/elapsed:,.0f} tok/s")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
