"""
Interactive text generation CLI.

USAGE:
    # Interactive mode (type prompts, get completions)
    python scripts/generate_text.py --checkpoint checkpoints/best.pt

    # Single prompt
    python scripts/generate_text.py --checkpoint checkpoints/best.pt \
        --prompt "Once upon a time"

    # Adjust generation parameters
    python scripts/generate_text.py --checkpoint checkpoints/best.pt \
        --temperature 0.5 --top-k 20 --max-new-tokens 300

WHAT THIS SCRIPT DOES:
    1. Loads a trained model from a checkpoint
    2. Loads the tokenizer
    3. Enters an interactive loop where you type prompts and the model
       generates continuations
    4. Displays the generated text with timing information
"""

import os
import sys
import argparse
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_vc.config import ModelConfig
from llama_vc.model import LLaMA
from llama_vc.tokenizer import Tokenizer
from llama_vc.generate import generate
from llama_vc.device import get_device, device_info
from llama_vc.utils import count_parameters


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[LLaMA, dict]:
    """Load a model from a training checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model config from checkpoint
    model_config = ModelConfig.from_dict(checkpoint["model_config"])
    print(f"Model config: {model_config.dim}d, {model_config.n_layers}L, "
          f"{model_config.n_heads}H, {model_config.n_kv_heads}KV")

    # Create model and load weights
    model = LLaMA(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Parameters: {count_parameters(model):,}")
    print(f"Trained for {checkpoint.get('step', '?')} steps")
    print(f"Val loss: {checkpoint.get('val_loss', '?')}")

    return model, checkpoint


def interactive_loop(
    model: LLaMA,
    tokenizer: Tokenizer,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Run an interactive generation loop."""
    print("\n" + "=" * 60)
    print("Interactive Text Generation")
    print("=" * 60)
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("\nType a prompt and press Enter. Type 'quit' to exit.")
    print("Type 'settings' to see/change generation parameters.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            print("Goodbye!")
            break
        if prompt.lower() == "settings":
            print(f"  temperature: {args.temperature}")
            print(f"  top_k: {args.top_k}")
            print(f"  top_p: {args.top_p}")
            print(f"  max_new_tokens: {args.max_new_tokens}")
            continue

        # Generate text
        start_time = time.perf_counter()
        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        elapsed = time.perf_counter() - start_time

        # Count generated tokens (approximate)
        prompt_tokens = tokenizer.encode(prompt, bos=True, eos=False)
        total_tokens = tokenizer.encode(result, bos=False, eos=False)
        n_generated = len(total_tokens) - len(prompt_tokens) + 1

        print(f"\n{result}")
        print(f"\n--- {n_generated} tokens in {elapsed:.2f}s "
              f"({n_generated/elapsed:.1f} tok/s) ---\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with a trained LLaMA model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--tokenizer-path", type=str, default="data/tokenizer.model",
        help="Path to tokenizer model"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Single prompt to generate from (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=200,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (0=greedy, 1=default, >1=more random)"
    )
    parser.add_argument(
        "--top-k", type=int, default=40,
        help="Top-k sampling (0=disabled)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Top-p (nucleus) sampling threshold"
    )

    args = parser.parse_args()

    # Setup device
    device = get_device()
    print(device_info(device))

    # Load tokenizer
    tokenizer = Tokenizer(args.tokenizer_path)
    print(f"Tokenizer loaded: {tokenizer.vocab_size} tokens")

    # Load model
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)

    if args.prompt:
        # Single prompt mode
        start_time = time.perf_counter()
        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        elapsed = time.perf_counter() - start_time
        print(f"\n{result}")
        print(f"\n--- Generated in {elapsed:.2f}s ---")
    else:
        # Interactive mode
        interactive_loop(model, tokenizer, device, args)


if __name__ == "__main__":
    main()
