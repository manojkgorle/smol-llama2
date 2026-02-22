"""
CLI script to train the LLaMA model.

USAGE:
    # Train on TinyStories (default)
    python scripts/train_model.py

    # Train on a math dataset
    python scripts/train_model.py --dataset gsm8k

    # Train on all datasets combined
    python scripts/train_model.py --dataset mixed

    # Resume from a checkpoint
    python scripts/train_model.py --resume checkpoints/step_005000.pt

    # Custom settings
    python scripts/train_model.py --batch-size 32 --max-steps 5000 --lr 1e-4

AVAILABLE DATASETS:
    tinystories  ~2.1M short stories (default)
    gsm8k        8.5K grade school math word problems
    simplemath   100K basic arithmetic problems
    aqua_rat     98K word problems with reasoning
    mixed        All of the above combined

PREREQUISITES:
    1. Train the tokenizer first: python scripts/train_tokenizer.py
       (use --datasets all if training on math data)
    2. Ensure you have the dependencies: pip install -r requirements.txt

WHAT THIS SCRIPT DOES:
    1. Creates ModelConfig and TrainConfig from CLI arguments
    2. Downloads and tokenizes the selected dataset (if needed)
    3. Calls the train() function from llama_vc.train
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_vc.config import ModelConfig, TrainConfig
from llama_vc.train import train
from llama_vc.tokenizer import Tokenizer
from llama_vc.dataset import DATASET_NAMES, prepare_dataset, prepare_mixed


def main():
    parser = argparse.ArgumentParser(
        description="Train a LLaMA model on selected dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset selection
    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument(
        "--dataset", type=str, default="tinystories",
        help="Dataset to train on. "
             f"Choices: {', '.join(DATASET_NAMES)}, mixed"
    )

    # Model architecture arguments
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--dim", type=int, default=384, help="Model dimension")
    model_group.add_argument("--n-layers", type=int, default=8, help="Number of layers")
    model_group.add_argument("--n-heads", type=int, default=6, help="Number of query heads")
    model_group.add_argument("--n-kv-heads", type=int, default=2, help="Number of KV heads (GQA)")
    model_group.add_argument("--hidden-dim", type=int, default=1024, help="FFN hidden dimension")
    model_group.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
    model_group.add_argument("--vocab-size", type=int, default=4096, help="Vocabulary size")
    model_group.add_argument("--weight-tying", action="store_true", help="Tie embedding and output weights")
    model_group.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--batch-size", type=int, default=64, help="Micro-batch size")
    train_group.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    train_group.add_argument("--max-steps", type=int, default=10000, help="Total training steps")
    train_group.add_argument("--warmup-steps", type=int, default=500, help="LR warmup steps")
    train_group.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    train_group.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate")
    train_group.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    train_group.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    train_group.add_argument("--dtype", type=str, default="auto",
                            choices=["auto", "float16", "bfloat16", "float32"],
                            help="Training precision")
    train_group.add_argument("--compile", action="store_true", help="Use torch.compile (CUDA only)")
    train_group.add_argument("--seed", type=int, default=42, help="Random seed")

    # Evaluation and checkpointing
    eval_group = parser.add_argument_group("Evaluation & Checkpointing")
    eval_group.add_argument("--eval-interval", type=int, default=250, help="Eval every N steps")
    eval_group.add_argument("--eval-steps", type=int, default=20, help="Batches per evaluation")
    eval_group.add_argument("--save-interval", type=int, default=1000, help="Save every N steps")
    eval_group.add_argument("--log-interval", type=int, default=10, help="Log every N steps")

    # Paths
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument("--data-dir", type=str, default="data/", help="Data directory")
    path_group.add_argument("--checkpoint-dir", type=str, default="checkpoints/", help="Checkpoint directory")
    path_group.add_argument("--log-dir", type=str, default="logs/", help="Log directory")
    path_group.add_argument("--tokenizer-path", type=str, default="data/tokenizer.model", help="Tokenizer model path")
    path_group.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Validate dataset choice
    valid_datasets = DATASET_NAMES + ["mixed"]
    if args.dataset not in valid_datasets:
        parser.error(
            f"Unknown dataset '{args.dataset}'. "
            f"Available: {', '.join(valid_datasets)}"
        )

    # Build configs from arguments
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.max_seq_len,
        weight_tying=args.weight_tying,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    train_config = TrainConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        dtype=args.dtype,
        compile_model=args.compile,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        seed=args.seed,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        tokenizer_path=args.tokenizer_path,
    )

    # Validate model config
    model_config.validate()

    # Print configuration summary
    print("=" * 60)
    print("LLaMA Training Configuration")
    print("=" * 60)
    print(f"\nDataset: {args.dataset}")
    print(f"Model: {args.dim}d, {args.n_layers}L, {args.n_heads}H, {args.n_kv_heads}KV")
    print(f"FFN hidden: {args.hidden_dim}, Vocab: {args.vocab_size}, Seq: {args.max_seq_len}")
    print(f"\nTraining: {args.max_steps} steps, batch {args.batch_size}×{args.grad_accum}")
    print(f"LR: {args.lr} → {args.min_lr} (warmup: {args.warmup_steps})")
    print(f"Precision: {args.dtype}")
    print(f"\nData: {args.data_dir}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    print("=" * 60 + "\n")

    # Prepare the selected dataset
    tokenizer = Tokenizer(args.tokenizer_path)
    if args.dataset == "mixed":
        train_bin, val_bin = prepare_mixed(args.data_dir, tokenizer)
    else:
        train_bin, val_bin = prepare_dataset(args.dataset, args.data_dir, tokenizer)

    # Run training with resolved data paths
    train(
        model_config, train_config,
        resume_from=args.resume,
        train_bin=train_bin,
        val_bin=val_bin,
    )


if __name__ == "__main__":
    main()
