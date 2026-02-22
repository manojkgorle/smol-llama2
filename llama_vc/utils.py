"""
Utility functions for the LLaMA training and inference pipeline.

This module contains cross-cutting concerns that don't belong in any
specific component: reproducibility (seeding), diagnostics (parameter
counting, model summaries), timing, and logging.

These utilities are intentionally simple — no frameworks, no dependencies
beyond PyTorch and the standard library.
"""

import os
import time
import random
from typing import Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all random number generators.

    WHY SEED EVERYTHING:
      Deep learning training involves randomness at multiple levels:
        1. Python's random: Used by data shuffling in some libraries
        2. NumPy: Used by data preprocessing and augmentation
        3. PyTorch CPU: Weight initialization, dropout masks
        4. PyTorch CUDA: Same as CPU but on GPU
        5. cuDNN: CUDA's deep learning library chooses between multiple
           algorithms for conv/matmul — some are non-deterministic

      Setting all seeds ensures that, given the same data and code,
      training produces the same results. This is critical for:
        - Debugging (reproduce a failure)
        - Research (compare approaches fairly)
        - Testing (unit tests pass consistently)

    NOTE ON FULL DETERMINISM:
      Even with all seeds set, GPU operations may still be non-deterministic
      unless torch.use_deterministic_algorithms(True) is called. We don't
      enable this by default because:
        1. It can be 10-50% slower
        2. Some operations don't have deterministic implementations
        3. For training (not testing), slight non-determinism is acceptable

    Args:
        seed: The random seed value. Use the same seed for reproducible runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed also seeds CUDA, but we call cuda seed explicitly
    # for clarity and to handle multi-GPU cases.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the total number of parameters in a model.

    WHY THIS MATTERS:
      Parameter count is the primary measure of model "size" and directly
      impacts:
        - Memory: Each fp32 param = 4 bytes, fp16/bf16 = 2 bytes
        - Compute: More params = more FLOPs per forward/backward pass
        - Capacity: More params = more patterns the model can learn
        - Training data needed: Roughly, you need ~20× params in tokens
          for good training (Chinchilla scaling laws)

    For our model, we expect exactly 15,735,168 trainable parameters.

    Args:
        model: The PyTorch model.
        trainable_only: If True, count only parameters with requires_grad=True.
                       Frozen/non-trainable parameters are excluded.

    Returns:
        Total number of (trainable) parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module) -> str:
    """
    Print a detailed breakdown of model parameters by component.

    This helps verify that the parameter count matches expectations and
    identify which components dominate the parameter budget.

    Example output:
      ┌──────────────────────────────────────────────────────────┐
      │ Model Parameter Summary                                  │
      ├──────────────────────────────────────────────────────────┤
      │ tok_embeddings             1,572,864  (10.0%)            │
      │ layers.0.attention.wq       147,456  ( 0.9%)            │
      │ layers.0.attention.wk        49,152  ( 0.3%)            │
      │ ...                                                      │
      │ output                     1,572,864  (10.0%)            │
      ├──────────────────────────────────────────────────────────┤
      │ TOTAL (trainable):        15,735,168                     │
      │ TOTAL (all):              15,735,168                     │
      │ Memory (fp32):                60.0 MB                    │
      │ Memory (fp16/bf16):           30.1 MB                    │
      └──────────────────────────────────────────────────────────┘

    Returns:
        The summary as a string (also printed to stdout).
    """
    lines = []
    total_params = 0
    trainable_params = 0

    lines.append("=" * 65)
    lines.append("Model Parameter Summary")
    lines.append("=" * 65)
    lines.append(f"{'Name':<40} {'Params':>12} {'%':>7}")
    lines.append("-" * 65)

    # Collect all named parameters and their sizes
    param_list = [(name, p) for name, p in model.named_parameters()]
    grand_total = sum(p.numel() for _, p in param_list)

    for name, param in param_list:
        n = param.numel()
        total_params += n
        if param.requires_grad:
            trainable_params += n
        pct = 100.0 * n / grand_total if grand_total > 0 else 0
        lines.append(f"  {name:<38} {n:>12,d} ({pct:>5.1f}%)")

    lines.append("-" * 65)
    lines.append(f"  {'TOTAL (trainable)':<38} {trainable_params:>12,d}")
    lines.append(f"  {'TOTAL (all)':<38} {total_params:>12,d}")

    # Memory estimates
    # Each fp32 parameter = 4 bytes, fp16/bf16 = 2 bytes
    # During training, optimizer states add ~2× (Adam: momentum + variance)
    fp32_mb = total_params * 4 / 1024**2
    fp16_mb = total_params * 2 / 1024**2
    lines.append(f"  {'Memory (fp32 weights only)':<38} {fp32_mb:>10.1f} MB")
    lines.append(f"  {'Memory (fp16/bf16 weights only)':<38} {fp16_mb:>10.1f} MB")
    lines.append(
        f"  {'Memory (training est. fp32+optim)':<38} "
        f"{fp32_mb * 3:>10.1f} MB"
    )
    lines.append("=" * 65)

    summary = "\n".join(lines)
    print(summary)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# TIMING
# ═══════════════════════════════════════════════════════════════════════════

class Timer:
    """
    Simple context manager for timing code blocks.

    Usage:
        with Timer("Forward pass"):
            output = model(input)
        # Prints: "Forward pass: 0.0234s"

    WHY NOT just time.time()?
      This is slightly cleaner and handles CUDA synchronization.
      On CUDA, operations are asynchronous — time.time() might return
      before the GPU finishes. We call torch.cuda.synchronize() to
      ensure accurate timing.
    """

    def __init__(self, name: str = "Block", device: Optional[torch.device] = None):
        """
        Args:
            name: Label for the timing output.
            device: If CUDA device, synchronize before timing for accuracy.
        """
        self.name = name
        self.device = device
        self.elapsed: float = 0.0

    def __enter__(self):
        # Synchronize CUDA to ensure all prior GPU work is complete
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        # Synchronize again to ensure the timed work is complete
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self.start

    def __str__(self):
        return f"{self.name}: {self.elapsed:.4f}s"


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOGGER
# ═══════════════════════════════════════════════════════════════════════════

class TrainingLogger:
    """
    Lightweight training logger that writes to console and optional log file.

    Tracks key metrics over training:
      - Loss (cross-entropy): Should decrease over time. Final loss for our
        model on TinyStories should be ~1.0-1.5 (perplexity ~3-4).
      - Learning rate: Follows the warmup + cosine schedule.
      - Tokens/second: Throughput metric. Higher = faster training.
      - Memory: GPU memory usage (helps detect memory leaks).
      - Validation loss/perplexity: Generalization metric.

    WHY NOT use wandb/tensorboard?
      Those are great for real projects, but add dependencies and setup.
      For learning purposes, a simple logger that prints to console is
      easier to understand and debug. You can add wandb later.
    """

    def __init__(self, log_dir: Optional[str] = None):
        """
        Args:
            log_dir: Directory for log files. If None, only console output.
        """
        self.log_file = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(log_dir, f"train_{timestamp}.log")
            self.log_file = open(log_path, "w")
            print(f"Logging to: {log_path}")

    def _write(self, msg: str) -> None:
        """Write message to console and optionally to log file."""
        print(msg)
        if self.log_file:
            self.log_file.write(msg + "\n")
            self.log_file.flush()  # Ensure immediate write (don't lose data on crash)

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        tokens_per_sec: float,
        memory_mb: float,
        total_steps: int,
    ) -> None:
        """
        Log metrics for a single training step.

        Example output:
          step 100/10000 | loss 5.234 | lr 6.00e-05 | 25,432 tok/s | mem 245 MB
        """
        self._write(
            f"step {step:>6d}/{total_steps} | "
            f"loss {loss:.4f} | "
            f"lr {lr:.2e} | "
            f"{tokens_per_sec:>8,.0f} tok/s | "
            f"mem {memory_mb:>6.0f} MB"
        )

    def log_eval(
        self,
        step: int,
        val_loss: float,
    ) -> None:
        """
        Log validation metrics.

        Perplexity = exp(loss). It represents the model's "confusion":
          - Perplexity 1.0 = perfect prediction (impossible in practice)
          - Perplexity N = model is as confused as choosing randomly among
            N equally likely options
          - For TinyStories, we aim for perplexity ~3-5 (good for 15M model)
        """
        perplexity = torch.exp(torch.tensor(val_loss)).item()
        self._write(
            f"{'─' * 60}\n"
            f"EVAL step {step} | val_loss {val_loss:.4f} | "
            f"perplexity {perplexity:.2f}\n"
            f"{'─' * 60}"
        )

    def log_info(self, msg: str) -> None:
        """Log an informational message."""
        self._write(f"[INFO] {msg}")

    def close(self) -> None:
        """Close the log file if open."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    model_config: dict,
    train_config: dict,
    path: str,
) -> None:
    """
    Save a complete training checkpoint.

    WHAT'S IN A CHECKPOINT:
      A checkpoint saves everything needed to:
        1. Resume training (model + optimizer states + step number)
        2. Run inference (model state + model config)
        3. Reproduce the experiment (both configs)

    The model_config is crucial: without it, you can't reconstruct the
    model architecture to load the weights into. This is a common pitfall
    — always save the config alongside the weights!

    WHY save optimizer state?
      Adam/AdamW maintains per-parameter "momentum" (1st moment) and
      "velocity" (2nd moment) running averages. Without these, resuming
      training effectively restarts the optimizer, causing a spike in
      loss and wasting computation.

    WHY save RNG states?
      For exact reproducibility when resuming. Without this, the data
      ordering and dropout masks (if any) will differ after resume.

    Args:
        model: The model to save.
        optimizer: The optimizer (for resume).
        step: Current training step.
        val_loss: Best validation loss so far.
        model_config: ModelConfig as dict (for architecture reconstruction).
        train_config: TrainConfig as dict (for documentation).
        path: File path for the checkpoint (.pt file).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "model_config": model_config,
        "train_config": train_config,
        # RNG states for exact reproducibility
        "rng_state_python": random.getstate(),
        "rng_state_numpy": np.random.get_state(),
        "rng_state_torch": torch.random.get_rng_state(),
    }

    # Also save CUDA RNG state if available
    if torch.cuda.is_available():
        checkpoint["rng_state_cuda"] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path} (step {step}, val_loss {val_loss:.4f})")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: The model to load weights into (must match architecture).
        optimizer: If provided, load optimizer state for training resume.
        device: Device to map tensors to (handles CPU→GPU, GPU→CPU).

    Returns:
        Dict with 'step', 'val_loss', 'model_config', 'train_config'.
    """
    map_location = device if device else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore RNG states for exact reproducibility
    if "rng_state_python" in checkpoint:
        random.setstate(checkpoint["rng_state_python"])
    if "rng_state_numpy" in checkpoint:
        np.random.set_state(checkpoint["rng_state_numpy"])
    if "rng_state_torch" in checkpoint:
        torch.random.set_rng_state(checkpoint["rng_state_torch"])
    if "rng_state_cuda" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["rng_state_cuda"])

    info = {
        "step": checkpoint.get("step", 0),
        "val_loss": checkpoint.get("val_loss", float("inf")),
        "model_config": checkpoint.get("model_config", {}),
        "train_config": checkpoint.get("train_config", {}),
    }

    print(f"Checkpoint loaded: {path} (step {info['step']})")
    return info
