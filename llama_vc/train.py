"""
Training pipeline for the LLaMA model.

This module implements the complete training loop with:
  - Mixed precision training (fp16/bf16/fp32 depending on hardware)
  - Gradient accumulation (simulate large batches on limited memory)
  - Cosine learning rate schedule with linear warmup
  - Gradient clipping (prevent gradient explosions)
  - Periodic evaluation on validation set
  - Checkpoint saving and resumption

TRAINING OVERVIEW:
  Language model training is surprisingly simple at its core:
    1. Feed the model a sequence of tokens
    2. The model predicts the next token at each position
    3. Compare predictions to actual next tokens (cross-entropy loss)
    4. Compute gradients of the loss with respect to all parameters
    5. Update parameters to reduce the loss
    6. Repeat for millions of steps

  The complexity comes from making this EFFICIENT and STABLE:
    - Mixed precision: Use lower precision (fp16/bf16) for speed
    - Gradient accumulation: Process data in small chunks, accumulate gradients
    - LR scheduling: Start slow (warmup), peak, then decay
    - Gradient clipping: Prevent catastrophically large updates
    - Checkpointing: Save progress so training can be resumed

THE TRAINING LOOP IN DETAIL:
  For each optimizer step:
    accumulated_loss = 0
    for micro_step in range(gradient_accumulation_steps):
      x, y = get_batch()                        # Load batch
      with autocast:                              # Mixed precision
        logits, loss = model(x, y)               # Forward pass
        loss = loss / gradient_accumulation_steps # Scale for accumulation
      scaler.scale(loss).backward()               # Backward pass
      accumulated_loss += loss.item()

    scaler.unscale_(optimizer)                   # Unscale gradients
    clip_grad_norm_(model.parameters(), max_norm) # Clip gradients
    scaler.step(optimizer)                        # Update weights
    scaler.update()                               # Adjust scale factor
    optimizer.zero_grad()                         # Reset gradients

  This processes gradient_accumulation_steps × batch_size sequences per
  optimizer step, giving an effective batch size of 256 × 512 = 131K tokens.
"""

import os
import math
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from llama_vc.config import ModelConfig, TrainConfig
from llama_vc.model import LLaMA
from llama_vc.tokenizer import Tokenizer
from llama_vc.dataset import prepare_data, create_dataloader
from llama_vc.device import (
    get_device,
    get_dtype,
    get_autocast_context,
    get_grad_scaler,
    device_info,
    get_memory_usage,
)
from llama_vc.utils import (
    set_seed,
    count_parameters,
    print_model_summary,
    save_checkpoint,
    load_checkpoint,
    TrainingLogger,
    Timer,
)


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    learning_rate: float,
    min_learning_rate: float,
) -> float:
    """
    Compute the learning rate for a given training step.

    LEARNING RATE SCHEDULE: Cosine Annealing with Linear Warmup

    This is the standard LR schedule used by LLaMA, GPT-3, and most modern LLMs.

    THREE PHASES:
      Phase 1 — Linear Warmup (steps 0 to warmup_steps):
        LR increases linearly from 0 to learning_rate.

        WHY: At the start of training, model parameters are randomly initialized.
        Gradients are noisy and point in somewhat random directions. A large
        learning rate would cause the model to take huge steps in random
        directions, potentially entering a bad region of the loss landscape
        that's hard to recover from.

        By starting with a small LR and gradually increasing, we let the model
        "find its footing" — the gradients become more consistent as the model
        learns basic patterns, and then we increase the step size.

        Formula: lr = learning_rate × (step / warmup_steps)

      Phase 2 — Cosine Decay (steps warmup_steps to max_steps):
        LR decreases following a cosine curve from learning_rate to min_learning_rate.

        WHY cosine (not linear or step decay)?
          - Cosine is smooth: no sudden LR changes that cause loss spikes
          - Slow initial decay: spends more time at high LR (maximum learning)
          - Slow final decay: gently settles into a minimum
          - Empirically produces better final loss than other schedules

        Formula:
          progress = (step - warmup_steps) / (max_steps - warmup_steps)
          lr = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × progress))

        At progress=0: cos(0) = 1, so lr = max_lr
        At progress=0.5: cos(π/2) = 0, so lr = (max_lr + min_lr) / 2
        At progress=1: cos(π) = -1, so lr = min_lr

      Phase 3 — Constant min_lr (steps > max_steps, if training continues):
        LR stays at min_learning_rate.
        WHY: Don't let LR go to 0. A small but non-zero LR allows the model
        to continue making small improvements.

    VISUALIZATION:
      LR │     ╱‾‾‾‾‾‾╲
         │    ╱         ╲
         │   ╱           ╲
         │  ╱             ╲______ min_lr
         │ ╱
         │╱
         └────────────────────── step
         0   warmup    max_steps

    Args:
        step: Current training step.
        warmup_steps: Number of warmup steps (500).
        max_steps: Total training steps (10000).
        learning_rate: Peak learning rate (3e-4).
        min_learning_rate: Minimum learning rate (3e-5).

    Returns:
        The learning rate for this step.
    """
    # Phase 1: Linear warmup
    if step < warmup_steps:
        return learning_rate * (step / warmup_steps)

    # Phase 3: Past max_steps, hold at minimum
    if step >= max_steps:
        return min_learning_rate

    # Phase 2: Cosine decay
    # progress goes from 0.0 (at warmup_steps) to 1.0 (at max_steps)
    progress = (step - warmup_steps) / (max_steps - warmup_steps)

    # Cosine multiplier: goes from 1.0 to 0.0 as progress goes 0→1
    # The (1 + cos(π * progress)) / 2 maps the cosine from [-1,1] to [0,1]
    cosine_multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))

    # Interpolate between max and min learning rate
    return min_learning_rate + (learning_rate - min_learning_rate) * cosine_multiplier


@torch.no_grad()
def evaluate(
    model: LLaMA,
    val_dataloader: DataLoader,
    device: torch.device,
    autocast_ctx,
    max_steps: int = 20,
) -> float:
    """
    Evaluate the model on the validation set.

    Computes the average cross-entropy loss over max_steps batches of
    validation data. This gives us a measure of how well the model
    generalizes to unseen text.

    @torch.no_grad(): Disables gradient computation during evaluation.
      This saves memory (no need to store activations for backward pass)
      and speeds up computation.

    WHAT THE VALIDATION LOSS TELLS US:
      - val_loss ≈ train_loss: Model generalizes well (not overfitting)
      - val_loss >> train_loss: Overfitting (memorizing training data)
      - val_loss < train_loss: Rare, usually means underfitting
      - Perplexity = exp(val_loss): Interpretable metric
        - Perplexity 1.0 = perfect prediction (impossible)
        - Perplexity 10 = model is "10-way confused" on average
        - Our target: perplexity ~3-5 (good for 15M on TinyStories)

    Args:
        model: The model to evaluate (set to eval mode).
        val_dataloader: DataLoader for validation data.
        device: Compute device.
        autocast_ctx: Mixed precision context manager.
        max_steps: Number of validation batches to evaluate.

    Returns:
        Average validation loss (float).
    """
    model.eval()
    total_loss = 0.0
    n_steps = 0

    for x, y in val_dataloader:
        if n_steps >= max_steps:
            break

        x = x.to(device)
        y = y.to(device)

        with autocast_ctx:
            _, loss = model(x, targets=y)

        total_loss += loss.item()
        n_steps += 1

    model.train()
    return total_loss / max(n_steps, 1)


def train(
    model_config: Optional[ModelConfig] = None,
    train_config: Optional[TrainConfig] = None,
    resume_from: Optional[str] = None,
    train_bin: Optional[str] = None,
    val_bin: Optional[str] = None,
) -> None:
    """
    Main training function.

    This orchestrates the entire training process:
      1. Setup: Device detection, model creation, optimizer
      2. Data: Load tokenizer, prepare tokenized data, create DataLoaders
      3. Training loop: Forward, backward, optimize, log, evaluate, checkpoint

    The function is designed to be called from scripts/train_model.py or
    directly from a notebook/REPL.

    Args:
        model_config: Model architecture config. If None, uses defaults.
        train_config: Training hyperparameters. If None, uses defaults.
        resume_from: Path to checkpoint file to resume training from.
        train_bin: Pre-resolved path to tokenized training data (.bin).
            If provided, skips data preparation.
        val_bin: Pre-resolved path to tokenized validation data (.bin).
            If provided, skips data preparation.
    """
    # ── Use defaults if not provided ───────────────────────────────────────
    if model_config is None:
        model_config = ModelConfig()
    if train_config is None:
        train_config = TrainConfig()

    # ── Setup ──────────────────────────────────────────────────────────────
    set_seed(train_config.seed)

    # Device and precision
    device = get_device()
    dtype = get_dtype(train_config.dtype, device)
    autocast_ctx = get_autocast_context(device, dtype)
    scaler = get_grad_scaler(device, dtype)

    # Logging
    logger = TrainingLogger(train_config.log_dir)
    logger.log_info(device_info(device))
    logger.log_info(f"Training dtype: {dtype}")
    logger.log_info(f"GradScaler: {'enabled' if scaler else 'disabled'}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    logger.log_info(f"Loading tokenizer from {train_config.tokenizer_path}")
    tokenizer = Tokenizer(train_config.tokenizer_path)

    # Override vocab_size if tokenizer has different size
    if tokenizer.vocab_size != model_config.vocab_size:
        logger.log_info(
            f"Updating vocab_size from {model_config.vocab_size} "
            f"to {tokenizer.vocab_size} (from tokenizer)"
        )
        model_config.vocab_size = tokenizer.vocab_size

    # ── Data Preparation ──────────────────────────────────────────────────
    if train_bin is not None and val_bin is not None:
        logger.log_info(f"Using provided data paths:")
        logger.log_info(f"  Train: {train_bin}")
        logger.log_info(f"  Val:   {val_bin}")
    else:
        logger.log_info("Preparing training data...")
        train_bin, val_bin = prepare_data(train_config.data_dir, tokenizer)

    # Create DataLoaders
    train_loader = create_dataloader(
        train_bin,
        seq_len=model_config.max_seq_len,
        batch_size=train_config.batch_size,
        shuffle=True,
        pin_memory=(device.type != "cpu"),
    )
    val_loader = create_dataloader(
        val_bin,
        seq_len=model_config.max_seq_len,
        batch_size=train_config.batch_size,
        shuffle=False,
        pin_memory=(device.type != "cpu"),
    )

    # ── Model ──────────────────────────────────────────────────────────────
    logger.log_info("Creating model...")
    model = LLaMA(model_config).to(device)

    n_params = count_parameters(model)
    logger.log_info(f"Model parameters: {n_params:,}")
    print_model_summary(model)

    # Optional: torch.compile for CUDA speedup
    # torch.compile traces the model and generates optimized CUDA kernels.
    # This can give 10-30% speedup but adds compilation time on first step.
    # NOT supported on MPS as of PyTorch 2.x.
    if train_config.compile_model and device.type == "cuda":
        logger.log_info("Compiling model with torch.compile()...")
        model = torch.compile(model)
    elif train_config.compile_model and device.type != "cuda":
        logger.log_info("torch.compile not supported on this device, skipping")

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = model.configure_optimizers(
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(train_config.beta1, train_config.beta2),
        device=device,
    )

    # ── Resume from checkpoint ─────────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")

    if resume_from and os.path.exists(resume_from):
        logger.log_info(f"Resuming from checkpoint: {resume_from}")
        ckpt_info = load_checkpoint(resume_from, model, optimizer, device)
        start_step = ckpt_info["step"]
        best_val_loss = ckpt_info["val_loss"]
        logger.log_info(f"Resumed at step {start_step}, val_loss {best_val_loss:.4f}")

    # ── Training Loop ──────────────────────────────────────────────────────
    logger.log_info(
        f"Starting training: {train_config.max_steps} steps, "
        f"batch_size={train_config.batch_size}, "
        f"grad_accum={train_config.gradient_accumulation_steps}, "
        f"effective_batch={train_config.batch_size * train_config.gradient_accumulation_steps}"
    )

    model.train()
    train_iter = iter(train_loader)
    tokens_per_step = (
        train_config.batch_size
        * train_config.gradient_accumulation_steps
        * model_config.max_seq_len
    )

    for step in range(start_step, train_config.max_steps):
        step_start = time.perf_counter()

        # ── Learning Rate Update ───────────────────────────────────────────
        # We update LR at the START of each step (before the optimizer step).
        # This is the convention used by most LLM training codebases.
        lr = get_lr(
            step,
            train_config.warmup_steps,
            train_config.max_steps,
            train_config.learning_rate,
            train_config.min_learning_rate,
        )
        # Apply the computed LR to all parameter groups in the optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Gradient Accumulation Loop ─────────────────────────────────────
        # Process multiple micro-batches, accumulating gradients, before
        # performing one optimizer step.
        #
        # WHY: We want an effective batch size of 256 sequences, but fitting
        # 256 × 512 tokens in GPU memory at once may not be possible.
        # Instead, we process 64 sequences × 4 micro-steps = 256 effective.
        #
        # The loss is divided by gradient_accumulation_steps so that the
        # accumulated gradient is equivalent to processing all 256 at once.
        accumulated_loss = 0.0

        for micro_step in range(train_config.gradient_accumulation_steps):
            # Get next batch (wrap around when DataLoader is exhausted)
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward pass with mixed precision
            with autocast_ctx:
                _, loss = model(x, targets=y)
                # Scale loss for gradient accumulation
                # If we don't do this, the accumulated gradient would be
                # gradient_accumulation_steps times too large
                loss = loss / train_config.gradient_accumulation_steps

            # Backward pass
            # If using GradScaler (fp16): scaler scales the loss up before
            # backward to prevent gradient underflow, then scales down before
            # the optimizer step.
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()

        # ── Gradient Clipping ──────────────────────────────────────────────
        # Clip the global gradient norm to prevent gradient explosions.
        # Gradient explosions can happen when:
        #   - The model encounters an unusual batch
        #   - LR is too high
        #   - Training is in an unstable region of the loss landscape
        #
        # How it works:
        #   total_norm = sqrt(sum(grad_i^2 for all parameters))
        #   if total_norm > max_norm:
        #     scale = max_norm / total_norm
        #     grad_i *= scale  (for all parameters)
        #
        # This preserves the gradient DIRECTION but limits its MAGNITUDE.
        if scaler is not None:
            # Unscale gradients before clipping (GradScaler scaled them up)
            scaler.unscale_(optimizer)

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), train_config.max_grad_norm
        )

        # ── Optimizer Step ─────────────────────────────────────────────────
        # Update all model parameters using the accumulated gradients.
        # AdamW: w_new = w - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)
        # Where m_hat and v_hat are bias-corrected first and second moments.
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()  # Adjust scale factor based on grad overflow detection
        else:
            optimizer.step()

        # Reset gradients to zero for the next accumulation cycle
        optimizer.zero_grad(set_to_none=True)
        # set_to_none=True: Sets .grad to None instead of zeroing.
        # This is slightly faster and uses less memory because:
        #   - No zero-filling operation
        #   - Memory for .grad is freed until next backward()

        # ── Timing ─────────────────────────────────────────────────────────
        # Synchronize GPU before measuring time (GPU ops are asynchronous)
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start
        tokens_per_sec = tokens_per_step / step_time

        # ── Logging ────────────────────────────────────────────────────────
        if step % train_config.log_interval == 0:
            mem = get_memory_usage(device)
            logger.log_step(
                step=step,
                loss=accumulated_loss,
                lr=lr,
                tokens_per_sec=tokens_per_sec,
                memory_mb=mem["allocated_mb"],
                total_steps=train_config.max_steps,
            )

        # ── Evaluation ─────────────────────────────────────────────────────
        if step > 0 and step % train_config.eval_interval == 0:
            val_loss = evaluate(
                model, val_loader, device, autocast_ctx,
                max_steps=train_config.eval_steps,
            )
            logger.log_eval(step, val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(
                    train_config.checkpoint_dir, "best.pt"
                )
                save_checkpoint(
                    model, optimizer, step, val_loss,
                    model_config.to_dict(), train_config.to_dict(),
                    best_path,
                )

        # ── Periodic Checkpoint ────────────────────────────────────────────
        if step > 0 and step % train_config.save_interval == 0:
            ckpt_path = os.path.join(
                train_config.checkpoint_dir, f"step_{step:06d}.pt"
            )
            save_checkpoint(
                model, optimizer, step, best_val_loss,
                model_config.to_dict(), train_config.to_dict(),
                ckpt_path,
            )

    # ── Final checkpoint ──────────────────────────────────────────────────
    final_path = os.path.join(train_config.checkpoint_dir, "final.pt")
    save_checkpoint(
        model, optimizer, train_config.max_steps, best_val_loss,
        model_config.to_dict(), train_config.to_dict(),
        final_path,
    )

    # Final evaluation
    val_loss = evaluate(
        model, val_loader, device, autocast_ctx,
        max_steps=train_config.eval_steps,
    )
    logger.log_eval(train_config.max_steps, val_loss)

    # Update best checkpoint if final eval is the best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_path = os.path.join(train_config.checkpoint_dir, "best.pt")
        save_checkpoint(
            model, optimizer, train_config.max_steps, val_loss,
            model_config.to_dict(), train_config.to_dict(),
            best_path,
        )

    logger.log_info("Training complete!")
    logger.log_info(f"Best validation loss: {best_val_loss:.4f}")
    logger.log_info(
        f"Best perplexity: {math.exp(best_val_loss):.2f}"
    )
    logger.close()
