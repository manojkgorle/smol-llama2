"""
Configuration for the LLaMA model and training pipeline.

This module is the SINGLE SOURCE OF TRUTH for all hyperparameters. No magic
numbers should appear anywhere else in the codebase — everything is defined
here and imported where needed.

We use Python dataclasses for configuration because:
  1. Type safety: IDE can catch typos and type mismatches.
  2. Immutability: frozen=False but we treat them as read-only after creation.
  3. Serialization: Easy to save/load with dataclasses.asdict().
  4. Defaults: Every parameter has a sensible default for our ~15M param model.

ARCHITECTURE OVERVIEW:
  Our model follows LLaMA (Touvron et al., 2023) with these key specs:
  - Decoder-only transformer (no encoder, no cross-attention)
  - Pre-normalization with RMSNorm (not LayerNorm)
  - Rotary Positional Embeddings (RoPE, not learned/sinusoidal)
  - SwiGLU activation in FFN (not ReLU/GELU)
  - Grouped Query Attention (GQA, not standard MHA)
  - No bias terms in any linear layer
  - No dropout (relies on data scale for regularization)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import os


@dataclass
class ModelConfig:
    """
    Architecture hyperparameters for the LLaMA model.

    These parameters define the model's capacity and structure. Changing any
    of these creates a fundamentally different model that cannot load weights
    from a differently-configured model.

    PARAMETER COUNT BREAKDOWN (with defaults):
    ─────────────────────────────────────────────
    Token Embedding (vocab_size × dim):        1,572,864  (10.0%)
    8 Transformer Layers:                     12,589,056  (80.0%)
      Per layer:
        Wq (dim × dim):                         147,456
        Wk (dim × n_kv_heads*head_dim):           49,152
        Wv (dim × n_kv_heads*head_dim):           49,152
        Wo (dim × dim):                          147,456
        W_gate (dim × hidden_dim):               393,216
        W_up (dim × hidden_dim):                 393,216
        W_down (hidden_dim × dim):               393,216
        2× RMSNorm (dim each):                       768
        Layer total:                           1,573,632
    Final RMSNorm (dim):                             384  (0.0%)
    Output Projection (dim × vocab_size):      1,572,864  (10.0%)
    ─────────────────────────────────────────────
    TOTAL:                                    15,735,168  (15.74M)
    """

    # ── Vocabulary ──────────────────────────────────────────────────────────
    # The number of unique tokens the model can process. Our BPE tokenizer
    # uses 4096 tokens, which is sufficient for TinyStories (simple English).
    # Larger vocab = more expressive tokenization but larger embedding matrix.
    # Real LLaMA uses 32000 (v1/v2) or 128256 (v3).
    vocab_size: int = 4096

    # ── Model Dimensions ───────────────────────────────────────────────────
    # The "width" of the model — the size of the hidden state vector that
    # flows through the residual stream. Every token is represented as a
    # vector of this dimension throughout the model.
    # 384 is chosen because: 384 / 6 heads = 64 head_dim (standard for RoPE).
    # Real LLaMA uses 4096 (7B), 5120 (13B), 8192 (65B/70B).
    dim: int = 384

    # ── Depth ──────────────────────────────────────────────────────────────
    # Number of stacked transformer decoder layers. More layers = deeper
    # feature hierarchy, but more parameters and slower training/inference.
    # 8 layers is a sweet spot for ~15M params with dim=384.
    # Real LLaMA uses 32 (7B), 40 (13B), 80 (65B/70B).
    n_layers: int = 8

    # ── Attention Heads ────────────────────────────────────────────────────
    # n_heads: Number of QUERY attention heads. Each head independently
    # attends to different parts of the sequence, learning different
    # "attention patterns" (e.g., one head might focus on recent tokens,
    # another on syntactically related tokens).
    # head_dim = dim / n_heads = 384 / 6 = 64
    n_heads: int = 6

    # n_kv_heads: Number of KEY and VALUE heads for Grouped Query Attention.
    # When n_kv_heads < n_heads, multiple query heads share the same K,V pair.
    # This is GQA (Grouped Query Attention), introduced in LLaMA 2.
    #   - n_kv_heads == n_heads: standard Multi-Head Attention (MHA)
    #   - n_kv_heads == 1: Multi-Query Attention (MQA, most aggressive sharing)
    #   - 1 < n_kv_heads < n_heads: GQA (the middle ground LLaMA uses)
    # Our config: 6 query heads share 2 KV heads → ratio 3:1
    # Each KV head serves a GROUP of 3 query heads.
    # Benefits: 3× smaller KV cache during inference, fewer parameters.
    n_kv_heads: int = 2

    # ── Sequence Length ────────────────────────────────────────────────────
    # Maximum number of tokens the model can process in one forward pass.
    # This determines the size of precomputed RoPE frequencies and the
    # maximum KV cache size during inference.
    # 512 is adequate for TinyStories (avg ~150 tokens per story).
    # Real LLaMA: 2048 (v1), 4096 (v2), 8192+ (v3).
    max_seq_len: int = 512

    # ── Feed-Forward Network ───────────────────────────────────────────────
    # The intermediate (hidden) dimension of the SwiGLU FFN.
    # In standard transformers, FFN hidden = 4 × dim.
    # In SwiGLU, we use 3 matrices instead of 2, so to keep the same total
    # parameter count, hidden_dim ≈ (8/3) × dim = (8/3) × 384 ≈ 1024.
    # We round to 1024 (a power of 2) for hardware efficiency.
    # Formula: standard 4×dim = 1536 params/matrix × 2 matrices = 3072 total
    #          SwiGLU (8/3)×dim = 1024 params/matrix × 3 matrices = 3072 total
    hidden_dim: int = 1024

    # ── Normalization ──────────────────────────────────────────────────────
    # Epsilon for RMSNorm to prevent division by zero.
    # This is added inside the square root: 1/sqrt(mean(x²) + eps).
    # 1e-5 is the standard value used by LLaMA.
    norm_eps: float = 1e-5

    # ── Positional Encoding ────────────────────────────────────────────────
    # Base frequency for Rotary Positional Embeddings (RoPE).
    # Controls how quickly the rotation angles change across dimensions.
    # Higher theta = positions are encoded at lower frequencies = better
    # extrapolation to longer sequences (but may hurt short-range patterns).
    # 10000.0 is the original RoPE/LLaMA v1 value.
    # LLaMA 3 uses 500000.0 for better long-context performance.
    rope_theta: float = 10000.0

    # ── Regularization ─────────────────────────────────────────────────────
    # Dropout probability. LLaMA uses 0.0 (no dropout) because the training
    # data is large enough to act as a regularizer. For our tiny model on
    # TinyStories (~2M stories), this is also fine.
    # If you're overfitting on a small custom dataset, try 0.1.
    dropout: float = 0.0

    # ── Memory Optimization ────────────────────────────────────────────────
    # Gradient checkpointing trades compute for memory: during the backward
    # pass, intermediate activations are recomputed instead of stored.
    # ~30% more compute but ~60-70% less activation memory.
    # Not needed for our tiny model, but essential for training large models.
    use_gradient_checkpointing: bool = False

    # ── Weight Tying ───────────────────────────────────────────────────────
    # Whether to share the token embedding matrix with the output projection.
    # If True: output logits = hidden_state @ embedding_weight.T
    #   Saves vocab_size × dim = 4096 × 384 = 1,572,864 parameters.
    #   Total would be ~14.16M instead of ~15.74M.
    # If False: separate output projection matrix (more expressive).
    # Large LLaMA models do NOT tie weights. We default to False.
    weight_tying: bool = False

    @property
    def head_dim(self) -> int:
        """
        Dimension of each attention head.

        Computed as dim / n_heads. This MUST divide evenly.
        head_dim=64 is standard and optimal for RoPE (needs even dimension
        for the rotation pairs) and for hardware efficiency (64 is a nice
        power of 2 that maps well to GPU warp sizes).
        """
        assert self.dim % self.n_heads == 0, (
            f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        )
        return self.dim // self.n_heads

    @property
    def n_kv_groups(self) -> int:
        """
        Number of query heads per KV head group.

        With n_heads=6, n_kv_heads=2: each KV head serves 3 query heads.
        This is the "grouping" in Grouped Query Attention.
        Must divide evenly: n_heads must be a multiple of n_kv_heads.
        """
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by "
            f"n_kv_heads ({self.n_kv_heads})"
        )
        return self.n_heads // self.n_kv_heads

    def validate(self) -> None:
        """
        Validate configuration constraints.

        Called before model creation to catch configuration errors early
        rather than getting cryptic shape mismatch errors deep in the model.
        """
        assert self.dim > 0, "dim must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.n_kv_heads > 0, "n_kv_heads must be positive"
        assert self.n_kv_heads <= self.n_heads, (
            f"n_kv_heads ({self.n_kv_heads}) cannot exceed n_heads ({self.n_heads})"
        )
        assert self.dim % self.n_heads == 0, (
            f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        assert self.head_dim % 2 == 0, (
            f"head_dim ({self.head_dim}) must be even for RoPE rotation pairs"
        )
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization (e.g., saving in checkpoints)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Reconstruct from dictionary (e.g., loading from checkpoints)."""
        return cls(**d)

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


@dataclass
class TrainConfig:
    """
    Training hyperparameters.

    These control HOW the model is trained, not its architecture. Changing
    these does not affect model loading — you can resume training with
    different training hyperparameters from the same checkpoint.

    TRAINING BUDGET ESTIMATE:
    ─────────────────────────
    Dataset: TinyStories ≈ 300M tokens (estimated)
    Tokens per step: batch_size × grad_accum × seq_len = 64 × 4 × 512 = 131,072
    Steps for 1 epoch: 300M / 131K ≈ 2,290 steps
    10,000 steps ≈ 4.4 epochs

    Measured wall time:
      CUDA (A100):   ~15 min
      MPS (M4 Pro):  ~3.5 hours
      CUDA (T4):     OOM at default settings — reduce batch_size/grad_accum
    """

    # ── Batch Size ─────────────────────────────────────────────────────────
    # Number of sequences processed in one forward/backward pass (micro-batch).
    # The EFFECTIVE batch size = batch_size × gradient_accumulation_steps.
    # Larger batches = more stable gradients but more memory.
    # 64 sequences × 512 tokens = 32,768 tokens per micro-batch.
    batch_size: int = 64

    # ── Gradient Accumulation ──────────────────────────────────────────────
    # Number of micro-batches to accumulate before performing an optimizer step.
    # This simulates a larger batch size without requiring more memory.
    # Effective batch = 64 × 4 = 256 sequences = 131,072 tokens per step.
    # HOW IT WORKS: Each micro-batch computes loss/grad_accum_steps and calls
    # .backward(). Gradients accumulate in .grad attributes. After all micro-
    # batches, we clip gradients and call optimizer.step().
    gradient_accumulation_steps: int = 4

    # ── Training Duration ──────────────────────────────────────────────────
    # Total number of optimizer steps (not micro-batches).
    # 10,000 steps × 131,072 tokens/step = ~1.3B tokens processed.
    # This is about 4.4 epochs over TinyStories.
    max_steps: int = 10_000

    # ── Learning Rate Schedule ─────────────────────────────────────────────
    # We use cosine annealing with linear warmup, the standard for LLMs.
    #
    # Schedule visualization (not to scale):
    #   LR │    ╱‾‾‾‾╲
    #      │   ╱       ╲
    #      │  ╱         ╲___
    #      │ ╱           min_lr
    #      └──────────────────→ step
    #        ↑warmup    ↑cosine decay
    #
    # WHY warmup: At initialization, model parameters are random and gradients
    # are noisy. Large learning rates would cause destructive updates. Warmup
    # gradually increases LR so the model can find a reasonable region of the
    # loss landscape before taking large steps.
    #
    # WHY cosine decay: Smoothly reduces LR as training progresses. Empirically
    # produces better final loss than step/linear decay. The slow reduction near
    # the end allows fine-grained refinement of the solution.
    warmup_steps: int = 500
    learning_rate: float = 3e-4     # Peak learning rate (after warmup)
    min_learning_rate: float = 3e-5  # Floor LR = 10% of peak (prevents stalling)

    # ── Optimizer: AdamW ───────────────────────────────────────────────────
    # AdamW is the standard optimizer for transformer training.
    # Key difference from Adam: weight decay is applied DIRECTLY to weights,
    # not through the gradient (decoupled weight decay regularization).
    #
    # beta1: Exponential decay rate for first moment (mean of gradients).
    #   0.9 = each step uses 90% of previous momentum + 10% of current gradient.
    #   This smooths out noisy gradients by maintaining a running average.
    #
    # beta2: Exponential decay rate for second moment (variance of gradients).
    #   0.95 (LLaMA) vs 0.999 (default). Lower beta2 adapts faster to gradient
    #   magnitude changes, which helps with the non-stationary loss landscape
    #   of language model training.
    #
    # weight_decay: L2 regularization strength applied to weight matrices only.
    #   Encourages smaller weights, reducing overfitting. 0.1 is standard.
    #   IMPORTANT: Only applied to 2D parameters (weight matrices), NOT to
    #   1D parameters (RMSNorm gamma, biases if any, embeddings).
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    # ── Gradient Clipping ──────────────────────────────────────────────────
    # Maximum L2 norm for the gradient vector (across all parameters).
    # If ||grad|| > max_grad_norm, gradients are scaled down proportionally.
    # Prevents gradient explosions that can destroy training, especially
    # important in the early phase when the loss landscape is rugged.
    max_grad_norm: float = 1.0

    # ── Mixed Precision ────────────────────────────────────────────────────
    # Controls the floating-point precision used during training.
    # "auto": Automatically selects the best dtype for the device:
    #   - CUDA (Ampere+): bfloat16 (no GradScaler needed)
    #   - CUDA (older): float16 (needs GradScaler to prevent underflow)
    #   - MPS: float32 with float16 autocast
    #   - CPU: float32
    # Can be overridden to "float16", "bfloat16", or "float32".
    dtype: str = "auto"

    # ── torch.compile ──────────────────────────────────────────────────────
    # Whether to use torch.compile() to JIT-compile the model.
    # This can give 10-30% speedup on CUDA by fusing operations.
    # NOT supported on MPS as of PyTorch 2.x, so we auto-disable on Mac.
    compile_model: bool = False

    # ── Evaluation ─────────────────────────────────────────────────────────
    # How often (in optimizer steps) to evaluate on the validation set.
    eval_interval: int = 250
    # How many batches to use for each evaluation (for speed).
    eval_steps: int = 20

    # ── Checkpointing ──────────────────────────────────────────────────────
    # How often to save model checkpoints (in optimizer steps).
    save_interval: int = 1000

    # ── Logging ────────────────────────────────────────────────────────────
    # How often to print training metrics to console (in optimizer steps).
    log_interval: int = 10

    # ── Reproducibility ────────────────────────────────────────────────────
    # Random seed for torch, numpy, and Python's random module.
    # Setting this ensures deterministic data ordering and weight initialization.
    # NOTE: Full determinism on GPU requires torch.use_deterministic_algorithms(True)
    # which we don't enable by default as it can significantly slow down training.
    seed: int = 42

    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: str = "data/"
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"
    tokenizer_path: str = "data/tokenizer.model"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        """Reconstruct from dictionary."""
        return cls(**d)
