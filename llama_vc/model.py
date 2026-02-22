"""
LLaMA Model Architecture — Complete Implementation from Scratch.

This is the CORE of the project. Every component of the LLaMA architecture
is implemented here with detailed explanations of:
  - WHAT it does
  - WHY LLaMA uses it (vs the vanilla transformer alternative)
  - HOW it works mathematically
  - The implementation details that matter

ARCHITECTURE OVERVIEW (bottom-up reading order):
  1. RMSNorm          — Normalization layer (replaces LayerNorm)
  2. RoPE             — Rotary Positional Embeddings (replaces learned/sinusoidal)
  3. FeedForward      — SwiGLU FFN (replaces ReLU FFN)
  4. Attention        — Grouped Query Attention with KV Cache
  5. TransformerBlock — One decoder layer combining attention + FFN
  6. LLaMA            — The full model stacking N TransformerBlocks

KEY DIFFERENCES FROM VANILLA TRANSFORMER (Vaswani et al., 2017):
  ┌─────────────────────┬──────────────────────┬────────────────────────┐
  │ Component           │ Vanilla Transformer  │ LLaMA                  │
  ├─────────────────────┼──────────────────────┼────────────────────────┤
  │ Normalization       │ LayerNorm (post)     │ RMSNorm (pre)          │
  │ Position encoding   │ Sinusoidal/learned   │ Rotary (RoPE)          │
  │ Activation          │ ReLU                 │ SwiGLU                 │
  │ Attention           │ Multi-Head (MHA)     │ Grouped Query (GQA)    │
  │ FFN hidden dim      │ 4 × dim              │ (8/3) × dim            │
  │ Bias terms          │ Yes                  │ No                     │
  │ Dropout             │ Yes (0.1)            │ No (0.0)               │
  │ Architecture        │ Encoder-Decoder      │ Decoder-only           │
  └─────────────────────┴──────────────────────┴────────────────────────┘

REFERENCES:
  - LLaMA: Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023)
  - LLaMA 2: Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023)
  - RMSNorm: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
  - RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
  - SwiGLU: Shazeer, "GLU Variants Improve Transformer" (2020)
  - GQA: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from llama_vc.config import ModelConfig


# ═══════════════════════════════════════════════════════════════════════════
# 1. RMSNorm — Root Mean Square Layer Normalization
# ═══════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    WHAT IT DOES:
      Normalizes the input tensor so that its root-mean-square (RMS) value
      is approximately 1, then scales by a learned parameter gamma.

    WHY LLaMA USES THIS INSTEAD OF LAYERNORM:
      LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
      RMSNorm(x)  = x / sqrt(mean(x²) + eps) * gamma

      RMSNorm removes two things:
        1. Mean centering (x - mean(x)): The mean subtraction re-centers
           the distribution. Empirically, this doesn't help much and costs
           one extra reduction operation per forward pass.
        2. Beta (shift) parameter: With mean centering gone, the shift
           parameter is also unnecessary.

      Benefits:
        - ~7-10% faster than LayerNorm (one fewer reduction)
        - Fewer parameters (no beta, only gamma)
        - Empirically equivalent performance to LayerNorm

    MATH:
      Given input x of shape (..., dim):

        rms(x) = sqrt( (1/dim) * sum(x_i²) + eps )
        RMSNorm(x) = (x / rms(x)) * gamma

      Where:
        - sum(x_i²) sums over the last dimension
        - eps (1e-5) prevents division by zero when x ≈ 0
        - gamma is a learned scale parameter of shape (dim,), initialized to 1.0

      The RMS acts as a normalization factor. After normalization:
        - The magnitude of x is standardized (RMS ≈ 1)
        - The direction of x is preserved (no mean centering)
        - gamma allows the network to learn the optimal scale per dimension

    WHERE IT'S USED IN LLAMA:
      Pre-normalization: applied BEFORE attention and BEFORE FFN in each layer.
      Also applied once after the final transformer layer, before the output
      projection. This is called "pre-norm" architecture.

      Pre-norm is more stable than post-norm for training deep networks because
      the residual connection carries the unnormalized signal, and gradients
      flow through it without being affected by normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Args:
            dim: The dimension to normalize over (last dimension of input).
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        # gamma (scale parameter): learned, initialized to ones.
        # Shape: (dim,) — one scale factor per dimension.
        # nn.Parameter makes it a trainable parameter that's included in
        # model.parameters() and saved in state_dict.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            x: Input tensor of shape (..., dim). The last dimension is normalized.
               Typical shapes: (batch, seq_len, dim) during training,
               (batch, 1, dim) during generation.

        Returns:
            Normalized tensor of the same shape as input.
        """
        # Step 1: Compute the RMS (root mean square) of x along the last dimension
        #
        # x.float(): Cast to float32 for numerical stability during norm computation.
        # Even in mixed precision training, normalization should be done in fp32
        # to avoid precision issues. torch.amp.autocast handles this automatically
        # for LayerNorm, but since we're implementing RMSNorm manually, we need
        # to do it explicitly.
        #
        # x.pow(2): Square each element → x²
        # .mean(-1, keepdim=True): Average over last dim → mean(x²), shape (..., 1)
        # + self.eps: Add epsilon for numerical stability
        # torch.rsqrt: Compute 1/sqrt(x) — faster than sqrt then divide
        rms_inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

        # Step 2: Normalize and scale
        # x * rms_inv: Normalize so RMS ≈ 1
        # * self.weight: Scale by learned gamma
        # .type_as(x): Cast back to the original dtype (e.g., float16/bfloat16)
        return (x.float() * rms_inv).type_as(x) * self.weight


# ═══════════════════════════════════════════════════════════════════════════
# 2. Rotary Positional Embeddings (RoPE)
# ═══════════════════════════════════════════════════════════════════════════

def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin tables for Rotary Positional Embeddings.

    WHAT IS ROPE?
      RoPE (Rotary Position Embedding) encodes position information by
      ROTATING query and key vectors. Unlike traditional approaches that
      ADD position information to embeddings, RoPE applies a rotation in
      2D subspaces of the head dimension.

    WHY ROTATION INSTEAD OF ADDITION?
      Traditional: embedding = token_embedding + position_embedding
        - Absolute positions: token at position 5 always gets the same encoding
        - Doesn't naturally capture RELATIVE positions (distance between tokens)

      RoPE: q_rotated = rotate(q, position)
        - The attention score dot(q_m, k_n) depends on (m - n), the RELATIVE
          distance between the query and key positions
        - This emerges naturally from the mathematics of rotation, without any
          special design for relative positions
        - No learned parameters needed (positions are encoded through geometry)

    MATHEMATICAL DERIVATION:
      Consider a head_dim-dimensional vector. We split it into head_dim/2 pairs
      of consecutive dimensions: (d₀,d₁), (d₂,d₃), ..., (d_{head_dim-2}, d_{head_dim-1})

      Each pair is treated as a 2D vector and rotated by a position-dependent angle.

      For the i-th pair at position m:
        θᵢ = theta^(-2i/head_dim)    where theta=10000 (base frequency)

        The rotation angle is: m × θᵢ

        Applied as a 2D rotation matrix:
          [d_{2i}']     [cos(m·θᵢ)  -sin(m·θᵢ)] [d_{2i}  ]
          [d_{2i+1}'] = [sin(m·θᵢ)   cos(m·θᵢ)] [d_{2i+1}]

      FREQUENCY INTERPRETATION:
        - Low i (first dimensions): θᵢ is large → slow rotation → encodes LONG-range positions
        - High i (last dimensions): θᵢ is small → fast rotation → encodes SHORT-range positions
        - This creates a multi-scale position encoding, similar to sinusoidal encodings
          but applied multiplicatively through rotation

      KEY PROPERTY — RELATIVE POSITION ENCODING:
        When computing attention: score = dot(q_rotated_m, k_rotated_n)

        dot(R(θ,m)·q, R(θ,n)·k)
          = dot(q, R(θ,m)ᵀ · R(θ,n) · k)     [rotation matrices are orthogonal]
          = dot(q, R(θ, n-m) · k)              [R(θ,m)ᵀ · R(θ,n) = R(θ, n-m)]
          = f(q, k, n-m)                        [depends only on relative position!]

        This means the attention score between positions m and n depends ONLY
        on the difference (n-m), not on the absolute positions. The model
        naturally learns relative attention patterns.

      DECAY PROPERTY:
        RoPE naturally creates a decay in attention with distance. Positions
        that are far apart have larger rotation angle differences, which tends
        to reduce the dot product. This gives the model an inductive bias
        toward attending to nearby tokens, which is linguistically sensible.

    PRECOMPUTATION:
      Since the cos/sin values depend only on position and dimension (not on
      the input data), we compute them once and store them as buffers.
      During the forward pass, we just look up the values we need.

    Args:
        head_dim: Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length to precompute for.
        theta: Base frequency. Default 10000.0 (original RoPE/LLaMA v1).
               LLaMA 3 uses 500000.0 for better long-context performance.
        device: Device to create tensors on.

    Returns:
        Tuple of (freqs_cos, freqs_sin), each of shape (max_seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"

    # Step 1: Compute the frequency for each dimension pair
    # freqs[i] = theta^(-2i/head_dim) = 1 / theta^(2i/head_dim)
    #
    # For head_dim=64: i = 0, 1, 2, ..., 31 (32 pairs)
    #   freqs[0]  = 1/10000^(0/64)  = 1.0        (slowest rotation)
    #   freqs[1]  = 1/10000^(2/64)  ≈ 0.72
    #   freqs[15] = 1/10000^(30/64) ≈ 0.013
    #   freqs[31] = 1/10000^(62/64) ≈ 0.00015    (fastest rotation)
    #
    # This creates a geometric sequence of frequencies, each pair rotating
    # at a different rate. The slowest-rotating dimensions capture long-range
    # position information, while the fastest capture fine-grained positions.
    dim_indices = torch.arange(0, head_dim, 2, device=device).float()
    freqs = 1.0 / (theta ** (dim_indices / head_dim))
    # freqs shape: (head_dim // 2,)

    # Step 2: Compute the rotation angles for each position
    # For position m, the angle for dimension pair i is: m * freqs[i]
    # positions shape: (max_seq_len,)
    positions = torch.arange(max_seq_len, device=device).float()

    # Outer product: all (position, frequency) combinations
    # angles[m, i] = m * freqs[i] = the rotation angle at position m for pair i
    angles = torch.outer(positions, freqs)
    # angles shape: (max_seq_len, head_dim // 2)

    # Step 3: Precompute cos and sin of the angles
    # These are the rotation matrix components that will be applied to Q and K
    freqs_cos = angles.cos()
    freqs_sin = angles.sin()
    # Each has shape: (max_seq_len, head_dim // 2)

    return freqs_cos, freqs_sin


def apply_rotary_embeddings(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply precomputed rotary embeddings to query or key tensors.

    This is where the actual rotation happens. Each consecutive pair of
    dimensions (d₀,d₁), (d₂,d₃), etc. is rotated by the position-dependent
    angle computed in precompute_rope_frequencies.

    THE ROTATION FORMULA:
      For each pair (x₀, x₁) at position m with angle θ:
        x₀' = x₀ · cos(θ) - x₁ · sin(θ)
        x₁' = x₀ · sin(θ) + x₁ · cos(θ)

      This is exactly the 2D rotation matrix:
        [x₀']   [cos(θ)  -sin(θ)] [x₀]
        [x₁'] = [sin(θ)   cos(θ)] [x₁]

    IMPLEMENTATION TRICK:
      Instead of explicitly pairing dimensions, we:
        1. Reshape x from (..., head_dim) to (..., head_dim//2, 2)
        2. Split into x_even = x[..., 0] and x_odd = x[..., 1]
        3. Apply: x_even' = x_even * cos - x_odd * sin
                  x_odd'  = x_even * sin + x_odd * cos
        4. Stack and flatten back to (..., head_dim)

      This is equivalent to the rotation but vectorized efficiently.

    IMPORTANT: RoPE is applied to Q and K only, NOT to V.
      Why? The rotation encodes position in the attention score (Q·K dot product).
      V carries the content that gets aggregated — it doesn't need position
      encoding because the attention weights (from Q·K) already encode which
      positions to attend to.

    Args:
        x: Query or Key tensor of shape (batch, seq_len, n_heads, head_dim).
        freqs_cos: Precomputed cos values, shape (seq_len, head_dim // 2).
        freqs_sin: Precomputed sin values, shape (seq_len, head_dim // 2).

    Returns:
        Rotated tensor of the same shape as input.
    """
    # x shape: (batch, seq_len, n_heads, head_dim)

    # Step 1: Reshape to expose dimension pairs
    # (..., head_dim) → (..., head_dim//2, 2)
    # The last dimension of 2 contains each (even, odd) pair
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    # x_reshaped shape: (batch, seq_len, n_heads, head_dim//2, 2)

    # Step 2: Split into even and odd dimensions
    x_even = x_reshaped[..., 0]  # (batch, seq_len, n_heads, head_dim//2)
    x_odd = x_reshaped[..., 1]   # (batch, seq_len, n_heads, head_dim//2)

    # Step 3: Broadcast cos/sin to match x's shape
    # freqs_cos/sin: (seq_len, head_dim//2) → need (1, seq_len, 1, head_dim//2)
    # The 1s allow broadcasting across batch and n_heads dimensions
    cos = freqs_cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
    sin = freqs_sin.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)

    # Step 4: Apply the 2D rotation
    # This is the core of RoPE: rotate each (even, odd) pair
    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot = x_even * sin + x_odd * cos

    # Step 5: Interleave back to original layout
    # Stack along last dim → (..., head_dim//2, 2), then flatten → (..., head_dim)
    x_rotated = torch.stack([x_even_rot, x_odd_rot], dim=-1)
    x_rotated = x_rotated.flatten(-2)  # (..., head_dim)

    return x_rotated.type_as(x)


# ═══════════════════════════════════════════════════════════════════════════
# 3. SwiGLU Feed-Forward Network
# ═══════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network (Shazeer, 2020).

    WHAT IT DOES:
      The FFN is the "thinking" part of each transformer layer. While attention
      gathers information from different positions, the FFN processes the
      information at each position independently, applying learned nonlinear
      transformations.

    WHY SWIGLU INSTEAD OF STANDARD FFN:
      Standard FFN (vanilla transformer):
        FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂
        - 2 weight matrices, 2 biases
        - ReLU activation: max(0, x)
        - Simple but less expressive

      SwiGLU FFN (LLaMA):
        FFN(x) = (Swish(x·W_gate) ⊙ (x·W_up)) · W_down
        - 3 weight matrices, NO biases
        - Swish activation: x·σ(x) where σ is sigmoid
        - ⊙ is element-wise multiplication (the "gating")
        - More expressive due to the gating mechanism

    THE GATING MECHANISM (what makes GLU special):
      The key insight is the element-wise multiplication (⊙):
        gate = Swish(x·W_gate)    → produces a "gate" signal in [~-0.28, +∞)
        up   = x·W_up             → produces the actual content
        out  = gate ⊙ up          → gate controls what content passes through

      Think of it like a soft switch at each dimension:
        - gate ≈ 0: block this information
        - gate ≈ 1: pass this information through
        - gate > 1: amplify this information (Swish can exceed 1)

      This selective filtering makes the network more expressive than a
      simple nonlinearity. The network learns WHAT to compute (W_up) and
      WHETHER to use it (W_gate) independently.

    SWISH ACTIVATION:
      Swish(x) = x · σ(x)  where σ(x) = 1/(1 + e^(-x))

      Also known as SiLU (Sigmoid Linear Unit) in PyTorch: F.silu(x)

      Properties:
        - Smooth everywhere (unlike ReLU's kink at 0)
        - Non-monotonic: slightly dips below 0 for negative inputs
        - Self-gated: the sigmoid σ(x) gates the linear term x
        - Empirically outperforms ReLU and GELU in many settings

      Comparison:
        ReLU(x)  = max(0, x)           — hard cutoff at 0
        GELU(x)  = x · Φ(x)           — Φ is the Gaussian CDF
        Swish(x) = x · σ(x)           — smooth, slight negative values

    PARAMETER COUNT:
      Standard FFN (hidden = 4·dim):
        2 matrices: W₁(dim, 4·dim) + W₂(4·dim, dim) = 8·dim² params
        For dim=384: 8 × 384² = 1,179,648 params

      SwiGLU FFN (hidden = (8/3)·dim):
        3 matrices: W_gate(dim, h) + W_up(dim, h) + W_down(h, dim) = 3·dim·h
        With h = (8/3)·dim ≈ 1024: 3 × 384 × 1024 = 1,179,648 params

      SAME total parameter count! SwiGLU achieves better performance by
      distributing the same number of parameters across 3 matrices with
      a gating mechanism, rather than 2 matrices with a simple nonlinearity.

    NO BIAS:
      Following LLaMA, all linear layers omit bias terms. This is a minor
      simplification that:
        - Reduces parameter count slightly
        - Has negligible impact on model quality
        - Is standard practice in modern LLMs
    """

    def __init__(self, config: ModelConfig):
        """
        Args:
            config: ModelConfig with dim and hidden_dim.
        """
        super().__init__()

        # W_gate: Projects input to gate space
        # Shape: (dim, hidden_dim) = (384, 1024)
        # The gate output is passed through Swish activation
        self.w_gate = nn.Linear(config.dim, config.hidden_dim, bias=False)

        # W_up: Projects input to "content" space
        # Shape: (dim, hidden_dim) = (384, 1024)
        # This content is then filtered by the gate
        self.w_up = nn.Linear(config.dim, config.hidden_dim, bias=False)

        # W_down: Projects back from hidden space to model dimension
        # Shape: (hidden_dim, dim) = (1024, 384)
        # This is the "output projection" of the FFN
        self.w_down = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SwiGLU FFN.

        Data flow:
          x (batch, seq, 384)
            ├─→ w_gate → Swish → gate (batch, seq, 1024)
            └─→ w_up          → up   (batch, seq, 1024)
                    gate ⊙ up → (batch, seq, 1024)
                    w_down    → (batch, seq, 384)

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Output tensor of shape (batch, seq_len, dim).
        """
        # Compute the gate signal: Swish(x·W_gate)
        # F.silu is PyTorch's implementation of Swish/SiLU: x * sigmoid(x)
        gate = F.silu(self.w_gate(x))

        # Compute the content: x·W_up (no activation — the gate handles nonlinearity)
        up = self.w_up(x)

        # Apply gating: element-wise multiply gate and content
        # gate values near 0 block the corresponding content dimensions
        # gate values near 1+ pass/amplify the content
        gated = gate * up

        # Project back to model dimension
        return self.w_down(gated)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Grouped Query Attention (GQA) with KV Cache
# ═══════════════════════════════════════════════════════════════════════════

class Attention(nn.Module):
    """
    Multi-Head Attention with Grouped Query Attention (GQA) and KV Cache.

    WHAT IS ATTENTION?
      Attention is the mechanism that allows each token to "look at" every
      other token in the sequence and gather relevant information. It answers:
      "For this token, which other tokens should I pay attention to?"

      Three projections create the attention inputs:
        Q (Query):  "What am I looking for?"
        K (Key):    "What do I contain?" (matched against queries)
        V (Value):  "What information do I provide?" (weighted and aggregated)

      Attention score: how much token i attends to token j
        score(i,j) = softmax(Q_i · K_j / √d_k)

      Output: weighted sum of values
        output_i = Σ_j score(i,j) · V_j

    WHAT IS GROUPED QUERY ATTENTION (GQA)?
      Standard MHA:  6 Q heads, 6 K heads, 6 V heads → 18 sets of projections
      GQA (3:1):     6 Q heads, 2 K heads, 2 V heads → 10 sets of projections
      MQA (6:1):     6 Q heads, 1 K head,  1 V head  →  8 sets of projections

      In GQA, multiple query heads SHARE the same Key and Value head:
        Q heads [0, 1, 2] share KV head 0  (Group 0)
        Q heads [3, 4, 5] share KV head 1  (Group 1)

      WHY THIS HELPS:
        1. Memory: The KV cache during inference stores past K,V values.
           With GQA, the cache is n_heads/n_kv_heads = 3× smaller.
           For our tiny model: negligible savings.
           For LLaMA 70B: this means ~60GB vs ~180GB of KV cache!

        2. Parameters: Fewer K,V projection matrices = fewer parameters.
           Wk: dim×(n_kv_heads×head_dim) = 384×128 instead of 384×384
           Saved: ~200K params per layer (modest for our model)

        3. Speed: Fewer KV computations during inference.
           The attention computation itself uses PyTorch's optimized
           scaled_dot_product_attention which handles GQA efficiently.

        4. Quality: Empirically, GQA with ratio ≤4:1 matches MHA quality
           while being more efficient. The intuition is that K,V heads
           capture "content" that can be shared across multiple "viewpoints"
           (query heads), similar to how a single document can answer
           multiple different questions.

    CAUSAL MASKING:
      In decoder-only models (like LLaMA), each token can only attend to
      tokens at the SAME or EARLIER positions. This is called "causal"
      masking because it enforces the causal constraint: the prediction
      for position i depends only on positions 0..i, not future positions.

      Implementation: We use a triangular mask where mask[i,j] = -inf for j > i.
      After softmax, e^(-inf) = 0, so future positions get zero attention weight.

      PyTorch's scaled_dot_product_attention handles this with is_causal=True,
      which automatically applies the causal mask using an efficient fused kernel.

    KV CACHE (inference optimization):
      During autoregressive generation (producing one token at a time):

      WITHOUT cache (naive):
        Step 1: Process "The"         → compute Q,K,V for 1 token, attend over 1 token
        Step 2: Process "The cat"     → compute Q,K,V for 2 tokens, attend over 2 tokens
        Step 3: Process "The cat sat" → compute Q,K,V for 3 tokens, attend over 3 tokens
        ...
        Step N: recompute everything for N tokens → O(N²) total work to generate N tokens

      WITH cache:
        Step 1: Process "The"         → compute Q₁,K₁,V₁, cache K₁,V₁
        Step 2: Process "cat"         → compute Q₂,K₂,V₂, cache K₂,V₂, attend Q₂ to [K₁,K₂]
        Step 3: Process "sat"         → compute Q₃,K₃,V₃, cache K₃,V₃, attend Q₃ to [K₁,K₂,K₃]
        ...
        Step N: compute only 1 new Q,K,V, attend to cached K,V → O(N) total work!

      The KV cache stores previous K,V tensors so we never recompute them.
      For our model, the cache is tiny (~2MB), but for large models it's the
      primary memory bottleneck during inference.

    FLASH ATTENTION:
      PyTorch's F.scaled_dot_product_attention automatically selects the most
      efficient attention implementation:
        - FlashAttention-2: O(N) memory instead of O(N²), uses tiling to keep
          data in fast SRAM. Available on CUDA (Ampere+).
        - Memory-efficient attention: Similar benefits, broader hardware support.
        - Standard math attention: Fallback when others aren't available.

      We get this optimization for FREE by using the PyTorch API instead of
      implementing attention manually. No need to install flash-attn separately.
    """

    def __init__(self, config: ModelConfig):
        """
        Args:
            config: ModelConfig with dim, n_heads, n_kv_heads, etc.
        """
        super().__init__()

        self.n_heads = config.n_heads          # Number of query heads (6)
        self.n_kv_heads = config.n_kv_heads    # Number of KV heads (2)
        self.head_dim = config.head_dim        # Dimension per head (64)
        self.n_kv_groups = config.n_kv_groups  # Queries per KV group (3)

        # Query projection: maps input to all query heads
        # Input: (batch, seq, dim=384) → Output: (batch, seq, n_heads*head_dim=384)
        # Since n_heads * head_dim = dim in our config, Wq is square (384×384)
        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)

        # Key projection: maps input to KV heads (fewer than query heads for GQA)
        # Input: (batch, seq, dim=384) → Output: (batch, seq, n_kv_heads*head_dim=128)
        # Much smaller than Wq because n_kv_heads (2) < n_heads (6)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)

        # Value projection: same size as Key projection
        self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)

        # Output projection: maps concatenated head outputs back to model dimension
        # Input: (batch, seq, n_heads*head_dim=384) → Output: (batch, seq, dim=384)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of Grouped Query Attention.

        Data flow:
          x (batch, seq, 384)
            ├─→ Wq → reshape → RoPE → Q (batch, seq, 6, 64) → transpose → (batch, 6, seq, 64)
            ├─→ Wk → reshape → RoPE → K (batch, seq, 2, 64) → [cache] → transpose → (batch, 2, kv_len, 64)
            └─→ Wv → reshape        → V (batch, seq, 2, 64) → [cache] → transpose → (batch, 2, kv_len, 64)

          Attention: SDPA(Q, K, V) with GQA broadcasting
            → (batch, 6, seq, 64) → transpose → (batch, seq, 384) → Wo → (batch, seq, 384)

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            freqs_cos: RoPE cos frequencies for the current positions.
            freqs_sin: RoPE sin frequencies for the current positions.
            mask: Optional attention mask. None = causal masking via is_causal flag.
            kv_cache: Optional (cached_k, cached_v) from previous steps.
                     Each has shape (batch, cache_len, n_kv_heads, head_dim).
            start_pos: Position offset for KV cache (inference only).

        Returns:
            Tuple of:
              - Output tensor of shape (batch, seq_len, dim)
              - Updated KV cache tuple, or None if not caching
        """
        batch_size, seq_len, _ = x.shape

        # ── Step 1: Project to Q, K, V ─────────────────────────────────────
        # Linear projections (no activation — attention IS the nonlinearity)
        q = self.wq(x)  # (batch, seq, n_heads * head_dim)
        k = self.wk(x)  # (batch, seq, n_kv_heads * head_dim)
        v = self.wv(x)  # (batch, seq, n_kv_heads * head_dim)

        # ── Step 2: Reshape into multiple heads ────────────────────────────
        # Split the last dimension into (n_heads, head_dim) for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # q: (batch, seq, 6, 64)
        # k: (batch, seq, 2, 64)
        # v: (batch, seq, 2, 64)

        # ── Step 3: Apply RoPE to Q and K (NOT V) ─────────────────────────
        # RoPE encodes position through rotation. Applied only to Q and K
        # because position information is encoded in the attention SCORES (Q·K),
        # not in the values that get aggregated.
        q = apply_rotary_embeddings(q, freqs_cos, freqs_sin)
        k = apply_rotary_embeddings(k, freqs_cos, freqs_sin)

        # ── Step 4: KV Cache handling (inference) ──────────────────────────
        # kv_cache can be:
        #   - None: No caching (training or first layer call without caching)
        #   - (cached_k, cached_v): Existing cache to append to
        # use_cache flag (from caller) determines if we should return new cache
        new_kv_cache = None
        if kv_cache is not None:
            # Subsequent inference step: append new K, V to existing cache
            cached_k, cached_v = kv_cache
            # cached_k shape: (batch, prev_len, n_kv_heads, head_dim)
            # k shape: (batch, seq_len, n_kv_heads, head_dim) where seq_len=1 during generation

            # Concatenate along the sequence dimension
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

        # Always return (k, v) as cache so caller can store it for next step.
        # The caller (TransformerBlock/LLaMA) decides whether to actually use it.
        new_kv_cache = (k, v)

        # ── Step 5: Transpose for attention computation ────────────────────
        # PyTorch's SDPA expects: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch, n_kv_heads, kv_len, head_dim)
        v = v.transpose(1, 2)  # (batch, n_kv_heads, kv_len, head_dim)
        # Note: kv_len >= seq_len when using cache (includes cached history)

        # ── Step 6: GQA head expansion ─────────────────────────────────────
        # For GQA, we need to expand K and V to match the number of Q heads.
        # Each KV head is shared by n_kv_groups query heads.
        #
        # Two approaches:
        #   a) Explicit repeat: k = k.repeat_interleave(n_kv_groups, dim=1)
        #      Pro: Simple. Con: Allocates 3× more memory for K,V.
        #
        #   b) PyTorch SDPA with enable_gqa=True (PyTorch 2.5+):
        #      Pro: Handles broadcasting internally, no extra memory.
        #      Con: Requires recent PyTorch version.
        #
        # We implement approach (a) for maximum compatibility, with a note
        # about (b) for when the user has a recent enough PyTorch.
        if self.n_kv_heads != self.n_heads:
            # Repeat each KV head n_kv_groups times to match Q heads
            # Before: k shape (batch, 2, kv_len, 64)
            # After:  k shape (batch, 6, kv_len, 64)
            # Pattern: [KV0, KV0, KV0, KV1, KV1, KV1]
            k = k.repeat_interleave(self.n_kv_groups, dim=1)
            v = v.repeat_interleave(self.n_kv_groups, dim=1)

        # ── Step 7: Scaled Dot-Product Attention ───────────────────────────
        # score(Q, K) = softmax(Q·K^T / √d_k) · V
        #
        # The scaling factor √d_k prevents the dot products from growing too
        # large as head_dim increases. Without scaling, softmax would produce
        # near-one-hot distributions (one position gets all attention).
        #
        # F.scaled_dot_product_attention is PyTorch's optimized implementation.
        # It automatically selects the best kernel:
        #   - FlashAttention-2: Tiled, O(N) memory, fastest on A100/H100
        #   - Memory-efficient: Good for longer sequences
        #   - Math: Fallback implementation
        #
        # is_causal=True: Automatically applies causal mask (triangle mask)
        #   where position i can only attend to positions ≤ i.
        #   This is more efficient than passing an explicit mask because
        #   FlashAttention can fuse the mask into the computation.
        #
        # When we have a custom mask (e.g., during cached inference where
        # is_causal doesn't apply correctly), we pass it explicitly.
        if mask is not None:
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )
        else:
            output = F.scaled_dot_product_attention(
                q, k, v, is_causal=(seq_len > 1)
            )
        # output shape: (batch, n_heads, seq_len, head_dim) = (batch, 6, seq, 64)

        # ── Step 8: Concatenate heads and project output ───────────────────
        # Transpose back: (batch, n_heads, seq, head_dim) → (batch, seq, n_heads, head_dim)
        # Then flatten heads: (batch, seq, n_heads * head_dim) = (batch, seq, 384)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Output projection: mix information across heads
        # Wo: (n_heads*head_dim, dim) = (384, 384) — maps combined heads to model dim
        return self.wo(output), new_kv_cache


# ═══════════════════════════════════════════════════════════════════════════
# 5. Transformer Block (One Decoder Layer)
# ═══════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    A single LLaMA transformer decoder layer.

    ARCHITECTURE:
      Input x (from previous layer or embedding)
        │
        ├─→ RMSNorm ─→ Attention ─→ + ─→ (attention output)
        │                            │
        └────────────────────────────┘  ← Residual connection
        │
        ├─→ RMSNorm ─→ FeedForward ─→ + ─→ Output (to next layer)
        │                              │
        └──────────────────────────────┘  ← Residual connection

    KEY DESIGN CHOICES:

      1. PRE-NORMALIZATION (norm BEFORE attention/FFN):
         Vanilla transformer (post-norm):
           x = LayerNorm(x + Attention(x))    ← norm AFTER residual add
         LLaMA (pre-norm):
           x = x + Attention(RMSNorm(x))      ← norm BEFORE the sublayer

         Why pre-norm is better:
           - In post-norm, gradients must flow THROUGH the normalization layer
             during backpropagation. This can distort gradient magnitudes.
           - In pre-norm, the residual connection provides a "gradient highway":
             gradients flow directly through the addition, bypassing the sublayer.
           - This makes training more stable, especially for deep models (32+ layers).
           - Pre-norm models typically don't need learning rate warmup as much
             (though we still use it as a best practice).

      2. TWO RESIDUAL CONNECTIONS per layer:
         The residual stream is the central "highway" that carries information
         through the model. Attention and FFN are additive modifications to
         this stream.

         After L layers:
           output = x + attn_1(x) + ffn_1(x) + attn_2(x) + ffn_2(x) + ... + attn_L(x) + ffn_L(x)

         Each sublayer adds a refinement. The original signal is never destroyed,
         only modified. This is crucial for training deep networks:
           - Gradient can flow directly from loss to any layer via the residual
           - No vanishing gradient problem (unlike pre-residual architectures)
           - Each layer can focus on computing a useful DELTA to add

      3. NO DROPOUT:
         LLaMA uses no dropout at all. The reasoning is:
           - With enough training data (1-2T tokens for real LLaMA), dropout
             regularization is unnecessary — the data itself regularizes.
           - Dropout during training is equivalent to adding noise, which helps
             prevent memorization. But with diverse enough data, the model
             doesn't memorize anyway.
           - For our TinyStories experiment (~300M tokens), this is fine because
             the dataset is diverse enough for our tiny 15M model.
    """

    def __init__(self, layer_id: int, config: ModelConfig):
        """
        Args:
            layer_id: Index of this layer (0 to n_layers-1). Used for logging
                     and for scaling the output projection initialization.
            config: ModelConfig with all architecture hyperparameters.
        """
        super().__init__()
        self.layer_id = layer_id

        # Pre-attention normalization
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

        # Multi-head attention with GQA
        self.attention = Attention(config)

        # Pre-FFN normalization
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

        # SwiGLU Feed-Forward Network
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of one transformer layer.

        Data flow:
          x → [RMSNorm → Attention → + residual] → [RMSNorm → FFN → + residual] → output

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            freqs_cos, freqs_sin: RoPE frequencies for current positions.
            mask: Optional attention mask.
            kv_cache: Optional KV cache for inference.
            start_pos: Position offset for KV cache.

        Returns:
            Tuple of (output tensor, updated KV cache or None).
        """
        # ── Attention sublayer with residual connection ────────────────────
        # 1. Normalize (pre-norm: normalize BEFORE attention)
        # 2. Apply attention
        # 3. Add residual (skip connection from input)
        normed = self.attention_norm(x)
        attn_output, new_kv_cache = self.attention(
            normed, freqs_cos, freqs_sin, mask, kv_cache, start_pos
        )
        x = x + attn_output  # Residual connection

        # ── FFN sublayer with residual connection ──────────────────────────
        # Same pattern: normalize → process → add residual
        normed = self.ffn_norm(x)
        ffn_output = self.feed_forward(normed)
        x = x + ffn_output  # Residual connection

        return x, new_kv_cache


# ═══════════════════════════════════════════════════════════════════════════
# 6. Complete LLaMA Model
# ═══════════════════════════════════════════════════════════════════════════

class LLaMA(nn.Module):
    """
    Complete LLaMA decoder-only transformer language model.

    FULL ARCHITECTURE:
      Token IDs (batch, seq_len)
        │
        ▼
      Token Embedding: id → vector of dim=384
        │
        ▼
      8× TransformerBlock:
        │  RMSNorm → GQA Attention (with RoPE) → + residual
        │  RMSNorm → SwiGLU FFN → + residual
        │
        ▼
      Final RMSNorm
        │
        ▼
      Output Projection: vector → logits over vocabulary
        │
        ▼
      Logits (batch, seq_len, vocab_size=4096)

    IMPORTANT OBSERVATIONS:

      1. NO POSITIONAL EMBEDDING LAYER:
         Unlike GPT-2/3 which have a learned position embedding table,
         LLaMA has NO separate position encoding. Positions are encoded
         ENTIRELY through RoPE, which is applied inside each attention layer.
         This means the model has no maximum sequence length hard-coded into
         the parameters (though RoPE's effectiveness does degrade beyond
         the training context length).

      2. DECODER-ONLY ARCHITECTURE:
         LLaMA is a decoder-only model (like GPT), not encoder-decoder
         (like the original Transformer or T5). This means:
           - Only causal (left-to-right) attention
           - No cross-attention layers
           - The model is trained on next-token prediction only
           - Input and output share the same "sequence" dimension
         Decoder-only is simpler and has dominated the LLM landscape
         since GPT-2 demonstrated its effectiveness.

      3. LANGUAGE MODEL HEAD:
         The output projection maps from the model dimension (384) to the
         vocabulary size (4096). Each element of the output represents the
         model's confidence that the corresponding token comes next.

         These raw scores are called "logits" (unnormalized log-probabilities).
         To get actual probabilities: probs = softmax(logits).
         During training, we compute cross-entropy loss directly from logits
         (more numerically stable than softmax + log).

    WEIGHT INITIALIZATION:
      Proper initialization is critical for stable training. Bad init can
      cause gradients to vanish or explode from the very first step.

      We follow the GPT-2 / nanoGPT convention:
        - All weight matrices: Normal(mean=0, std=0.02)
        - Embedding: Normal(mean=0, std=0.02)
        - RMSNorm weight (gamma): initialized to 1.0 (default)
        - Output projection of attention (wo): scaled by 1/√(2·n_layers)

      WHY scale wo?
        The residual stream accumulates outputs from each layer:
          x = x + attn_out_1 + ffn_out_1 + attn_out_2 + ...
        Without scaling, the variance of x grows with depth (each addition
        adds variance). Scaling wo by 1/√(2·n_layers) ensures the total
        contribution from all layers has roughly unit variance.
        The factor 2 accounts for both attention and FFN per layer.

    GRADIENT CHECKPOINTING:
      When enabled (config.use_gradient_checkpointing=True), intermediate
      activations inside transformer blocks are NOT stored during the
      forward pass. Instead, they are RECOMPUTED during the backward pass.

      Trade-off:
        - Memory: ~60-70% less activation memory (huge for large models)
        - Compute: ~30% more time (recomputing is cheap vs storing in memory)

      For our 15M model, this is unnecessary (activations are tiny).
      For training 7B+ models on limited GPU memory, it's essential.

    MODEL PARALLELISM HOOKS:
      While we implement single-device training, the architecture is designed
      to be easily adapted for model parallelism:
        - Each TransformerBlock is a self-contained unit that can be placed
          on a different device (pipeline parallelism)
        - The attention projections can be sharded across devices
          (tensor parallelism: split heads across GPUs)
        - The vocabulary embedding can be sharded (vocabulary parallelism)

      For real large-scale training, you'd use PyTorch FSDP (Fully Sharded
      Data Parallel) or Megatron-LM style parallelism. Our clean modular
      design makes this straightforward to add later.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the LLaMA model.

        Args:
            config: ModelConfig with all architecture hyperparameters.
                   The config is validated and stored for checkpoint serialization.
        """
        super().__init__()
        config.validate()
        self.config = config

        # ── Token Embedding ────────────────────────────────────────────────
        # Maps each token ID (integer) to a dense vector of dim=384.
        # This is a lookup table of shape (vocab_size, dim).
        # During forward pass: embedding[token_id] → vector
        # Parameters: 4096 × 384 = 1,572,864
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # ── Transformer Layers ─────────────────────────────────────────────
        # Stack of identical transformer blocks. Each block refines the
        # representation through attention (gather information from context)
        # and FFN (process information at each position).
        self.layers = nn.ModuleList([
            TransformerBlock(layer_id=i, config=config)
            for i in range(config.n_layers)
        ])

        # ── Final Normalization ────────────────────────────────────────────
        # One final RMSNorm after all transformer layers.
        # This stabilizes the output before the final projection.
        self.norm = RMSNorm(config.dim, config.norm_eps)

        # ── Output Projection (Language Model Head) ────────────────────────
        # Maps from model dimension (384) to vocabulary size (4096).
        # Produces logits: one score per vocabulary token.
        # Parameters: 384 × 4096 = 1,572,864
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # ── Weight Tying (optional) ────────────────────────────────────────
        # Share the token embedding matrix with the output projection.
        # This means output.weight IS the same tensor as tok_embeddings.weight.
        #
        # Intuition: The embedding learns "what does this token mean?" and
        # the output projection learns "which token best matches this meaning?"
        # These are conceptually inverse operations, so sharing weights makes sense.
        #
        # When weight_tying=True, the shared weights get gradient contributions
        # from both the embedding lookup and the output projection, which can
        # improve learning for both tasks.
        if config.weight_tying:
            self.output.weight = self.tok_embeddings.weight

        # ── Precompute RoPE frequencies ────────────────────────────────────
        # These are fixed values (not learned), computed once and stored as
        # buffers. register_buffer ensures they are:
        #   - Saved in state_dict (for checkpoint loading)
        #   - Moved to the correct device with model.to(device)
        #   - NOT included in model.parameters() (not trainable)
        freqs_cos, freqs_sin = precompute_rope_frequencies(
            config.head_dim, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # ── Initialize weights ─────────────────────────────────────────────
        self.apply(self._init_weights)

        # Apply special scaled initialization to output projections
        # (attention wo and ffn w_down) to prevent residual stream growth
        for layer in self.layers:
            # Scale factor: 1/√(2 * n_layers)
            # The 2 accounts for both attention and FFN per layer
            scale = 1.0 / math.sqrt(2 * config.n_layers)
            nn.init.normal_(layer.attention.wo.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(layer.feed_forward.w_down.weight, mean=0.0, std=0.02 * scale)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights using the GPT-2 / nanoGPT convention.

        This is called via model.apply() which recursively visits every submodule.

        INITIALIZATION STRATEGY:
          - Linear layers: Normal(0, 0.02)
            Why 0.02? It's a reasonable scale that prevents activations from
            being too large or too small at initialization. The exact value
            matters less than being in the right ballpark (0.01 - 0.1).

          - Embedding: Normal(0, 0.02)
            Same reasoning as linear layers.

          - RMSNorm: Already initialized to ones by __init__ (no change needed).

        WHY NOT Xavier/Kaiming init?
          Those are designed for specific activation functions (sigmoid/ReLU).
          For transformers with pre-norm and residual connections, the simple
          Normal(0, 0.02) initialization works well empirically and is the
          standard across GPT-2, GPT-3, LLaMA, and most modern LLMs.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
        start_pos: int = 0,
    ):
        """
        Forward pass of the full LLaMA model.

        TRAINING MODE (kv_caches=None):
          Input: tokens of shape (batch, seq_len) — token IDs
          Output: (logits, loss)
            - logits: (batch, seq_len, vocab_size) — predictions for each position
            - loss: scalar — cross-entropy loss averaged over all positions, or None

        INFERENCE MODE (kv_caches is a list):
          Input: tokens of shape (batch, seq_len) — can be 1 token with KV cache
          Output: (logits, loss, new_kv_caches)
            - logits: (batch, seq_len, vocab_size) — only the last position matters
            - loss: None (no targets during inference)
            - new_kv_caches: Updated KV caches for next decode step

        Args:
            tokens: Input token IDs, shape (batch, seq_len).
                   Values in [0, vocab_size).
            targets: Target token IDs for loss computation, shape (batch, seq_len).
                    targets[i, j] = the correct next token for position j.
                    In practice, targets = tokens shifted by 1 position.
            kv_caches: Controls caching behavior:
                      - None: No caching (training mode). Returns (logits, loss).
                      - List of (K, V) tuples or Nones: Caching enabled (inference).
                        Pass [None]*n_layers for the first call to initialize caches.
                        Returns (logits, loss, new_kv_caches).
            start_pos: Position offset when using KV cache. The new tokens
                      start at this position in the sequence.

        Returns:
            Training: Tuple of (logits, loss)
            Inference: Tuple of (logits, loss, new_kv_caches)
        """
        batch_size, seq_len = tokens.shape

        # ── Step 1: Token Embedding ────────────────────────────────────────
        # Look up the embedding vector for each token ID
        # tokens: (batch, seq_len) of integers → h: (batch, seq_len, dim=384)
        h = self.tok_embeddings(tokens)

        # ── Step 2: Get RoPE frequencies for current positions ─────────────
        # During training: positions 0..seq_len-1
        # During inference with cache: positions start_pos..start_pos+seq_len-1
        freqs_cos = self.freqs_cos[start_pos: start_pos + seq_len]
        freqs_sin = self.freqs_sin[start_pos: start_pos + seq_len]

        # ── Step 3: Pass through all transformer layers ────────────────────
        use_cache = kv_caches is not None
        new_kv_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            # Get cached KV for this layer (if any)
            # kv_caches[i] can be None (first call) or (K, V) tuple (subsequent calls)
            layer_kv_cache = kv_caches[i] if use_cache else None

            if self.config.use_gradient_checkpointing and self.training:
                # Gradient checkpointing: Don't store intermediate activations.
                # They will be recomputed during backward pass.
                # NOTE: checkpoint requires the function not to have any
                # non-tensor arguments that affect the output, so we wrap
                # the layer call to handle the kv_cache and start_pos.
                def create_custom_forward(module):
                    def custom_forward(h, freqs_cos, freqs_sin):
                        return module(h, freqs_cos, freqs_sin, None, None, 0)
                    return custom_forward

                h_out, _ = gradient_checkpoint(
                    create_custom_forward(layer),
                    h, freqs_cos, freqs_sin,
                    use_reentrant=False,
                )
                h = h_out
                if use_cache:
                    new_kv_caches.append(None)
            else:
                h, new_kv = layer(h, freqs_cos, freqs_sin, None, layer_kv_cache, start_pos)
                if use_cache:
                    new_kv_caches.append(new_kv)

        # ── Step 4: Final normalization ────────────────────────────────────
        h = self.norm(h)

        # ── Step 5: Output projection → logits ────────────────────────────
        # h: (batch, seq_len, dim=384) → logits: (batch, seq_len, vocab_size=4096)
        # Each logit[b, t, v] represents the model's confidence that token v
        # is the correct next token at position t in batch b.
        logits = self.output(h)

        # ── Step 6: Compute loss (training only) ──────────────────────────
        loss = None
        if targets is not None:
            # Cross-entropy loss: measures how well the predicted probability
            # distribution matches the true next token.
            #
            # CE(y, ŷ) = -log(ŷ[y_true])
            #
            # Where ŷ[y_true] is the predicted probability of the correct token.
            # Lower loss = model assigns higher probability to correct tokens.
            #
            # F.cross_entropy expects:
            #   input: (N, C) where N = batch*seq_len, C = vocab_size
            #   target: (N,) where each value is in [0, vocab_size)
            #
            # We reshape to combine batch and sequence dimensions.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (batch*seq, vocab_size)
                targets.view(-1),                   # (batch*seq,)
                ignore_index=-1,  # Ignore padding tokens (if any)
            )

        # Return format depends on whether caching is enabled
        # Training: (logits, loss) — simple tuple
        # Inference: (logits, loss, kv_caches) — includes updated caches
        if use_cache:
            return logits, loss, new_kv_caches
        return logits, loss

    def configure_optimizers(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: Tuple[float, float],
        device: torch.device,
    ) -> torch.optim.AdamW:
        """
        Create an AdamW optimizer with separate parameter groups.

        WHY SEPARATE PARAMETER GROUPS?
          Weight decay is a regularization technique that shrinks weights
          toward zero: w_new = w - lr * (grad + weight_decay * w)

          But weight decay should NOT be applied to all parameters:
            - 1D parameters (RMSNorm gamma, biases): These have specific
              semantic meaning. Decaying them toward zero would remove the
              normalization scale or constant offset.
            - Embeddings: While technically 2D, weight decay on embeddings
              is controversial. We include them in the decay group following
              the LLaMA/GPT convention, but some practitioners exclude them.

          Weight decay SHOULD be applied to:
            - 2D weight matrices (attention projections, FFN weights):
              These benefit from the regularization, which prevents any
              single weight from becoming too large.

        Args:
            learning_rate: Peak learning rate.
            weight_decay: L2 regularization strength for decayed params.
            betas: (beta1, beta2) for Adam momentum/velocity decay.
            device: Target device (for fused optimizer on CUDA).

        Returns:
            Configured AdamW optimizer.
        """
        # Separate parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Apply weight decay only to 2D parameters (weight matrices)
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Report parameter group sizes
        n_decay = sum(p.numel() for p in decay_params)
        n_no_decay = sum(p.numel() for p in no_decay_params)
        print(f"Optimizer parameter groups:")
        print(f"  Decay params:    {n_decay:>12,d} ({len(decay_params)} tensors)")
        print(f"  No-decay params: {n_no_decay:>12,d} ({len(no_decay_params)} tensors)")

        # Use fused AdamW on CUDA for ~5-10% speedup (fuses all param updates
        # into a single kernel launch instead of one per parameter)
        use_fused = device.type == "cuda"
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused,
        )

        return optimizer
