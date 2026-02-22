"""
Inference pipeline: text generation with KV cache and sampling strategies.

This module implements autoregressive text generation — the process by which
an LLM produces text one token at a time:
  1. Encode the prompt into tokens
  2. Feed the prompt through the model (PREFILL phase)
  3. Sample the next token from the model's predictions
  4. Feed that token back into the model (DECODE phase)
  5. Repeat steps 3-4 until done

TWO-PHASE GENERATION:

  PREFILL PHASE (processing the prompt):
    The entire prompt is fed through the model in one forward pass.
    This populates the KV cache with all prompt token representations.
    We get predictions for the last prompt token (which tells us the
    first token to generate).

    Example: Prompt = "Once upon a"
      Forward pass: model(["Once", "upon", "a"])
      KV cache now stores K,V for all 3 positions
      Logits for last position → sample "time" as first generated token

  DECODE PHASE (generating new tokens):
    Each new token is generated one at a time. Only the single new token
    is fed through the model; the KV cache provides the context.

    Step 1: model("time", cache=[K,V for "Once upon a"])
            → sample "," → append to sequence
    Step 2: model(",", cache=[K,V for "Once upon a time"])
            → sample "there" → append to sequence
    ...and so on until EOS or max_new_tokens

  The KV cache is what makes this efficient: without it, we'd need to
  reprocess the entire sequence at every step (O(N²) total work).
  With cache, we process only 1 new token per step (O(N) total work).

SAMPLING STRATEGIES:
  The model outputs logits (raw scores) for each vocabulary token.
  How we convert these logits into a token choice is the "sampling strategy".

  1. TEMPERATURE: Controls randomness
     - logits = logits / temperature
     - temperature < 1.0: Sharper distribution → more deterministic
     - temperature > 1.0: Flatter distribution → more random/creative
     - temperature = 0: Greedy decoding (always pick the most likely token)

  2. TOP-K: Restrict to K most likely tokens
     - Only consider the top K tokens by probability
     - All other tokens get probability 0
     - Prevents sampling from the "long tail" of unlikely tokens

  3. TOP-P (Nucleus Sampling): Adaptive restriction
     - Find the smallest set of tokens whose cumulative probability ≥ p
     - More adaptive than top-k: narrows when model is confident,
       widens when model is uncertain

  Applied in order: temperature → top-k → top-p → sample
"""

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from typing import Optional

from llama_vc.config import ModelConfig
from llama_vc.model import LLaMA
from llama_vc.tokenizer import Tokenizer


@dataclass
class GenerateResult:
    """Result of text generation with inference metrics."""
    text: str
    prompt_tokens: int      # number of tokens in the prompt (including BOS)
    generated_tokens: int   # number of tokens generated
    prefill_ms: float       # time to process the prompt (ms)
    decode_ms: float        # time spent in the decode loop (ms)
    total_ms: float         # total wall time (ms)
    peak_memory_mb: float   # peak GPU memory during generation (0 if CPU)
    temperature: float      # sampling temperature used
    top_k: int              # top-k value used
    top_p: float            # top-p value used

    @property
    def ttft_ms(self) -> float:
        """Time to first token — same as prefill time."""
        return self.prefill_ms

    @property
    def decode_tok_per_sec(self) -> float:
        """Decode throughput (tokens/sec), excluding prefill."""
        if self.decode_ms <= 0:
            return 0.0
        return self.generated_tokens / (self.decode_ms / 1000)

    @property
    def overall_tok_per_sec(self) -> float:
        """Overall throughput including prefill."""
        if self.total_ms <= 0:
            return 0.0
        return self.generated_tokens / (self.total_ms / 1000)

    def stats_string(self) -> str:
        """Formatted summary of inference metrics."""
        lines = [
            f"Sampling       : temp={self.temperature}, top_k={self.top_k}, top_p={self.top_p}",
            f"Prompt tokens  : {self.prompt_tokens}",
            f"Output tokens  : {self.generated_tokens}",
            f"TTFT           : {self.ttft_ms:.1f} ms",
            f"Decode speed   : {self.decode_tok_per_sec:.1f} tok/s",
            f"Overall speed  : {self.overall_tok_per_sec:.1f} tok/s",
            f"Total time     : {self.total_ms:.1f} ms",
        ]
        if self.peak_memory_mb > 0:
            lines.append(f"Peak GPU mem   : {self.peak_memory_mb:.1f} MB")
        return "\n".join(lines)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Top-p (nucleus) sampling: keep tokens whose cumulative probability ≥ p.

    HOW IT WORKS:
      1. Sort tokens by probability (descending)
      2. Compute cumulative sum of probabilities
      3. Find the cutoff point where cumsum first exceeds p
      4. Zero out all tokens after the cutoff
      5. Renormalize so the remaining probabilities sum to 1
      6. Sample from the filtered distribution

    EXAMPLE:
      probs = [0.4, 0.3, 0.15, 0.1, 0.05]  (sorted descending)
      cumsum = [0.4, 0.7, 0.85, 0.95, 1.0]
      p = 0.9 → cutoff at index 3 (cumsum first exceeds 0.9)
      kept = [0.4, 0.3, 0.15, 0.1] → renormalized

    WHY TOP-P IS BETTER THAN TOP-K:
      Top-k with k=10 always keeps 10 tokens, regardless of how confident
      the model is. If the model is very confident (one token has 99%
      probability), top-k still keeps 10 tokens.

      Top-p adapts:
        - Confident prediction: only 1-2 tokens might exceed p=0.9
        - Uncertain prediction: many tokens needed to reach p=0.9
      This gives more natural and coherent text.

    Args:
        probs: Probability distribution of shape (vocab_size,).
        p: Cumulative probability threshold (0.0 to 1.0).

    Returns:
        Sampled token index as a tensor.
    """
    # Sort probabilities in descending order
    probs_sorted, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative sum
    cumsum = torch.cumsum(probs_sorted, dim=-1)

    # Create mask: True for tokens to REMOVE (cumsum exceeds p)
    # We shift cumsum right by 1 so the token that crosses p is kept
    # Without shift: the crossing token itself would be removed
    mask = cumsum - probs_sorted > p

    # Zero out removed tokens
    probs_sorted[mask] = 0.0

    # Renormalize
    probs_sorted /= probs_sorted.sum()

    # Sample from filtered distribution
    sampled_idx = torch.multinomial(probs_sorted, num_samples=1)

    # Map back to original indices (squeeze to scalar for consistency)
    return sorted_indices[sampled_idx].squeeze(0)


@torch.inference_mode()
def generate(
    model: LLaMA,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate text from a prompt using the trained LLaMA model.

    @torch.inference_mode():
      More efficient than torch.no_grad() — also disables autograd's
      version counting and other tracking, giving a small speedup.
      Used whenever we know we won't need gradients (all of inference).

    GENERATION ALGORITHM:
      1. Encode prompt → tokens
      2. PREFILL: Forward entire prompt, get KV caches + logits
      3. Sample first new token from last position's logits
      4. DECODE LOOP:
         a. Forward new token (1 token) with KV caches
         b. Sample next token from logits
         c. If EOS or max tokens reached: stop
         d. Update KV caches, go to (a)
      5. Decode all generated tokens → text

    Args:
        model: Trained LLaMA model (in eval mode).
        tokenizer: Tokenizer for encode/decode.
        prompt: Input text to continue from.
        max_new_tokens: Maximum tokens to generate (stopping condition).
        temperature: Sampling temperature.
            0.0 = greedy (deterministic), 1.0 = default, >1.0 = more random.
        top_k: Number of top tokens to consider (0 = disabled).
        top_p: Nucleus sampling threshold (1.0 = disabled).
        device: Device to run on. If None, uses model's device.

    Returns:
        GenerateResult with generated text and inference metrics.
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Track peak GPU memory if on CUDA
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    t_start = time.perf_counter()

    # ── Step 1: Encode the prompt ──────────────────────────────────────────
    # BOS (Beginning Of Sequence) signals to the model that this is the start
    # of a new generation. No EOS because we want the model to continue.
    prompt_tokens = tokenizer.encode(prompt, bos=True, eos=False)
    tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    # tokens shape: (1, prompt_len) — batch size of 1

    prompt_len = tokens.shape[1]

    # ── Step 2: PREFILL phase ──────────────────────────────────────────────
    # Process the entire prompt in one forward pass.
    # This populates the KV caches and gives us logits for the last position.
    # We pass [None]*n_layers to signal "initialize fresh KV caches".
    n_layers = model.config.n_layers
    logits, _, kv_caches = model(tokens, kv_caches=[None] * n_layers, start_pos=0)
    # logits shape: (1, prompt_len, vocab_size)
    # kv_caches: list of (K, V) per layer, K/V shape: (1, prompt_len, n_kv_heads, head_dim)

    if use_cuda:
        torch.cuda.synchronize(device)
    t_prefill = time.perf_counter()

    # We only need logits at the LAST position (for predicting the next token)
    next_logits = logits[:, -1, :]  # (1, vocab_size)

    # Track the starting position for the decode phase
    # New tokens will be at positions prompt_len, prompt_len+1, etc.
    cur_pos = prompt_len

    # ── Step 3-4: DECODE loop ──────────────────────────────────────────────
    generated_tokens = []

    for _ in range(max_new_tokens):
        # ── Sample the next token ──────────────────────────────────────────
        next_token = _sample_token(next_logits, temperature, top_k, top_p)
        generated_tokens.append(next_token.item())

        # ── Check stopping condition ───────────────────────────────────────
        if next_token.item() == tokenizer.eos_id:
            break

        # ── Forward the new token through the model ────────────────────────
        # Only process 1 token (the one we just sampled).
        # The KV caches provide all the context from previous tokens.
        new_token_tensor = next_token.unsqueeze(0).unsqueeze(0)  # (1, 1)
        logits, _, kv_caches = model(
            new_token_tensor,
            kv_caches=kv_caches,
            start_pos=cur_pos,
        )
        # logits shape: (1, 1, vocab_size)
        next_logits = logits[:, -1, :]  # (1, vocab_size)
        cur_pos += 1

    if use_cuda:
        torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    # ── Step 5: Decode generated tokens back to text ───────────────────────
    # Combine prompt tokens (without BOS) and generated tokens
    all_tokens = prompt_tokens[1:] + generated_tokens  # Skip BOS for clean output
    generated_text = tokenizer.decode(all_tokens)

    # ── Collect metrics ────────────────────────────────────────────────────
    prefill_ms = (t_prefill - t_start) * 1000
    decode_ms = (t_end - t_prefill) * 1000
    total_ms = (t_end - t_start) * 1000
    peak_memory_mb = 0.0
    if use_cuda:
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    return GenerateResult(
        text=generated_text,
        prompt_tokens=prompt_len,
        generated_tokens=len(generated_tokens),
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        total_ms=total_ms,
        peak_memory_mb=peak_memory_mb,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """
    Sample a single token from logits using temperature + top-k + top-p.

    SAMPLING PIPELINE (applied in order):

    1. TEMPERATURE SCALING:
       logits = logits / temperature

       Temperature controls the "sharpness" of the probability distribution:
         - temperature = 1.0: Original distribution (default)
         - temperature < 1.0: Sharper peaks → more deterministic
           Example: [0.4, 0.3, 0.2, 0.1] → [0.52, 0.28, 0.14, 0.06]
         - temperature > 1.0: Flatter distribution → more random
           Example: [0.4, 0.3, 0.2, 0.1] → [0.30, 0.27, 0.23, 0.20]
         - temperature → 0: Approaches greedy decoding (argmax)

       WHY: Allows trading off between quality/coherence (low temp) and
       diversity/creativity (high temp).

    2. TOP-K FILTERING:
       Keep only the top K highest logits, set rest to -infinity.

       This prevents the model from sampling tokens that have very low
       probability — the "long tail" of unlikely tokens. Without top-k,
       there's always a small chance of sampling a completely nonsensical token.

       WHY: Removes the long tail of improbable tokens. Even if the model
       assigns 0.001% probability to "xyzzy", without top-k it could be sampled.

    3. TOP-P (NUCLEUS) FILTERING:
       Keep the smallest set of tokens whose cumulative probability ≥ p.

       WHY: More adaptive than top-k. When the model is confident, fewer
       tokens are kept. When uncertain, more tokens are available.

    4. MULTINOMIAL SAMPLING:
       Convert filtered logits to probabilities via softmax, then sample
       one token according to those probabilities.

    Args:
        logits: Raw prediction scores, shape (1, vocab_size).
        temperature: Sampling temperature (0 = greedy).
        top_k: Number of top tokens to keep (0 = disabled).
        top_p: Cumulative probability threshold (1.0 = disabled).

    Returns:
        Sampled token ID as a scalar tensor.
    """
    logits = logits.squeeze(0)  # (vocab_size,)

    # Special case: greedy decoding
    if temperature == 0.0:
        return logits.argmax()

    # Step 1: Temperature scaling
    logits = logits / temperature

    # Step 2: Top-k filtering
    if top_k > 0:
        # Find the k-th largest logit value
        top_k = min(top_k, logits.size(-1))  # Can't top-k more than vocab
        kth_value = torch.topk(logits, top_k).values[-1]
        # Set all logits below the k-th largest to -infinity
        # After softmax, these will become 0 probability
        logits[logits < kth_value] = float("-inf")

    # Step 3: Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Step 4: Top-p filtering and sampling
    if top_p < 1.0:
        return sample_top_p(probs, top_p)
    else:
        # No top-p filtering, just sample from the distribution
        return torch.multinomial(probs, num_samples=1).squeeze(0)


@torch.inference_mode()
def generate_batch(
    model: LLaMA,
    tokenizer: Tokenizer,
    prompts: list[str],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> list[str]:
    """
    Generate text for multiple prompts.

    For simplicity, this generates each prompt sequentially (no batched
    generation with padding). For our tiny model, the overhead is negligible.

    For production systems, you'd want to:
      - Pad prompts to the same length
      - Batch the prefill phase
      - Handle different completion lengths per prompt

    Args:
        model: Trained LLaMA model.
        tokenizer: Tokenizer.
        prompts: List of input prompts.
        max_new_tokens: Maximum tokens to generate per prompt.
        temperature, top_k, top_p: Sampling parameters.
        device: Compute device.

    Returns:
        List of GenerateResult (one per prompt).
    """
    results = []
    for prompt in prompts:
        result = generate(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )
        results.append(result)
    return results
