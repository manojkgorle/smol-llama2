"""PyTorch forward hooks for capturing LLaMA model internals.

Two operating modes:
  - TRAINING: lightweight scalar stats only (norms, sparsity). Minimal overhead.
  - INFERENCE: full tensor capture for post-hoc analysis.

Key challenge: F.scaled_dot_product_attention doesn't expose attention weights.
Solution: Hook wq/wk outputs, manually compute softmax(QK^T/sqrt(d)) with RoPE.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Literal

from llama_vc.model import apply_rotary_embeddings


@dataclass
class HookData:
    """Full tensor captures from a single inference forward pass."""
    # GQA attention weights: {layer_idx: Tensor(B, n_heads, T, T)}
    attention_weights: dict[int, torch.Tensor] = field(default_factory=dict)
    # Pre-RoPE Q, K: {layer_idx: (Q, K)}
    pre_rope_qk: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    # Post-RoPE Q, K: {layer_idx: (Q, K)}
    post_rope_qk: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    # Residual stream after each block: {layer_idx: Tensor(B, T, dim)}
    residual_states: dict[int, torch.Tensor] = field(default_factory=dict)
    # Attention sub-layer output: {layer_idx: Tensor(B, T, dim)}
    attn_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    # FFN sub-layer output: {layer_idx: Tensor(B, T, dim)}
    ffn_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    # SwiGLU internals: {layer_idx: {"gate": Tensor, "up": Tensor, "gated": Tensor}}
    swiglu_internals: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    # Embedding output: Tensor(B, T, dim)
    embedding_output: torch.Tensor | None = None
    # Pre-Wo attention outputs: {layer_idx: Tensor(B, T, n_heads*head_dim)}
    attn_pre_proj: dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class TrainingSummary:
    """Lightweight per-step scalar stats for training mode."""
    residual_norms: list[float] = field(default_factory=list)
    attn_output_norms: list[float] = field(default_factory=list)
    ffn_output_norms: list[float] = field(default_factory=list)
    ffn_gate_sparsity: list[float] = field(default_factory=list)


class HookManager:
    """Manages PyTorch forward hooks on a LLaMA model.

    Usage (inference):
        mgr = HookManager(model, mode="inference")
        mgr.attach()
        with torch.no_grad():
            logits, _ = model(input_ids)
        data = mgr.collect()  # HookData
        mgr.detach()

    Usage (training):
        mgr = HookManager(model, mode="training")
        mgr.attach()
        # ... training step ...
        summary = mgr.collect()  # TrainingSummary
        mgr.clear()
        mgr.detach()
    """

    def __init__(self, model, mode: Literal["training", "inference"] = "inference"):
        self.model = model
        self.mode = mode
        self._handles: list[torch.utils.hooks.RemovableHook] = []

        # Extract model config
        self.n_layers = model.config.n_layers
        self.n_heads = model.config.n_heads
        self.n_kv_heads = model.config.n_kv_heads
        self.n_kv_groups = model.config.n_kv_groups
        self.head_dim = model.config.head_dim
        self.dim = model.config.dim

        # Storage
        self._hook_data = HookData()
        self._training_summary = TrainingSummary()

        # Temporary storage for Q/K captures (needed to compute attention weights)
        self._q_captures: dict[int, torch.Tensor] = {}
        self._k_captures: dict[int, torch.Tensor] = {}

    def attach(self) -> None:
        """Register forward hooks. Does not modify model.py."""
        self.detach()  # clean slate

        # Hook on tok_embeddings — captures embedding output
        self._handles.append(
            self.model.tok_embeddings.register_forward_hook(self._hook_embedding())
        )

        for i in range(self.n_layers):
            layer = self.model.layers[i]

            if self.mode == "inference":
                # Hook wq and wk to capture Q, K for manual attention computation
                self._handles.append(
                    layer.attention.wq.register_forward_hook(self._hook_q_capture(i))
                )
                self._handles.append(
                    layer.attention.wk.register_forward_hook(self._hook_k_capture(i))
                )
                # Hook wo input to capture pre-projection outputs for DLA
                self._handles.append(
                    layer.attention.wo.register_forward_hook(self._hook_attn_pre_proj(i))
                )
                # Hook SwiGLU gate and up projections
                self._handles.append(
                    layer.feed_forward.w_gate.register_forward_hook(self._hook_swiglu_gate(i))
                )
                self._handles.append(
                    layer.feed_forward.w_up.register_forward_hook(self._hook_swiglu_up(i))
                )

            # Hook attention sub-layer output (via the Attention module)
            self._handles.append(
                layer.attention.register_forward_hook(self._hook_attn_output(i))
            )
            # Hook FFN sub-layer output
            self._handles.append(
                layer.feed_forward.register_forward_hook(self._hook_ffn_output(i))
            )
            # Hook block output (full residual stream)
            self._handles.append(
                layer.register_forward_hook(self._hook_residual(i))
            )

    def detach(self) -> None:
        """Remove all hooks, restoring model to original state."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def collect(self) -> HookData | TrainingSummary:
        """Return captured data. Clears internal buffers."""
        if self.mode == "inference":
            # Compute attention weights from captured Q, K
            self._compute_all_attention_weights()
            data = self._hook_data
            self._hook_data = HookData()
            self._q_captures.clear()
            self._k_captures.clear()
            return data
        else:
            raw = self._training_summary
            self._training_summary = TrainingSummary()
            return TrainingSummary(
                residual_norms=self._avg_per_layer(raw.residual_norms),
                attn_output_norms=self._avg_per_layer(raw.attn_output_norms),
                ffn_output_norms=self._avg_per_layer(raw.ffn_output_norms),
                ffn_gate_sparsity=self._avg_per_layer(raw.ffn_gate_sparsity),
            )

    def _avg_per_layer(self, values: list[float]) -> list[float]:
        """Average a flat list of [layer0, ..., layerN, layer0, ...] into N means."""
        if not values:
            return []
        n = self.n_layers
        num_passes = len(values) // n
        if num_passes == 0:
            return values
        result = [0.0] * n
        for i, v in enumerate(values):
            result[i % n] += v
        return [x / num_passes for x in result]

    def clear(self) -> None:
        """Clear captured data without detaching hooks."""
        self._hook_data = HookData()
        self._training_summary = TrainingSummary()
        self._q_captures.clear()
        self._k_captures.clear()

    def _compute_all_attention_weights(self) -> None:
        """Compute attention weights from captured Q, K using RoPE + GQA expansion."""
        for layer_idx in self._q_captures:
            if layer_idx not in self._k_captures:
                continue

            q_proj = self._q_captures[layer_idx]  # (B, T, n_heads * head_dim)
            k_proj = self._k_captures[layer_idx]  # (B, T, n_kv_heads * head_dim)

            B, T, _ = q_proj.shape

            # Reshape into heads
            q = q_proj.view(B, T, self.n_heads, self.head_dim)      # (B, T, 6, 64)
            k = k_proj.view(B, T, self.n_kv_heads, self.head_dim)   # (B, T, 2, 64)

            # Store pre-RoPE Q, K
            self._hook_data.pre_rope_qk[layer_idx] = (q.clone(), k.clone())

            # Get RoPE frequencies from the model
            freqs_cos = self.model.freqs_cos[:T].to(q.device)
            freqs_sin = self.model.freqs_sin[:T].to(q.device)

            # Apply RoPE
            q_rot = apply_rotary_embeddings(q, freqs_cos, freqs_sin)
            k_rot = apply_rotary_embeddings(k, freqs_cos, freqs_sin)

            # Store post-RoPE Q, K
            self._hook_data.post_rope_qk[layer_idx] = (
                q_rot.detach().cpu(), k_rot.detach().cpu()
            )

            # Transpose for attention: (B, n_heads, T, head_dim)
            q_t = q_rot.transpose(1, 2)
            k_t = k_rot.transpose(1, 2)

            # GQA expansion: (B, n_kv_heads, T, d) -> (B, n_heads, T, d)
            if self.n_kv_heads != self.n_heads:
                k_t = k_t.repeat_interleave(self.n_kv_groups, dim=1)

            # Compute scaled attention scores
            scale = self.head_dim ** -0.5
            attn_weights = (q_t @ k_t.transpose(-2, -1)) * scale  # (B, n_heads, T, T)

            # Causal mask
            causal_mask = torch.triu(
                torch.ones(T, T, device=attn_weights.device, dtype=torch.bool), diagonal=1
            )
            attn_weights.masked_fill_(causal_mask, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)

            self._hook_data.attention_weights[layer_idx] = attn_weights.detach().cpu()

    # --- Hook factories ---

    def _hook_embedding(self):
        def hook_fn(module, input, output):
            if self.mode == "inference":
                self._hook_data.embedding_output = output.detach().cpu()
        return hook_fn

    def _hook_q_capture(self, layer_idx: int):
        """Inference only: capture Q projection output."""
        def hook_fn(module, input, output):
            self._q_captures[layer_idx] = output.detach()
        return hook_fn

    def _hook_k_capture(self, layer_idx: int):
        """Inference only: capture K projection output."""
        def hook_fn(module, input, output):
            self._k_captures[layer_idx] = output.detach()
        return hook_fn

    def _hook_attn_pre_proj(self, layer_idx: int):
        """Inference only: capture wo input (concatenated head outputs before projection)."""
        def hook_fn(module, input, output):
            self._hook_data.attn_pre_proj[layer_idx] = input[0].detach().cpu()
        return hook_fn

    def _hook_swiglu_gate(self, layer_idx: int):
        """Inference only: capture gate projection output (before SiLU)."""
        def hook_fn(module, input, output):
            # output is w_gate(x), we need F.silu(output) for the actual gate signal
            gate = F.silu(output.detach())
            if layer_idx not in self._hook_data.swiglu_internals:
                self._hook_data.swiglu_internals[layer_idx] = {}
            self._hook_data.swiglu_internals[layer_idx]["gate"] = gate.cpu()
        return hook_fn

    def _hook_swiglu_up(self, layer_idx: int):
        """Inference only: capture up projection output."""
        def hook_fn(module, input, output):
            up = output.detach()
            if layer_idx not in self._hook_data.swiglu_internals:
                self._hook_data.swiglu_internals[layer_idx] = {}
            self._hook_data.swiglu_internals[layer_idx]["up"] = up.cpu()
            # Compute gated product if gate is already captured
            if "gate" in self._hook_data.swiglu_internals[layer_idx]:
                gate = self._hook_data.swiglu_internals[layer_idx]["gate"]
                self._hook_data.swiglu_internals[layer_idx]["gated"] = (gate * up.cpu())
        return hook_fn

    def _hook_attn_output(self, layer_idx: int):
        """Capture attention sub-layer output (the tuple return from Attention.forward)."""
        def hook_fn(module, input, output):
            # Attention.forward returns (output_tensor, kv_cache)
            attn_out = output[0] if isinstance(output, tuple) else output
            if self.mode == "inference":
                self._hook_data.attn_outputs[layer_idx] = attn_out.detach().cpu()
            else:
                norm_val = attn_out.detach().norm(dim=-1).mean().item()
                self._training_summary.attn_output_norms.append(norm_val)
        return hook_fn

    def _hook_ffn_output(self, layer_idx: int):
        """Capture FFN sub-layer output."""
        def hook_fn(module, input, output):
            if self.mode == "inference":
                self._hook_data.ffn_outputs[layer_idx] = output.detach().cpu()
            else:
                norm_val = output.detach().norm(dim=-1).mean().item()
                self._training_summary.ffn_output_norms.append(norm_val)
                # Gate sparsity: fraction of gate values near zero
                # We capture gate output during training too for sparsity
                gate_raw = self.model.layers[layer_idx].feed_forward.w_gate(input[0])
                gate = F.silu(gate_raw.detach())
                sparsity = (gate.abs() < 0.01).float().mean().item()
                self._training_summary.ffn_gate_sparsity.append(sparsity)
        return hook_fn

    def _hook_residual(self, layer_idx: int):
        """Capture full residual stream after block."""
        def hook_fn(module, input, output):
            # TransformerBlock.forward returns (output_tensor, kv_cache)
            block_out = output[0] if isinstance(output, tuple) else output
            if self.mode == "inference":
                self._hook_data.residual_states[layer_idx] = block_out.detach().cpu()
            else:
                norm_val = block_out.detach().norm(dim=-1).mean().item()
                self._training_summary.residual_norms.append(norm_val)
        return hook_fn
