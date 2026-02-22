"""Captum-based attribution methods for LLaMA interpretability.

Provides:
  - Integrated Gradients: token-level attribution via path integral
  - Layer Conductance: per-layer, per-token contribution scores
  - Gradient Saliency: simple gradient-based token importance (no Captum needed)
"""

import torch
import torch.nn as nn

from llama_vc.model import LLaMA
from viz.analysis import _get_tokenizer


class CaptumModelWrapper(nn.Module):
    """Wraps LLaMA so Captum can attribute over embedding space.

    Bypasses tok_embeddings, feeding embeddings directly into transformer layers.
    Returns a scalar (the logit for the target token at the target position).
    """

    def __init__(self, model: LLaMA, target_id: int, target_position: int = -1):
        super().__init__()
        self.model = model
        self.target_id = target_id
        self.target_position = target_position

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """input_embeds: (B, T, dim) -> scalar logit for target token."""
        h = input_embeds
        freqs_cos = self.model.freqs_cos[:h.shape[1]].to(h.device)
        freqs_sin = self.model.freqs_sin[:h.shape[1]].to(h.device)

        for layer in self.model.layers:
            h, _ = layer(h, freqs_cos, freqs_sin)

        h = self.model.norm(h)
        logits = self.model.output(h)  # (B, T, vocab_size)
        return logits[:, self.target_position, self.target_id]


def get_integrated_gradients(
    model: LLaMA, token_ids: list[int], device: str = "cpu",
    target_position: int = -1, n_steps: int = 50,
) -> dict:
    """Compute Integrated Gradients attribution over token embeddings.

    Returns per-token attribution scores (sum over embedding dim).
    """
    from captum.attr import IntegratedGradients

    tok = _get_tokenizer()
    token_strings = [tok.id_to_piece(tid) for tid in token_ids]
    T = len(token_ids)

    if target_position < 0:
        target_position = T + target_position
    if target_position >= T - 1:
        target_position = T - 2

    target_id = token_ids[target_position + 1]
    target_str = tok.id_to_piece(target_id)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Get input embeddings
    model.eval()
    with torch.no_grad():
        input_embeds = model.tok_embeddings(input_ids)  # (1, T, dim)

    # Create wrapper that takes embeddings and returns scalar
    wrapper = CaptumModelWrapper(model, target_id, target_position)
    wrapper.eval()

    ig = IntegratedGradients(wrapper)

    # Baseline: zero embeddings
    baseline = torch.zeros_like(input_embeds)

    # Need gradients for attribution
    input_embeds_attr = input_embeds.clone().requires_grad_(True)

    attributions, delta = ig.attribute(
        input_embeds_attr, baseline,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    # attributions shape: (1, T, dim)
    # Sum over embedding dimension for per-token scores
    per_token = attributions[0].sum(dim=-1).detach().cpu().tolist()  # (T,)

    return {
        "tokens": token_strings,
        "target_token": target_str,
        "target_position": target_position,
        "attributions": per_token,
        "convergence_delta": float(delta.item()) if delta.dim() == 0 else float(delta[0].item()),
    }


def get_layer_conductance(
    model: LLaMA, token_ids: list[int], device: str = "cpu",
    target_position: int = -1,
) -> dict:
    """Compute Layer Conductance for each transformer layer.

    Returns per-layer, per-token conductance scores.
    """
    from captum.attr import LayerConductance

    tok = _get_tokenizer()
    token_strings = [tok.id_to_piece(tid) for tid in token_ids]
    T = len(token_ids)

    if target_position < 0:
        target_position = T + target_position
    if target_position >= T - 1:
        target_position = T - 2

    target_id = token_ids[target_position + 1]
    target_str = tok.id_to_piece(target_id)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        input_embeds = model.tok_embeddings(input_ids)

    wrapper = CaptumModelWrapper(model, target_id, target_position)
    wrapper.eval()

    baseline = torch.zeros_like(input_embeds)
    layer_conductance = {}

    for i, layer in enumerate(model.layers):
        lc = LayerConductance(wrapper, layer)
        input_embeds_attr = input_embeds.clone().requires_grad_(True)
        conductance = lc.attribute(input_embeds_attr, baseline, n_steps=10)
        # conductance shape: (1, T, dim)
        per_token = conductance[0].sum(dim=-1).detach().cpu().tolist()
        layer_conductance[str(i)] = per_token

    return {
        "tokens": token_strings,
        "target_token": target_str,
        "target_position": target_position,
        "layer_conductance": layer_conductance,
    }


def get_token_saliency(
    model: LLaMA, token_ids: list[int], device: str = "cpu",
    target_position: int = -1,
) -> dict:
    """Simple gradient saliency: ||d(logit)/d(embedding)||_2 per token.

    No Captum dependency needed — pure PyTorch.
    """
    tok = _get_tokenizer()
    token_strings = [tok.id_to_piece(tid) for tid in token_ids]
    T = len(token_ids)

    if target_position < 0:
        target_position = T + target_position
    if target_position >= T - 1:
        target_position = T - 2

    target_id = token_ids[target_position + 1]
    target_str = tok.id_to_piece(target_id)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    model.eval()
    input_embeds = model.tok_embeddings(input_ids).detach().requires_grad_(True)

    # Forward pass through transformer layers
    h = input_embeds
    freqs_cos = model.freqs_cos[:T].to(device)
    freqs_sin = model.freqs_sin[:T].to(device)

    for layer in model.layers:
        h, _ = layer(h, freqs_cos, freqs_sin)

    h = model.norm(h)
    logits = model.output(h)

    # Backward from target logit
    target_logit = logits[0, target_position, target_id]
    target_logit.backward()

    # Gradient magnitude per token
    grad = input_embeds.grad[0]  # (T, dim)
    saliency = grad.norm(dim=-1).detach().cpu().tolist()  # (T,)

    return {
        "tokens": token_strings,
        "target_token": target_str,
        "target_position": target_position,
        "saliency": saliency,
    }
