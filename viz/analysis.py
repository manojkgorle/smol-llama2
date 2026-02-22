"""Post-hoc analysis functions for LLaMA mechanistic interpretability.

Loads a checkpoint, runs inference with hooks, returns JSON-serializable dicts.
Adapted for LLaMA-2 architecture: GQA, SwiGLU, RoPE, RMSNorm.
"""

import torch
import numpy as np

from llama_vc.config import ModelConfig
from llama_vc.model import LLaMA, apply_rotary_embeddings
from llama_vc.tokenizer import Tokenizer
from viz.hooks import HookManager

_tokenizer: Tokenizer | None = None
_tokenizer_path: str = "data/tokenizer.model"


def init_tokenizer(path: str) -> None:
    """Set the tokenizer model path. Call before any analysis functions."""
    global _tokenizer_path, _tokenizer
    _tokenizer_path = path
    _tokenizer = None  # reset so next call loads from new path


def _get_tokenizer() -> Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer(_tokenizer_path)
    return _tokenizer


def load_model_from_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> tuple[LLaMA, ModelConfig]:
    """Load model and config from a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = LLaMA(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config


def tokenize_prompt(prompt: str) -> tuple[list[int], list[str]]:
    """Tokenize prompt, return (token_ids, human-readable token strings)."""
    tok = _get_tokenizer()
    token_ids = tok.encode(prompt, bos=True, eos=False)
    token_strings = [tok.id_to_piece(tid) for tid in token_ids]
    return token_ids, token_strings


def get_attention_weights(model: LLaMA, token_ids: list[int], device: str = "cpu") -> dict:
    """Run forward pass with hooks, return attention weights + entropy + GQA grouping.

    Returns dict with: tokens, n_heads, n_kv_heads, n_kv_groups, layers, entropy,
    entropy_summary, kv_groups.
    """
    tok = _get_tokenizer()
    token_strings = [tok.id_to_piece(tid) for tid in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            model(input_ids)
        data = mgr.collect()
    finally:
        mgr.detach()

    n_heads = model.config.n_heads
    n_kv_heads = model.config.n_kv_heads
    n_kv_groups = model.config.n_kv_groups

    layers = {}
    entropy = {}
    entropy_summary = {}

    for layer_idx, attn_matrix in data.attention_weights.items():
        # attn_matrix: (1, n_heads, T, T)
        heads = {}
        layer_entropy = {}
        layer_entropy_summary = {}

        for head_idx in range(attn_matrix.shape[1]):
            weights = attn_matrix[0, head_idx]  # (T, T)
            heads[str(head_idx)] = weights.tolist()
            # Shannon entropy per query position
            log_w = torch.log(weights.clamp(min=1e-10))
            head_entropy = -(weights * log_w).sum(dim=-1)  # (T,)
            layer_entropy[str(head_idx)] = head_entropy.tolist()
            layer_entropy_summary[str(head_idx)] = float(head_entropy.mean())

        layers[str(layer_idx)] = heads
        entropy[str(layer_idx)] = layer_entropy
        entropy_summary[str(layer_idx)] = layer_entropy_summary

    # GQA grouping: which query heads share which KV heads
    kv_groups = {}
    for kv_idx in range(n_kv_heads):
        start_q = kv_idx * n_kv_groups
        end_q = start_q + n_kv_groups
        kv_groups[str(kv_idx)] = list(range(start_q, end_q))

    return {
        "tokens": token_strings,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "n_kv_groups": n_kv_groups,
        "layers": layers,
        "entropy": entropy,
        "entropy_summary": entropy_summary,
        "kv_groups": kv_groups,
    }


def get_activation_analysis(model: LLaMA, token_ids: list[int], device: str = "cpu") -> dict:
    """Compute activation statistics including SwiGLU internals and RMSNorm stats."""
    tok = _get_tokenizer()
    token_strings = [tok.id_to_piece(tid) for tid in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            model(input_ids)
        data = mgr.collect()
    finally:
        mgr.detach()

    layers = {}
    for i in range(model.config.n_layers):
        layer_data = {}

        # Residual stream norms per position
        if i in data.residual_states:
            residual = data.residual_states[i][0]  # (T, dim)
            layer_data["residual_norms"] = residual.norm(dim=-1).tolist()

        # Attention output norms per position
        if i in data.attn_outputs:
            attn_out = data.attn_outputs[i][0]  # (T, dim)
            layer_data["attn_output_norms"] = attn_out.norm(dim=-1).tolist()

        # FFN output norms per position
        if i in data.ffn_outputs:
            ffn_out = data.ffn_outputs[i][0]  # (T, dim)
            layer_data["ffn_output_norms"] = ffn_out.norm(dim=-1).tolist()

        # SwiGLU internals
        if i in data.swiglu_internals:
            swiglu = data.swiglu_internals[i]
            swiglu_stats = {}

            for key in ["gate", "up", "gated"]:
                if key not in swiglu:
                    continue
                tensor = swiglu[key][0]  # (T, hidden_dim)
                flat = tensor.flatten()

                counts, bin_edges = np.histogram(flat.numpy(), bins=100)
                bins_center = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()

                stats = {
                    "mean": float(flat.mean()),
                    "std": float(flat.std()),
                    "histogram": {"bins": bins_center, "counts": counts.tolist()},
                }

                if key == "gate":
                    stats["sparsity"] = float((flat.abs() < 0.01).float().mean())
                    stats["amplification"] = float((flat > 1.0).float().mean())
                    # Top gate neurons by mean activation
                    mean_act = tensor.mean(dim=0)  # (hidden_dim,)
                    top_vals, top_idxs = mean_act.abs().topk(min(20, mean_act.shape[0]))
                    stats["top_neurons"] = [
                        {"idx": int(idx), "mean_activation": float(mean_act[idx])}
                        for idx in top_idxs.tolist()
                    ]

                swiglu_stats[key] = stats

            layer_data["swiglu_stats"] = swiglu_stats

        layers[str(i)] = layer_data

    return {"tokens": token_strings, "layers": layers}


def get_logit_attribution(
    model: LLaMA, token_ids: list[int], device: str = "cpu", top_k: int = 10
) -> dict:
    """Compute per-layer logit attribution using the logit lens technique."""
    tok = _get_tokenizer()
    token_strings = [tok.id_to_piece(tid) for tid in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            logits, _ = model(input_ids)
        data = mgr.collect()
    finally:
        mgr.detach()

    # Build cumulative residual stream: [embedding, after_layer_0, ..., after_layer_7]
    cumulative = [data.embedding_output]
    for i in range(model.config.n_layers):
        cumulative.append(data.residual_states[i])

    norm = model.norm      # Final RMSNorm
    output = model.output  # LM head

    positions = []
    T = len(token_ids)

    for pos in range(T - 1):
        target_id = token_ids[pos + 1]
        target_str = tok.id_to_piece(target_id)

        layer_contributions = {}
        cumulative_predictions = {}
        prev_target_logit = None

        for depth_idx, label in enumerate(
            ["embedding"] + [str(i) for i in range(model.config.n_layers)]
        ):
            hidden = cumulative[depth_idx][:, pos, :].to(device)  # (1, dim)
            with torch.no_grad():
                depth_logits = output(norm(hidden))  # (1, vocab_size)

            target_logit = depth_logits[0, target_id].item()

            if prev_target_logit is None:
                layer_contributions[label] = target_logit
            else:
                layer_contributions[label] = target_logit - prev_target_logit
            prev_target_logit = target_logit

            probs = torch.softmax(depth_logits[0], dim=-1)
            top_probs, top_ids = probs.topk(top_k)
            cumulative_predictions[label] = [
                {"token": tok.id_to_piece(int(tid)), "prob": float(p)}
                for tid, p in zip(top_ids.tolist(), top_probs.tolist())
            ]

        final_probs = torch.softmax(logits[0, pos], dim=-1)
        final_prob = final_probs[target_id].item()

        positions.append({
            "position": pos,
            "token": token_strings[pos],
            "target": target_str,
            "layer_contributions": layer_contributions,
            "cumulative_predictions": cumulative_predictions,
            "final_prob": final_prob,
        })

    # Next token predictions
    last_probs = torch.softmax(logits[0, -1], dim=-1)
    top_probs, top_ids = last_probs.topk(top_k)
    next_token_predictions = [
        {"token": tok.id_to_piece(int(tid)), "prob": float(p)}
        for tid, p in zip(top_ids.tolist(), top_probs.tolist())
    ]

    return {
        "tokens": token_strings,
        "positions": positions,
        "next_token_predictions": next_token_predictions,
    }


def get_head_ablation(model: LLaMA, token_ids: list[int], device: str = "cpu") -> dict:
    """Zero-ablate each attention head and measure loss change. Returns 8x6 importance matrix."""
    tok = _get_tokenizer()
    token_strings = [tok.id_to_piece(tid) for tid in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    targets = torch.tensor([token_ids[1:] + [token_ids[-1]]], dtype=torch.long, device=device)

    n_layers = model.config.n_layers
    n_heads = model.config.n_heads
    head_dim = model.config.head_dim

    # Baseline loss
    with torch.no_grad():
        _, baseline_loss = model(input_ids, targets[:, :input_ids.shape[1]])
    baseline = baseline_loss.item()

    importance = []
    max_delta = {"layer": 0, "head": 0, "delta": 0.0}

    for layer_idx in range(n_layers):
        layer_deltas = []
        for head_idx in range(n_heads):
            def make_pre_hook(h_idx):
                def pre_hook(module, args):
                    x = args[0].clone()
                    x[:, :, h_idx * head_dim:(h_idx + 1) * head_dim] = 0
                    return (x,) + args[1:]
                return pre_hook

            handle = model.layers[layer_idx].attention.wo.register_forward_pre_hook(
                make_pre_hook(head_idx)
            )
            with torch.no_grad():
                _, ablated_loss = model(input_ids, targets[:, :input_ids.shape[1]])
            handle.remove()

            delta = ablated_loss.item() - baseline
            layer_deltas.append(round(delta, 4))
            if abs(delta) > abs(max_delta["delta"]):
                max_delta = {"layer": layer_idx, "head": head_idx, "delta": round(delta, 4)}

        importance.append(layer_deltas)

    return {
        "tokens": token_strings,
        "baseline_loss": round(baseline, 4),
        "importance": importance,
        "max_importance": max_delta,
    }


def get_direct_logit_attribution(
    model: LLaMA, token_ids: list[int], device: str = "cpu", position: int = -1
) -> dict:
    """Per-head and per-FFN direct logit attribution for LLaMA (8x6 heads + 8 FFN)."""
    tok = _get_tokenizer()
    token_strings = [tok.id_to_piece(tid) for tid in token_ids]
    T = len(token_ids)
    if position < 0:
        position = T + position
    if position >= T - 1:
        position = T - 2

    target_id = token_ids[position + 1]
    target_str = tok.id_to_piece(target_id)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            logits, _ = model(input_ids)
        data = mgr.collect()
    finally:
        mgr.detach()

    n_layers = model.config.n_layers
    n_heads = model.config.n_heads
    head_dim = model.config.head_dim

    # Unembed direction for target token
    unembed_dir = model.output.weight[target_id].detach().to(device)  # (dim,)

    # Embedding contribution
    emb = data.embedding_output[0, position, :].to(device)
    emb_logit = float((model.norm(emb.unsqueeze(0)) @ unembed_dir).item())

    # Per-head contributions
    head_contributions = []
    for layer_idx in range(n_layers):
        layer_heads = []
        pre_proj = data.attn_pre_proj[layer_idx][0, position, :]  # (n_heads*head_dim,)
        wo_weight = model.layers[layer_idx].attention.wo.weight  # (dim, n_heads*head_dim)

        for head_idx in range(n_heads):
            head_slice = pre_proj[head_idx * head_dim:(head_idx + 1) * head_dim].to(device)
            weight_slice = wo_weight[:, head_idx * head_dim:(head_idx + 1) * head_dim]
            head_out = head_slice @ weight_slice.T  # (dim,)
            contribution = float((head_out @ unembed_dir).item())
            layer_heads.append(round(contribution, 4))

        head_contributions.append(layer_heads)

    # Per-FFN contributions
    ffn_contributions = []
    for layer_idx in range(n_layers):
        ffn_out = data.ffn_outputs[layer_idx][0, position, :].to(device)
        contribution = float((ffn_out @ unembed_dir).item())
        ffn_contributions.append(round(contribution, 4))

    final_probs = torch.softmax(logits[0, position], dim=-1)
    final_prob = final_probs[target_id].item()

    total = emb_logit + sum(sum(row) for row in head_contributions) + sum(ffn_contributions)

    return {
        "tokens": token_strings,
        "position": position,
        "target": target_str,
        "target_id": target_id,
        "final_prob": round(final_prob, 4),
        "embedding_contribution": round(emb_logit, 4),
        "head_contributions": head_contributions,
        "ffn_contributions": ffn_contributions,
        "total_reconstructed": round(total, 4),
    }


def get_rope_analysis(model: LLaMA, token_ids: list[int], device: str = "cpu") -> dict:
    """Compare attention patterns with and without RoPE. Shows distance decay."""
    tok = _get_tokenizer()
    token_strings = [tok.id_to_piece(tid) for tid in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            model(input_ids)
        data = mgr.collect()
    finally:
        mgr.detach()

    n_heads = model.config.n_heads
    n_kv_heads = model.config.n_kv_heads
    n_kv_groups = model.config.n_kv_groups
    head_dim = model.config.head_dim

    layers = {}
    for layer_idx in data.pre_rope_qk:
        q_pre, k_pre = data.pre_rope_qk[layer_idx]  # (B, T, n_heads, d), (B, T, n_kv_heads, d)
        B, T = q_pre.shape[0], q_pre.shape[1]

        # Compute pre-RoPE attention
        q_t = q_pre.transpose(1, 2)  # (B, n_heads, T, d)
        k_t = k_pre.transpose(1, 2)  # (B, n_kv_heads, T, d)
        if n_kv_heads != n_heads:
            k_t = k_t.repeat_interleave(n_kv_groups, dim=1)

        scale = head_dim ** -0.5
        pre_attn = (q_t @ k_t.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        pre_attn.masked_fill_(causal_mask, float("-inf"))
        pre_attn = torch.softmax(pre_attn, dim=-1)

        # Post-RoPE attention is already in data.attention_weights
        post_attn = data.attention_weights[layer_idx]  # (B, n_heads, T, T)

        # Distance decay: average attention weight as function of distance
        distance_decay = []
        for d in range(T):
            # Gather all attention weights where key_pos = query_pos - d
            weights = []
            for q in range(d, T):
                k = q - d
                for h in range(n_heads):
                    weights.append(post_attn[0, h, q, k].item())
            distance_decay.append(float(np.mean(weights)) if weights else 0.0)

        # Pre-RoPE distance decay for comparison
        pre_distance_decay = []
        for d in range(T):
            weights = []
            for q in range(d, T):
                k = q - d
                for h in range(n_heads):
                    weights.append(pre_attn[0, h, q, k].item())
            pre_distance_decay.append(float(np.mean(weights)) if weights else 0.0)

        layers[str(layer_idx)] = {
            "post_rope_distance_decay": distance_decay,
            "pre_rope_distance_decay": pre_distance_decay,
        }

    return {
        "tokens": token_strings,
        "layers": layers,
    }


# =========================================================================
# Circuits: Causal intervention experiments
# =========================================================================


def get_activation_patching(
    model: LLaMA, clean_ids: list[int], corrupted_ids: list[int], device: str = "cpu"
) -> dict:
    """Activation patching: patch clean activations into corrupted forward pass."""
    tok = _get_tokenizer()
    tokens_clean = [tok.id_to_piece(tid) for tid in clean_ids]
    tokens_corrupted = [tok.id_to_piece(tid) for tid in corrupted_ids]

    clean_input = torch.tensor([clean_ids], dtype=torch.long, device=device)
    corrupted_input = torch.tensor([corrupted_ids], dtype=torch.long, device=device)
    n_layers = model.config.n_layers

    # 1. Clean forward pass with hooks
    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            clean_logits, _ = model(clean_input)
        clean_data = mgr.collect()
    finally:
        mgr.detach()

    # Target: what the clean prompt predicts
    clean_last_probs = torch.softmax(clean_logits[0, -1], dim=-1)
    target_id = int(clean_last_probs.argmax().item())
    target_str = tok.id_to_piece(target_id)
    clean_logit = clean_logits[0, -1, target_id].item()

    # 2. Corrupted baseline
    with torch.no_grad():
        corrupted_logits, _ = model(corrupted_input)
    corrupted_logit = corrupted_logits[0, -1, target_id].item()

    logit_range = clean_logit - corrupted_logit
    if abs(logit_range) < 1e-8:
        logit_range = 1.0

    min_T = min(len(clean_ids), len(corrupted_ids))

    # 3. Patch each (layer, component)
    patching_results = []
    max_recovery = {"layer": 0, "component": "residual", "recovery": 0.0}

    for layer_idx in range(n_layers):
        layer_result = {"layer": layer_idx}

        for comp_name, comp_key, get_module in [
            ("residual", "residual_states", lambda l: model.layers[l]),
            ("attn", "attn_outputs", lambda l: model.layers[l].attention),
            ("ffn", "ffn_outputs", lambda l: model.layers[l].feed_forward),
        ]:
            clean_act = getattr(clean_data, comp_key)[layer_idx].to(device)

            def make_hook(clean_tensor, min_t):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        patched = output[0].clone()
                        patched[:, :min_t, :] = clean_tensor[:, :min_t, :]
                        return (patched,) + output[1:]
                    patched = output.clone()
                    patched[:, :min_t, :] = clean_tensor[:, :min_t, :]
                    return patched
                return hook_fn

            handle = get_module(layer_idx).register_forward_hook(
                make_hook(clean_act, min_T)
            )
            with torch.no_grad():
                patched_logits, _ = model(corrupted_input)
            handle.remove()

            patched_val = patched_logits[0, -1, target_id].item()
            recovery = (patched_val - corrupted_logit) / logit_range
            layer_result[comp_name] = round(recovery, 4)

            if abs(recovery) > abs(max_recovery["recovery"]):
                max_recovery = {
                    "layer": layer_idx, "component": comp_name,
                    "recovery": round(recovery, 4),
                }

        patching_results.append(layer_result)

    return {
        "tokens_clean": tokens_clean,
        "tokens_corrupted": tokens_corrupted,
        "clean_logit": round(clean_logit, 4),
        "corrupted_logit": round(corrupted_logit, 4),
        "logit_gap": round(clean_logit - corrupted_logit, 4),
        "target_token": target_str,
        "patching_results": patching_results,
        "max_recovery": max_recovery,
    }


def get_activation_steering(
    model: LLaMA, token_ids: list[int], device: str = "cpu",
    layer: int = 0, component: str = "head", head: int = 0, scale: float = 0.0,
    top_k: int = 10,
) -> dict:
    """Scale a specific component and measure the output distribution change."""
    tok = _get_tokenizer()
    token_strings = [tok.id_to_piece(tid) for tid in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    head_dim = model.config.head_dim

    # Baseline
    with torch.no_grad():
        baseline_logits, _ = model(input_ids)
    baseline_probs = torch.softmax(baseline_logits[0, -1], dim=-1)
    top_bp, top_bi = baseline_probs.topk(top_k)
    baseline_predictions = [
        {"token": tok.id_to_piece(int(tid)), "prob": float(p)}
        for tid, p in zip(top_bi.tolist(), top_bp.tolist())
    ]

    # Steered forward pass
    if component == "head":
        def make_pre_hook(h_idx, s):
            def pre_hook(module, args):
                x = args[0].clone()
                x[:, :, h_idx * head_dim:(h_idx + 1) * head_dim] *= s
                return (x,) + args[1:]
            return pre_hook
        handle = model.layers[layer].attention.wo.register_forward_pre_hook(
            make_pre_hook(head, scale)
        )
    else:  # ffn
        def make_hook(s):
            def hook_fn(module, input, output):
                return output * s
            return hook_fn
        handle = model.layers[layer].feed_forward.register_forward_hook(
            make_hook(scale)
        )

    with torch.no_grad():
        steered_logits, _ = model(input_ids)
    handle.remove()

    steered_probs = torch.softmax(steered_logits[0, -1], dim=-1)
    top_sp, top_si = steered_probs.topk(top_k)
    steered_predictions = [
        {"token": tok.id_to_piece(int(tid)), "prob": float(p)}
        for tid, p in zip(top_si.tolist(), top_sp.tolist())
    ]

    kl = torch.sum(
        steered_probs * (
            torch.log(steered_probs.clamp(min=1e-10))
            - torch.log(baseline_probs.clamp(min=1e-10))
        )
    ).item()

    return {
        "tokens": token_strings,
        "layer": layer,
        "component": component,
        "head": head if component == "head" else None,
        "scale": scale,
        "baseline_predictions": baseline_predictions,
        "steered_predictions": steered_predictions,
        "kl_divergence": round(kl, 6),
    }


def get_activation_swapping(
    model: LLaMA, source_ids: list[int], target_ids: list[int], device: str = "cpu",
    layer: int = 0, component: str = "residual", top_k: int = 10,
) -> dict:
    """Swap activations from source prompt into target prompt's forward pass."""
    tok = _get_tokenizer()
    source_strings = [tok.id_to_piece(tid) for tid in source_ids]
    target_strings = [tok.id_to_piece(tid) for tid in target_ids]

    source_input = torch.tensor([source_ids], dtype=torch.long, device=device)
    target_input = torch.tensor([target_ids], dtype=torch.long, device=device)
    min_T = min(len(source_ids), len(target_ids))

    # 1. Source forward pass with hooks
    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            model(source_input)
        source_data = mgr.collect()
    finally:
        mgr.detach()

    # 2. Target baseline
    with torch.no_grad():
        baseline_logits, _ = model(target_input)
    baseline_probs = torch.softmax(baseline_logits[0, -1], dim=-1)
    top_bp, top_bi = baseline_probs.topk(top_k)
    baseline_predictions = [
        {"token": tok.id_to_piece(int(tid)), "prob": float(p)}
        for tid, p in zip(top_bi.tolist(), top_bp.tolist())
    ]

    # 3. Target with swapped activations
    comp_map = {
        "residual": ("residual_states", lambda l: model.layers[l]),
        "attn": ("attn_outputs", lambda l: model.layers[l].attention),
        "ffn": ("ffn_outputs", lambda l: model.layers[l].feed_forward),
    }
    data_key, get_module = comp_map[component]
    source_act = getattr(source_data, data_key)[layer].to(device)

    def make_hook(source_tensor, min_t):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                swapped = output[0].clone()
                swapped[:, :min_t, :] = source_tensor[:, :min_t, :]
                return (swapped,) + output[1:]
            swapped = output.clone()
            swapped[:, :min_t, :] = source_tensor[:, :min_t, :]
            return swapped
        return hook_fn

    handle = get_module(layer).register_forward_hook(make_hook(source_act, min_T))
    with torch.no_grad():
        swapped_logits, _ = model(target_input)
    handle.remove()

    swapped_probs = torch.softmax(swapped_logits[0, -1], dim=-1)
    top_sp, top_si = swapped_probs.topk(top_k)
    swapped_predictions = [
        {"token": tok.id_to_piece(int(tid)), "prob": float(p)}
        for tid, p in zip(top_si.tolist(), top_sp.tolist())
    ]

    kl = torch.sum(
        swapped_probs * (
            torch.log(swapped_probs.clamp(min=1e-10))
            - torch.log(baseline_probs.clamp(min=1e-10))
        )
    ).item()

    return {
        "source_tokens": source_strings,
        "target_tokens": target_strings,
        "layer": layer,
        "component": component,
        "baseline_predictions": baseline_predictions,
        "swapped_predictions": swapped_predictions,
        "kl_divergence": round(kl, 6),
    }


def get_precomputation_detection(
    model: LLaMA, token_ids: list[int], device: str = "cpu", top_k: int = 5
) -> dict:
    """Detect pre-computation: where future tokens first appear in intermediate layers."""
    tok = _get_tokenizer()
    token_strings = [tok.id_to_piece(tid) for tid in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    T = len(token_ids)

    n_layers = model.config.n_layers
    future_offsets = [2, 3, 4, 5]

    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            model(input_ids)
        data = mgr.collect()
    finally:
        mgr.detach()

    # Build cumulative: [embedding, after_layer_0, ..., after_layer_7]
    cumulative = [data.embedding_output]
    for i in range(n_layers):
        cumulative.append(data.residual_states[i])

    norm = model.norm
    output = model.output
    depth_labels = ["embedding"] + [str(i) for i in range(n_layers)]

    precomputation_matrix = []
    findings = []

    for pos in range(T):
        pos_row = [None] * len(future_offsets)

        for depth_idx, label in enumerate(depth_labels):
            hidden = cumulative[depth_idx][:, pos, :].to(device)
            with torch.no_grad():
                depth_logits = output(norm(hidden))
            top_ids = depth_logits[0].topk(top_k).indices.tolist()

            for off_idx, offset in enumerate(future_offsets):
                future_pos = pos + offset
                if future_pos >= T:
                    continue
                future_token_id = token_ids[future_pos]

                if future_token_id in top_ids and pos_row[off_idx] is None:
                    pos_row[off_idx] = depth_idx
                    if depth_idx < len(depth_labels) - 1:
                        findings.append({
                            "position": pos,
                            "future_offset": offset,
                            "first_depth": label,
                            "first_depth_idx": depth_idx,
                            "token": token_strings[pos],
                            "future_token": token_strings[future_pos],
                        })

        precomputation_matrix.append(pos_row)

    findings.sort(key=lambda f: (f["first_depth_idx"], -f["future_offset"]))

    return {
        "tokens": token_strings,
        "precomputation_matrix": precomputation_matrix,
        "future_offsets": future_offsets,
        "findings": findings[:20],
    }
