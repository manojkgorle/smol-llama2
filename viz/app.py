"""Flask application for LLaMA mechanistic interpretability visualization.

Serves:
  1. Static files (HTML/JS/CSS)
  2. REST API for post-hoc analysis (attention, activations, attribution, circuits, captum)
  3. WebSocket (flask-socketio) for real-time training updates

Usage:
    Standalone:  python scripts/run_viz.py --checkpoint checkpoints/best.pt
    During training: import init_app and launch as background thread
"""

import os
import glob as glob_module

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit

from viz.analysis import (
    init_tokenizer,
    load_model_from_checkpoint,
    tokenize_prompt,
    get_attention_weights,
    get_activation_analysis,
    get_logit_attribution,
    get_head_ablation,
    get_direct_logit_attribution,
    get_rope_analysis,
    get_activation_patching,
    get_activation_steering,
    get_activation_swapping,
    get_precomputation_detection,
    get_embedding_space,
    get_token_waterfall,
    get_prompt_comparison,
    get_neuron_overview,
    get_neuron_detail,
    get_neuron_network,
    get_attention_flow,
)
from viz.captum_analysis import (
    get_integrated_gradients,
    get_layer_conductance,
    get_token_saliency,
)

# Module-level state
_model = None
_config = None
_metrics_logger = None
_device = "cpu"

static_dir = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, static_folder=static_dir, static_url_path="/static")
app.config["SECRET_KEY"] = "llama-viz"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


def init_app(checkpoint_path=None, tokenizer_path="data/tokenizer.model", metrics_logger=None, device="cpu"):
    """Initialize the app with a model checkpoint and/or metrics logger."""
    global _model, _config, _metrics_logger, _device
    _device = device
    _metrics_logger = metrics_logger

    init_tokenizer(tokenizer_path)
    print(f"[viz] Tokenizer: {tokenizer_path}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        _model, _config = load_model_from_checkpoint(checkpoint_path, device)
        print(f"[viz] Model loaded from {checkpoint_path}")

    return app


def _require_model():
    if _model is None:
        return jsonify({"error": "No model loaded"}), 400
    return None


def _get_token_ids(prompt):
    token_ids, _ = tokenize_prompt(prompt)
    if _config and len(token_ids) > _config.max_seq_len:
        token_ids = token_ids[:_config.max_seq_len]
    return token_ids


# --- Static file serving ---

@app.route("/")
def index():
    return send_from_directory(static_dir, "index.html")


# --- Model info ---

@app.route("/api/model/info")
def model_info():
    if _config is None:
        return jsonify({"loaded": False})
    return jsonify({
        "loaded": True,
        "n_layers": _config.n_layers,
        "n_heads": _config.n_heads,
        "n_kv_heads": _config.n_kv_heads,
        "dim": _config.dim,
        "hidden_dim": _config.hidden_dim,
        "head_dim": _config.head_dim,
        "max_seq_len": _config.max_seq_len,
        "vocab_size": _config.vocab_size,
    })


# --- Checkpoint management ---

@app.route("/api/checkpoints")
def list_checkpoints():
    ckpt_dir = "checkpoints"
    if not os.path.isdir(ckpt_dir):
        return jsonify([])
    files = glob_module.glob(os.path.join(ckpt_dir, "*.pt"))
    return jsonify([os.path.basename(f) for f in sorted(files)])


@app.route("/api/load_checkpoint", methods=["POST"])
def load_checkpoint():
    global _model, _config
    data = request.get_json()
    path = data.get("path", "")
    if not os.path.exists(path):
        return jsonify({"error": f"Checkpoint not found: {path}"}), 404

    _model, _config = load_model_from_checkpoint(path, _device)
    return jsonify({"status": "ok", "n_layers": _config.n_layers})


# --- Training metrics ---

@app.route("/api/metrics/all")
def metrics_all():
    if _metrics_logger is None:
        return jsonify({"steps": [], "validations": []})
    return jsonify({
        "steps": _metrics_logger.get_all_steps(),
        "validations": _metrics_logger.get_all_validations(),
    })


@app.route("/api/metrics/since/<int:step>")
def metrics_since(step):
    if _metrics_logger is None:
        return jsonify({"steps": []})
    return jsonify({"steps": _metrics_logger.get_steps_since(step)})


# --- Post-hoc analysis endpoints ---

@app.route("/api/attention", methods=["POST"])
def attention():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids = _get_token_ids(prompt)
    result = get_attention_weights(_model, token_ids, _device)
    return jsonify(result)


@app.route("/api/activations", methods=["POST"])
def activations():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids = _get_token_ids(prompt)
    result = get_activation_analysis(_model, token_ids, _device)
    return jsonify(result)


@app.route("/api/attribution", methods=["POST"])
def attribution():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    top_k = data.get("top_k", 10)
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids = _get_token_ids(prompt)
    if len(token_ids) < 2:
        return jsonify({"error": "Prompt too short (need at least 2 tokens)"}), 400

    result = get_logit_attribution(_model, token_ids, _device, top_k=top_k)
    return jsonify(result)


@app.route("/api/ablation", methods=["POST"])
def ablation():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids = _get_token_ids(prompt)
    if len(token_ids) < 2:
        return jsonify({"error": "Prompt too short"}), 400

    result = get_head_ablation(_model, token_ids, _device)
    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def predict():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids, token_strings = tokenize_prompt(prompt)
    if _config and len(token_ids) > _config.max_seq_len:
        token_ids = token_ids[:_config.max_seq_len]
        token_strings = token_strings[:_config.max_seq_len]

    import torch
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=_device)
    with torch.no_grad():
        logits, _ = _model(input_ids)

    probs = torch.softmax(logits[0, -1], dim=-1)
    top_probs, top_ids = probs.topk(10)

    from viz.analysis import _get_tokenizer
    tok = _get_tokenizer()
    predictions = [
        {"token": tok.id_to_piece(int(tid)), "prob": float(p)}
        for tid, p in zip(top_ids.tolist(), top_probs.tolist())
    ]

    return jsonify({"tokens": token_strings, "predictions": predictions})


@app.route("/api/dla", methods=["POST"])
def dla():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    position = data.get("position", -1)
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids = _get_token_ids(prompt)
    if len(token_ids) < 2:
        return jsonify({"error": "Prompt too short"}), 400

    result = get_direct_logit_attribution(_model, token_ids, _device, position=position)
    return jsonify(result)


@app.route("/api/rope_analysis", methods=["POST"])
def rope_analysis():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids = _get_token_ids(prompt)
    result = get_rope_analysis(_model, token_ids, _device)
    return jsonify(result)


# --- Captum endpoints ---

@app.route("/api/captum/integrated_gradients", methods=["POST"])
def captum_ig():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    target_position = data.get("target_position", -1)
    n_steps = data.get("n_steps", 50)
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids = _get_token_ids(prompt)
    if len(token_ids) < 2:
        return jsonify({"error": "Prompt too short"}), 400

    result = get_integrated_gradients(
        _model, token_ids, _device,
        target_position=target_position, n_steps=n_steps,
    )
    return jsonify(result)


@app.route("/api/captum/layer_conductance", methods=["POST"])
def captum_lc():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    target_position = data.get("target_position", -1)
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids = _get_token_ids(prompt)
    if len(token_ids) < 2:
        return jsonify({"error": "Prompt too short"}), 400

    result = get_layer_conductance(
        _model, token_ids, _device, target_position=target_position,
    )
    return jsonify(result)


@app.route("/api/captum/saliency", methods=["POST"])
def captum_saliency():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    target_position = data.get("target_position", -1)
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids = _get_token_ids(prompt)
    if len(token_ids) < 2:
        return jsonify({"error": "Prompt too short"}), 400

    result = get_token_saliency(
        _model, token_ids, _device, target_position=target_position,
    )
    return jsonify(result)


# --- Circuits endpoints ---

@app.route("/api/circuits/patching", methods=["POST"])
def circuits_patching():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    clean_prompt = data.get("clean_prompt", "")
    corrupted_prompt = data.get("corrupted_prompt", "")
    if not clean_prompt or not corrupted_prompt:
        return jsonify({"error": "Both clean_prompt and corrupted_prompt required"}), 400

    clean_ids = _get_token_ids(clean_prompt)
    corrupted_ids = _get_token_ids(corrupted_prompt)
    if len(clean_ids) < 2 or len(corrupted_ids) < 2:
        return jsonify({"error": "Prompts too short"}), 400

    result = get_activation_patching(_model, clean_ids, corrupted_ids, _device)
    return jsonify(result)


@app.route("/api/circuits/steering", methods=["POST"])
def circuits_steering():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    layer = data.get("layer", 0)
    component = data.get("component", "head")
    head = data.get("head", 0)
    scale = data.get("scale", 0.0)

    if component not in ("head", "ffn"):
        return jsonify({"error": "component must be 'head' or 'ffn'"}), 400
    if not (0 <= layer < _config.n_layers):
        return jsonify({"error": f"layer must be 0-{_config.n_layers - 1}"}), 400
    if component == "head" and not (0 <= head < _config.n_heads):
        return jsonify({"error": f"head must be 0-{_config.n_heads - 1}"}), 400

    token_ids = _get_token_ids(prompt)
    result = get_activation_steering(
        _model, token_ids, _device,
        layer=layer, component=component, head=head, scale=scale,
    )
    return jsonify(result)


@app.route("/api/circuits/swapping", methods=["POST"])
def circuits_swapping():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    source_prompt = data.get("source_prompt", "")
    target_prompt = data.get("target_prompt", "")
    if not source_prompt or not target_prompt:
        return jsonify({"error": "Both source_prompt and target_prompt required"}), 400

    layer = data.get("layer", 0)
    component = data.get("component", "residual")
    if component not in ("residual", "attn", "ffn"):
        return jsonify({"error": "component must be 'residual', 'attn', or 'ffn'"}), 400
    if not (0 <= layer < _config.n_layers):
        return jsonify({"error": f"layer must be 0-{_config.n_layers - 1}"}), 400

    source_ids = _get_token_ids(source_prompt)
    target_ids = _get_token_ids(target_prompt)
    result = get_activation_swapping(
        _model, source_ids, target_ids, _device,
        layer=layer, component=component,
    )
    return jsonify(result)


@app.route("/api/circuits/precomputation", methods=["POST"])
def circuits_precomputation():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    top_k = data.get("top_k", 5)
    token_ids = _get_token_ids(prompt)
    if len(token_ids) < 3:
        return jsonify({"error": "Prompt too short (need at least 3 tokens)"}), 400

    result = get_precomputation_detection(_model, token_ids, _device, top_k=top_k)
    return jsonify(result)


# --- Embedding space ---

@app.route("/api/embeddings", methods=["POST"])
def embeddings():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    method = data.get("method", "pca")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids = _get_token_ids(prompt)
    result = get_embedding_space(_model, token_ids, _device, method=method)
    return jsonify(result)


# --- Token waterfall ---

@app.route("/api/waterfall", methods=["POST"])
def waterfall():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    top_k = data.get("top_k", 5)
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids = _get_token_ids(prompt)
    if len(token_ids) < 2:
        return jsonify({"error": "Prompt too short"}), 400

    result = get_token_waterfall(_model, token_ids, _device, top_k=top_k)
    return jsonify(result)


# --- Prompt comparison ---

@app.route("/api/compare", methods=["POST"])
def compare():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt_a = data.get("prompt_a", "")
    prompt_b = data.get("prompt_b", "")
    if not prompt_a or not prompt_b:
        return jsonify({"error": "Both prompt_a and prompt_b required"}), 400

    ids_a = _get_token_ids(prompt_a)
    ids_b = _get_token_ids(prompt_b)
    if len(ids_a) < 2 or len(ids_b) < 2:
        return jsonify({"error": "Prompts too short"}), 400

    result = get_prompt_comparison(_model, ids_a, ids_b, _device)
    return jsonify(result)


# --- Neuron browser ---

@app.route("/api/neurons/overview", methods=["POST"])
def neurons_overview():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    layer = data.get("layer", 0)
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    if not (0 <= layer < _config.n_layers):
        return jsonify({"error": f"layer must be 0-{_config.n_layers - 1}"}), 400

    token_ids = _get_token_ids(prompt)
    result = get_neuron_overview(_model, token_ids, _device, layer=layer)
    return jsonify(result)


@app.route("/api/neurons/detail", methods=["POST"])
def neurons_detail():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    layer = data.get("layer", 0)
    neuron_idx = data.get("neuron_idx", 0)
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    if not (0 <= layer < _config.n_layers):
        return jsonify({"error": f"layer must be 0-{_config.n_layers - 1}"}), 400

    token_ids = _get_token_ids(prompt)
    result = get_neuron_detail(_model, token_ids, _device, layer=layer, neuron_idx=neuron_idx)
    return jsonify(result)


@app.route("/api/neurons/network", methods=["POST"])
def neurons_network():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    layer = data.get("layer", 0)
    top_k = data.get("top_k", 20)
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    if not (0 <= layer < _config.n_layers):
        return jsonify({"error": f"layer must be 0-{_config.n_layers - 1}"}), 400

    token_ids = _get_token_ids(prompt)
    result = get_neuron_network(_model, token_ids, _device, layer=layer, top_k=top_k)
    return jsonify(result)


# --- Attention flow ---

@app.route("/api/attention_flow", methods=["POST"])
def attention_flow():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    token_ids = _get_token_ids(prompt)
    result = get_attention_flow(_model, token_ids, _device)
    return jsonify(result)


# --- Tokenizer utilities ---

@app.route("/api/tokenizer/encode", methods=["POST"])
def tokenizer_encode():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    token_ids, token_strings = tokenize_prompt(text)
    return jsonify({
        "token_ids": token_ids,
        "token_strings": token_strings,
        "num_tokens": len(token_ids),
    })


@app.route("/api/tokenizer/search", methods=["POST"])
def tokenizer_search():
    data = request.get_json()
    query = data.get("query", "")
    limit = data.get("limit", 50)
    if not query:
        return jsonify({"error": "No query provided"}), 400

    from viz.analysis import _get_tokenizer
    tok = _get_tokenizer()
    results = []
    for i in range(tok.vocab_size):
        piece = tok.id_to_piece(i)
        if query.lower() in piece.lower():
            results.append({"id": i, "token": piece})
            if len(results) >= limit:
                break

    return jsonify({"query": query, "results": results})


@app.route("/api/tokenizer/neighbors", methods=["POST"])
def tokenizer_neighbors():
    err = _require_model()
    if err:
        return err
    data = request.get_json()
    token_id = data.get("token_id", 0)
    top_k = data.get("top_k", 20)

    from viz.analysis import _get_tokenizer
    tok = _get_tokenizer()

    import torch
    embed = _model.tok_embeddings.weight.detach()  # (vocab_size, dim)
    vocab_size = embed.shape[0]
    if token_id < 0 or token_id >= vocab_size:
        return jsonify({"error": "Invalid token_id"}), 400

    target = embed[token_id].unsqueeze(0)  # (1, dim)
    sims = torch.nn.functional.cosine_similarity(target, embed, dim=1)  # (vocab_size,)
    top_sims, top_ids = sims.topk(top_k + 1)  # +1 to skip self

    neighbors = []
    for sim_val, tid in zip(top_sims.tolist(), top_ids.tolist()):
        if tid == token_id:
            continue
        neighbors.append({
            "id": tid,
            "token": tok.id_to_piece(tid),
            "similarity": round(sim_val, 4),
        })

    return jsonify({
        "token_id": token_id,
        "token": tok.id_to_piece(token_id),
        "neighbors": neighbors[:top_k],
    })


# --- WebSocket events ---

@socketio.on("connect")
def handle_connect():
    if _metrics_logger is not None:
        emit("metrics_history", {
            "steps": _metrics_logger.get_all_steps(),
            "validations": _metrics_logger.get_all_validations(),
        })


def emit_step_update(metrics: dict):
    socketio.emit("step_update", metrics)


def emit_val_update(metrics: dict):
    socketio.emit("val_update", metrics)


def emit_training_complete(best_val_loss: float):
    socketio.emit("training_complete", {"best_val_loss": best_val_loss})
