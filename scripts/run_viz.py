"""Launch the LLaMA-2 interpretability dashboard.

Usage:
    python scripts/run_viz.py --checkpoint checkpoints/best.pt --port 5001 --device cpu
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz.app import init_app, socketio


def main():
    parser = argparse.ArgumentParser(description="LLaMA-2 Interpretability Dashboard")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to model checkpoint (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="data/tokenizer.model",
        help="Path to SentencePiece tokenizer model (default: data/tokenizer.model)",
    )
    parser.add_argument(
        "--port", type=int, default=5001,
        help="Port to serve on (default: 5001)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run model on: cpu, cuda, mps (default: cpu)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable Flask debug mode",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Warning: Checkpoint not found at {args.checkpoint}")
        print("The dashboard will start but model-dependent features won't work.")
        print("Use /api/load_checkpoint to load a model later.\n")

    app = init_app(checkpoint_path=args.checkpoint, tokenizer_path=args.tokenizer, device=args.device)

    print(f"Starting LLaMA-2 Interpretability Dashboard")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Tokenizer:  {args.tokenizer}")
    print(f"  Device:     {args.device}")
    print(f"  URL:        http://localhost:{args.port}")
    print()

    socketio.run(app, host="0.0.0.0", port=args.port, debug=args.debug, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
