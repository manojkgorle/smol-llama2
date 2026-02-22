"""
llama-vc: Educational LLaMA-style LLM from scratch in pure PyTorch.

This package implements a complete LLaMA language model (~15M parameters)
for learning and understanding transformer architectures.

Key modules:
  - config:    Model and training configuration
  - model:     LLaMA architecture (RMSNorm, RoPE, SwiGLU, GQA, KV Cache)
  - tokenizer: SentencePiece BPE tokenizer with byte-fallback
  - dataset:   TinyStories data pipeline
  - train:     Training loop with mixed precision
  - generate:  Inference with KV cache and sampling
  - device:    Hardware abstraction (CUDA/MPS/CPU)
  - utils:     Logging, checkpointing, diagnostics
"""

__version__ = "0.1.0"
