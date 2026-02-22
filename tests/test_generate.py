"""
Unit tests for the generation / inference pipeline.

Tests verify:
  1. KV cache produces same output as full forward pass
  2. Greedy decoding (temperature=0) is deterministic
  3. Top-k restricts to exactly k candidates
  4. Top-p (nucleus) sampling works correctly
  5. Sampling functions handle edge cases
"""

import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_vc.config import ModelConfig
from llama_vc.model import LLaMA
from llama_vc.generate import _sample_token, sample_top_p


@pytest.fixture
def tiny_model():
    """Create a tiny model for testing."""
    config = ModelConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=128,
        max_seq_len=32,
    )
    model = LLaMA(config)
    model.eval()
    return model


class TestSampling:
    """Tests for token sampling functions."""

    def test_greedy_deterministic(self):
        """Temperature=0 should always return the argmax token."""
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
        token1 = _sample_token(logits, temperature=0.0, top_k=0, top_p=1.0)
        token2 = _sample_token(logits, temperature=0.0, top_k=0, top_p=1.0)
        assert token1.item() == 1  # Index of max value (5.0)
        assert token2.item() == 1

    def test_temperature_sharpening(self):
        """Low temperature should make distribution sharper."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        # Run many samples with low temperature
        samples_low = []
        for _ in range(100):
            t = _sample_token(logits.clone(), temperature=0.01, top_k=0, top_p=1.0)
            samples_low.append(t.item())

        # Most samples should be the argmax (index 3)
        assert samples_low.count(3) > 90, "Low temperature should heavily favor argmax"

    def test_top_k_restricts_vocab(self):
        """Top-k should only allow the top k tokens."""
        logits = torch.tensor([[10.0, 5.0, 3.0, 1.0, 0.5, 0.1]])

        # With top_k=2, only indices 0 and 1 should be sampled
        samples = set()
        for _ in range(100):
            t = _sample_token(logits.clone(), temperature=1.0, top_k=2, top_p=1.0)
            samples.add(t.item())

        assert samples.issubset({0, 1}), f"Top-k=2 sampled tokens outside top 2: {samples}"

    def test_top_p_basic(self):
        """Top-p should restrict to minimum set exceeding threshold."""
        # Create a distribution where first 2 tokens have ~90% probability
        probs = torch.tensor([0.5, 0.4, 0.05, 0.03, 0.02])

        samples = set()
        for _ in range(100):
            token = sample_top_p(probs.clone(), p=0.85)
            samples.add(token.item())

        # With p=0.85, only tokens 0 and 1 should be sampled (cumsum = 0.9)
        assert samples.issubset({0, 1}), f"Top-p=0.85 sampled: {samples}"

    def test_top_p_includes_crossing_token(self):
        """The token that makes cumsum cross p should be included."""
        probs = torch.tensor([0.3, 0.3, 0.2, 0.1, 0.1])
        # p=0.55: cumsum after 2 tokens = 0.6 > 0.55
        # So tokens 0 and 1 should be included

        samples = set()
        for _ in range(200):
            token = sample_top_p(probs.clone(), p=0.55)
            samples.add(token.item())

        assert 0 in samples and 1 in samples

    def test_top_p_one_disables(self):
        """top_p=1.0 should include all tokens."""
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])

        samples = set()
        for _ in range(200):
            token = sample_top_p(probs.clone(), p=1.0)
            samples.add(token.item())

        # All 4 tokens should appear
        assert len(samples) == 4


class TestKVCache:
    """Tests for KV cache consistency."""

    def test_cache_matches_full_forward(self, tiny_model):
        """Cached generation should produce same logits as full forward."""
        model = tiny_model
        n_layers = model.config.n_layers
        tokens = torch.randint(0, 256, (1, 16))

        with torch.no_grad():
            # Full forward
            full_logits, _ = model(tokens)

            # Prefill + decode (pass [None]*n_layers to init fresh caches)
            logits_pre, _, kv_caches = model(tokens[:, :15], kv_caches=[None] * n_layers, start_pos=0)
            logits_dec, _, _ = model(tokens[:, 15:16], kv_caches=kv_caches, start_pos=15)

        assert torch.allclose(
            full_logits[:, -1, :], logits_dec[:, -1, :], atol=1e-4
        )

    def test_sequential_decode(self, tiny_model):
        """Multiple decode steps should accumulate cache correctly."""
        model = tiny_model
        n_layers = model.config.n_layers
        tokens = torch.randint(0, 256, (1, 8))

        with torch.no_grad():
            # Full forward
            full_logits, _ = model(tokens)

            # Prefill first 4 tokens
            _, _, kv_caches = model(tokens[:, :4], kv_caches=[None] * n_layers, start_pos=0)

            # Decode remaining 4 tokens one by one
            for i in range(4, 8):
                logits, _, kv_caches = model(
                    tokens[:, i:i+1], kv_caches=kv_caches, start_pos=i
                )

        # Last logits should match
        assert torch.allclose(
            full_logits[:, -1, :], logits[:, -1, :], atol=1e-4
        )

    def test_cache_grows_correctly(self, tiny_model):
        """KV cache should grow by 1 after each decode step."""
        model = tiny_model
        n_layers = model.config.n_layers
        tokens = torch.randint(0, 256, (1, 4))

        with torch.no_grad():
            # Prefill
            _, _, kv_caches = model(tokens, kv_caches=[None] * n_layers, start_pos=0)

            # Check cache length
            cache_k, cache_v = kv_caches[0]  # First layer
            assert cache_k.shape[1] == 4, f"Cache should have 4 entries, got {cache_k.shape[1]}"

            # Decode 1 more token
            new_token = torch.randint(0, 256, (1, 1))
            _, _, kv_caches = model(new_token, kv_caches=kv_caches, start_pos=4)

            cache_k, cache_v = kv_caches[0]
            assert cache_k.shape[1] == 5, f"Cache should have 5 entries, got {cache_k.shape[1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
