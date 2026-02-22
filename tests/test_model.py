"""
Unit tests for the LLaMA model architecture.

Tests verify:
  1. Output shapes are correct for all components
  2. Parameter count matches expected 15,735,168
  3. Forward pass produces valid logits and loss
  4. Gradient flow: all trainable parameters receive gradients
  5. RMSNorm produces unit RMS output
  6. GQA reduces parameter count vs full MHA
"""

import sys
import os
import math

import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_vc.config import ModelConfig
from llama_vc.model import (
    RMSNorm,
    precompute_rope_frequencies,
    apply_rotary_embeddings,
    FeedForward,
    Attention,
    TransformerBlock,
    LLaMA,
)
from llama_vc.utils import count_parameters


@pytest.fixture
def config():
    """Default model config for testing."""
    return ModelConfig()


@pytest.fixture
def tiny_config():
    """Smaller config for faster tests."""
    return ModelConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=128,
        max_seq_len=32,
    )


class TestRMSNorm:
    """Tests for RMSNorm layer."""

    def test_output_shape(self):
        """RMSNorm should preserve input shape."""
        norm = RMSNorm(dim=384)
        x = torch.randn(2, 16, 384)
        out = norm(x)
        assert out.shape == x.shape

    def test_unit_rms(self):
        """After normalization (before scaling), RMS should be ~1.0."""
        norm = RMSNorm(dim=384)
        # Set gamma to 1 to isolate normalization
        norm.weight.data.fill_(1.0)
        x = torch.randn(2, 16, 384)
        out = norm(x)
        # Compute RMS of output
        rms = out.float().pow(2).mean(-1).sqrt()
        # Should be close to 1.0 (within tolerance)
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_learnable_scale(self):
        """RMSNorm weight should be a trainable parameter."""
        norm = RMSNorm(dim=64)
        assert norm.weight.requires_grad
        assert norm.weight.shape == (64,)


class TestRoPE:
    """Tests for Rotary Positional Embeddings."""

    def test_frequency_shape(self):
        """Precomputed frequencies should have correct shape."""
        cos, sin = precompute_rope_frequencies(head_dim=64, max_seq_len=512)
        assert cos.shape == (512, 32)  # (max_seq_len, head_dim//2)
        assert sin.shape == (512, 32)

    def test_rotation_preserves_magnitude(self):
        """RoPE rotation should preserve vector magnitude (it's a rotation)."""
        cos, sin = precompute_rope_frequencies(head_dim=64, max_seq_len=32)
        x = torch.randn(1, 32, 4, 64)  # (batch, seq, heads, head_dim)

        x_rotated = apply_rotary_embeddings(x, cos, sin)

        # L2 norm should be preserved (rotation is an isometry)
        orig_norm = x.float().norm(dim=-1)
        rotated_norm = x_rotated.float().norm(dim=-1)
        assert torch.allclose(orig_norm, rotated_norm, atol=1e-4)

    def test_different_positions_give_different_rotations(self):
        """Tokens at different positions should have different rotations."""
        cos, sin = precompute_rope_frequencies(head_dim=64, max_seq_len=32)
        x = torch.ones(1, 32, 1, 64)  # Same vector at all positions

        x_rotated = apply_rotary_embeddings(x, cos, sin)

        # Vectors at position 0 and position 1 should differ
        assert not torch.allclose(x_rotated[0, 0], x_rotated[0, 1], atol=1e-6)

    def test_relative_position_property(self):
        """
        Dot product of rotated vectors should depend on relative position.
        dot(R(q, m), R(k, n)) = dot(R(q, m+d), R(k, n+d)) for any offset d.
        """
        cos, sin = precompute_rope_frequencies(head_dim=64, max_seq_len=64)
        q = torch.randn(1, 1, 1, 64)
        k = torch.randn(1, 1, 1, 64)

        # Rotate q at pos 5, k at pos 3 (relative distance = 2)
        q_5 = apply_rotary_embeddings(q, cos[5:6], sin[5:6])
        k_3 = apply_rotary_embeddings(k, cos[3:4], sin[3:4])
        dot_1 = (q_5 * k_3).sum()

        # Rotate q at pos 15, k at pos 13 (same relative distance = 2)
        q_15 = apply_rotary_embeddings(q, cos[15:16], sin[15:16])
        k_13 = apply_rotary_embeddings(k, cos[13:14], sin[13:14])
        dot_2 = (q_15 * k_13).sum()

        # Dot products should be very close (same relative position)
        assert torch.allclose(dot_1, dot_2, atol=1e-4)


class TestFeedForward:
    """Tests for SwiGLU FFN."""

    def test_output_shape(self, config):
        """FFN output shape should match input shape."""
        ffn = FeedForward(config)
        x = torch.randn(2, 16, config.dim)
        out = ffn(x)
        assert out.shape == x.shape

    def test_three_matrices(self, config):
        """SwiGLU should have exactly 3 weight matrices."""
        ffn = FeedForward(config)
        assert hasattr(ffn, 'w_gate')
        assert hasattr(ffn, 'w_up')
        assert hasattr(ffn, 'w_down')

    def test_no_bias(self, config):
        """All FFN linear layers should have no bias (following LLaMA)."""
        ffn = FeedForward(config)
        assert ffn.w_gate.bias is None
        assert ffn.w_up.bias is None
        assert ffn.w_down.bias is None


class TestAttention:
    """Tests for Grouped Query Attention."""

    def test_output_shape(self, tiny_config):
        """Attention output should have shape (batch, seq, dim)."""
        attn = Attention(tiny_config)
        cos, sin = precompute_rope_frequencies(
            tiny_config.head_dim, tiny_config.max_seq_len
        )
        x = torch.randn(2, 16, tiny_config.dim)
        out, _ = attn(x, cos[:16], sin[:16])
        assert out.shape == (2, 16, tiny_config.dim)

    def test_gqa_fewer_kv_params(self, tiny_config):
        """GQA should have fewer KV parameters than full MHA."""
        gqa_attn = Attention(tiny_config)

        # Create MHA config (n_kv_heads == n_heads)
        mha_config = ModelConfig(
            dim=tiny_config.dim,
            n_heads=tiny_config.n_heads,
            n_kv_heads=tiny_config.n_heads,  # Full MHA
            hidden_dim=tiny_config.hidden_dim,
            max_seq_len=tiny_config.max_seq_len,
            vocab_size=tiny_config.vocab_size,
            n_layers=tiny_config.n_layers,
        )
        mha_attn = Attention(mha_config)

        gqa_params = count_parameters(gqa_attn)
        mha_params = count_parameters(mha_attn)

        assert gqa_params < mha_params, (
            f"GQA ({gqa_params}) should have fewer params than MHA ({mha_params})"
        )

    def test_causal_masking(self, tiny_config):
        """Output at position i should not depend on positions j > i."""
        attn = Attention(tiny_config)
        cos, sin = precompute_rope_frequencies(
            tiny_config.head_dim, tiny_config.max_seq_len
        )

        # Create input where the second half is different
        x = torch.randn(1, 8, tiny_config.dim)
        x_modified = x.clone()
        x_modified[:, 4:, :] = torch.randn(1, 4, tiny_config.dim)

        # Outputs at positions 0-3 should be identical
        out1, _ = attn(x, cos[:8], sin[:8])
        out2, _ = attn(x_modified, cos[:8], sin[:8])

        # First 4 positions should be the same (causal: don't see future)
        assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5)


class TestTransformerBlock:
    """Tests for a single transformer layer."""

    def test_output_shape(self, tiny_config):
        """TransformerBlock output should preserve shape."""
        block = TransformerBlock(layer_id=0, config=tiny_config)
        cos, sin = precompute_rope_frequencies(
            tiny_config.head_dim, tiny_config.max_seq_len
        )
        x = torch.randn(2, 16, tiny_config.dim)
        out, _ = block(x, cos[:16], sin[:16])
        assert out.shape == x.shape

    def test_residual_connection(self, tiny_config):
        """Output should be related to input via residual connection."""
        block = TransformerBlock(layer_id=0, config=tiny_config)
        cos, sin = precompute_rope_frequencies(
            tiny_config.head_dim, tiny_config.max_seq_len
        )
        x = torch.randn(2, 16, tiny_config.dim)
        out, _ = block(x, cos[:16], sin[:16])

        # Output should not be identical to input (the block does something)
        assert not torch.allclose(out, x)

        # But should be in a similar range (residual prevents drastic changes)
        assert out.std() < x.std() * 10  # Rough sanity check


class TestLLaMA:
    """Tests for the complete LLaMA model."""

    def test_param_count(self, config):
        """Total parameter count should match expected 15,735,168."""
        model = LLaMA(config)
        n_params = count_parameters(model)
        assert n_params == 15_735_168, (
            f"Expected 15,735,168 params, got {n_params:,}"
        )

    def test_forward_shape(self, tiny_config):
        """Forward pass should produce correct logits shape."""
        model = LLaMA(tiny_config)
        tokens = torch.randint(0, tiny_config.vocab_size, (2, 16))
        logits, loss = model(tokens)
        assert logits.shape == (2, 16, tiny_config.vocab_size)
        assert loss is None  # No targets provided

    def test_forward_with_loss(self, tiny_config):
        """Forward pass with targets should produce scalar loss."""
        model = LLaMA(tiny_config)
        tokens = torch.randint(0, tiny_config.vocab_size, (2, 16))
        targets = torch.randint(0, tiny_config.vocab_size, (2, 16))
        logits, loss = model(tokens, targets=targets)
        assert logits.shape == (2, 16, tiny_config.vocab_size)
        assert loss is not None
        assert loss.shape == ()  # Scalar
        assert loss.item() > 0  # Loss should be positive

    def test_gradient_flow(self, tiny_config):
        """All trainable parameters should receive gradients."""
        model = LLaMA(tiny_config)
        tokens = torch.randint(0, tiny_config.vocab_size, (2, 16))
        targets = torch.randint(0, tiny_config.vocab_size, (2, 16))

        _, loss = model(tokens, targets=targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_kv_cache_inference(self, tiny_config):
        """KV cache should produce same output as full forward pass."""
        model = LLaMA(tiny_config)
        model.eval()

        tokens = torch.randint(0, tiny_config.vocab_size, (1, 8))

        with torch.no_grad():
            # Full forward pass (no caching)
            full_logits, _ = model(tokens)

            # Prefill: first 7 tokens (initialize fresh caches with [None]*n_layers)
            logits_prefill, _, kv_caches = model(
                tokens[:, :7], kv_caches=[None] * tiny_config.n_layers, start_pos=0
            )

            # Decode: 8th token with cache
            logits_decode, _, _ = model(
                tokens[:, 7:8], kv_caches=kv_caches, start_pos=7
            )

        # Logits for the last position should match
        assert torch.allclose(
            full_logits[:, -1, :],
            logits_decode[:, -1, :],
            atol=1e-4,
        ), "KV cache output doesn't match full forward pass"

    def test_weight_tying(self):
        """When weight_tying=True, embedding and output should share weights."""
        config = ModelConfig(
            vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
            hidden_dim=128, max_seq_len=32, weight_tying=True,
        )
        model = LLaMA(config)
        assert model.tok_embeddings.weight is model.output.weight

    def test_no_weight_tying_default(self, config):
        """By default, weight_tying should be False."""
        model = LLaMA(config)
        assert model.tok_embeddings.weight is not model.output.weight


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
