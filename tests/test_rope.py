"""
Unit tests specifically for Rotary Positional Embeddings (RoPE).

These tests verify the mathematical properties that make RoPE work:
  1. Rotation preserves vector magnitude (isometry)
  2. Relative position encoding: dot products depend on distance
  3. Frequency computation is correct
  4. Edge cases: position 0, single position, long sequences
"""

import sys
import os
import math

import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_vc.model import precompute_rope_frequencies, apply_rotary_embeddings


class TestRoPEFrequencies:
    """Tests for frequency precomputation."""

    def test_shape(self):
        cos, sin = precompute_rope_frequencies(64, 512)
        assert cos.shape == (512, 32)
        assert sin.shape == (512, 32)

    def test_position_zero(self):
        """At position 0, all angles are 0, so cos=1, sin=0."""
        cos, sin = precompute_rope_frequencies(64, 512)
        assert torch.allclose(cos[0], torch.ones(32), atol=1e-6)
        assert torch.allclose(sin[0], torch.zeros(32), atol=1e-6)

    def test_frequencies_decrease(self):
        """Higher dimension indices should have lower frequencies."""
        cos, sin = precompute_rope_frequencies(64, 512, theta=10000.0)
        # At position 1, the angle for each dim pair is theta_i = 1/10000^(2i/d)
        # These should decrease monotonically
        angles_at_pos1 = torch.atan2(sin[1], cos[1])
        # Angles should be non-increasing (higher dims rotate slower)
        for i in range(len(angles_at_pos1) - 1):
            assert angles_at_pos1[i] >= angles_at_pos1[i + 1] - 1e-6

    def test_different_theta(self):
        """Different base theta should produce different frequencies."""
        cos1, sin1 = precompute_rope_frequencies(64, 512, theta=10000.0)
        cos2, sin2 = precompute_rope_frequencies(64, 512, theta=500000.0)
        assert not torch.allclose(cos1, cos2)

    def test_even_dim_required(self):
        """head_dim must be even for RoPE."""
        with pytest.raises(AssertionError):
            precompute_rope_frequencies(63, 512)


class TestRoPEApplication:
    """Tests for applying RoPE to tensors."""

    def test_output_shape(self):
        """apply_rotary_embeddings should preserve input shape."""
        cos, sin = precompute_rope_frequencies(64, 32)
        x = torch.randn(2, 32, 4, 64)  # (batch, seq, heads, head_dim)
        out = apply_rotary_embeddings(x, cos, sin)
        assert out.shape == x.shape

    def test_magnitude_preservation(self):
        """Rotation preserves L2 norm (isometry property)."""
        cos, sin = precompute_rope_frequencies(64, 32)
        x = torch.randn(4, 32, 8, 64)

        x_rot = apply_rotary_embeddings(x, cos, sin)

        orig_norms = x.float().norm(dim=-1)
        rot_norms = x_rot.float().norm(dim=-1)
        assert torch.allclose(orig_norms, rot_norms, atol=1e-4)

    def test_identity_at_position_zero(self):
        """At position 0, rotation should be identity (angle=0)."""
        cos, sin = precompute_rope_frequencies(64, 32)
        x = torch.randn(1, 1, 1, 64)  # Single position at pos 0

        x_rot = apply_rotary_embeddings(x, cos[:1], sin[:1])
        assert torch.allclose(x, x_rot, atol=1e-5)

    def test_deterministic(self):
        """Same input + same position should give same output."""
        cos, sin = precompute_rope_frequencies(64, 32)
        x = torch.randn(1, 8, 2, 64)

        out1 = apply_rotary_embeddings(x, cos[:8], sin[:8])
        out2 = apply_rotary_embeddings(x, cos[:8], sin[:8])
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_relative_distance_invariance(self):
        """
        Core RoPE property: dot(R(q,m), R(k,n)) depends only on (m-n).

        This is THE key property that makes RoPE encode relative positions.
        We test multiple (m,n) pairs with the same relative distance.
        """
        cos, sin = precompute_rope_frequencies(64, 100)
        q = torch.randn(1, 1, 1, 64)
        k = torch.randn(1, 1, 1, 64)

        dots = []
        relative_distance = 5

        for base_pos in [0, 10, 20, 50, 80]:
            m = base_pos + relative_distance
            n = base_pos
            if m >= 100:
                continue

            q_rot = apply_rotary_embeddings(q, cos[m:m+1], sin[m:m+1])
            k_rot = apply_rotary_embeddings(k, cos[n:n+1], sin[n:n+1])
            dot = (q_rot * k_rot).sum().item()
            dots.append(dot)

        # All dots should be approximately equal
        for d in dots:
            assert abs(d - dots[0]) < 1e-3, (
                f"Relative position property violated: dots = {dots}"
            )

    def test_different_heads_independent(self):
        """Each head should be rotated independently."""
        cos, sin = precompute_rope_frequencies(64, 32)
        x = torch.zeros(1, 8, 4, 64)
        x[:, :, 0, :] = 1.0  # Only head 0 has values
        x[:, :, 2, :] = 2.0  # Head 2 has different values

        out = apply_rotary_embeddings(x, cos[:8], sin[:8])

        # Heads should have different rotated values
        assert not torch.allclose(out[:, :, 0, :], out[:, :, 2, :])

    def test_dtype_preservation(self):
        """Output dtype should match input dtype."""
        cos, sin = precompute_rope_frequencies(64, 32)

        # Test with float32
        x_f32 = torch.randn(1, 8, 2, 64, dtype=torch.float32)
        out_f32 = apply_rotary_embeddings(x_f32, cos[:8], sin[:8])
        assert out_f32.dtype == torch.float32

        # Test with float16
        x_f16 = x_f32.half()
        out_f16 = apply_rotary_embeddings(x_f16, cos[:8], sin[:8])
        assert out_f16.dtype == torch.float16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
