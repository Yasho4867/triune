"""Numerical regression test for the non-FLA recurrent GLA fallback."""

from __future__ import annotations

import unittest

import torch

from triune.model.attention import _pytorch_chunk_gla


def _reference_gla(q, k, v, g, initial_state=None):
    batch, length, heads, head_dim = q.shape
    state = (
        initial_state.clone()
        if initial_state is not None
        else torch.zeros(batch, heads, head_dim, head_dim, dtype=q.dtype, device=q.device)
    )
    output = []
    for index in range(length):
        state = state * torch.exp(g[:, index]).unsqueeze(-1)
        state = state + k[:, index].unsqueeze(-1) @ v[:, index].unsqueeze(-2)
        output.append((q[:, index].unsqueeze(-2) @ state).squeeze(-2))
    return torch.stack(output, dim=1), state


class TestGLAFallback(unittest.TestCase):
    def test_matches_recurrent_reference_with_cached_state(self) -> None:
        torch.manual_seed(7)
        q = torch.randn(2, 5, 3, 4)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        g = torch.randn_like(q).clamp(max=0)
        initial_state = torch.randn(2, 3, 4, 4)

        output, state = _pytorch_chunk_gla(q, k, v, g, initial_state)
        expected_output, expected_state = _reference_gla(q, k, v, g, initial_state)

        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(state, expected_state)

    def test_chunked_vs_recurrent_equivalence_across_lengths_and_gradients(self) -> None:
        """Verifies _pytorch_chunk_gla matches exact token-by-token recurrence for outputs, states, and gradients."""
        torch.manual_seed(42)
        for length in (1, 63, 64, 65, 128, 256):
            for with_initial_state in (False, True):
                with self.subTest(length=length, with_initial_state=with_initial_state):
                    q_base = torch.randn(2, length, 2, 8, dtype=torch.float64)
                    k_base = torch.randn_like(q_base)
                    v_base = torch.randn_like(q_base)
                    g_base = -torch.rand_like(q_base) * 2.0 - 0.05
                    s_base = torch.randn(2, 2, 8, 8, dtype=torch.float64) if with_initial_state else None

                    # Chunked path (default chunk_size=64)
                    q1 = q_base.clone().requires_grad_(True)
                    k1 = k_base.clone().requires_grad_(True)
                    v1 = v_base.clone().requires_grad_(True)
                    g1 = g_base.clone().requires_grad_(True)
                    s1 = s_base.clone().requires_grad_(True) if s_base is not None else None

                    out_chunk, state_chunk = _pytorch_chunk_gla(q1, k1, v1, g1, initial_state=s1, chunk_size=64)
                    grad_out = torch.randn_like(out_chunk)
                    grad_state = torch.randn_like(state_chunk)
                    loss_chunk = (out_chunk * grad_out).sum() + (state_chunk * grad_state).sum()
                    loss_chunk.backward()

                    # Exact token-by-token reference
                    q2 = q_base.clone().requires_grad_(True)
                    k2 = k_base.clone().requires_grad_(True)
                    v2 = v_base.clone().requires_grad_(True)
                    g2 = g_base.clone().requires_grad_(True)
                    s2 = s_base.clone().requires_grad_(True) if s_base is not None else None

                    out_ref, state_ref = _reference_gla(q2, k2, v2, g2, initial_state=s2)
                    loss_ref = (out_ref * grad_out).sum() + (state_ref * grad_state).sum()
                    loss_ref.backward()

                    torch.testing.assert_close(out_chunk, out_ref, rtol=1e-10, atol=1e-10)
                    torch.testing.assert_close(state_chunk, state_ref, rtol=1e-10, atol=1e-10)
                    torch.testing.assert_close(q1.grad, q2.grad, rtol=1e-10, atol=1e-10)
                    torch.testing.assert_close(k1.grad, k2.grad, rtol=1e-10, atol=1e-10)
                    torch.testing.assert_close(v1.grad, v2.grad, rtol=1e-10, atol=1e-10)
                    torch.testing.assert_close(g1.grad, g2.grad, rtol=1e-10, atol=1e-10)
                    if with_initial_state:
                        torch.testing.assert_close(s1.grad, s2.grad, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()

