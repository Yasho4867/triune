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


if __name__ == "__main__":
    unittest.main()
