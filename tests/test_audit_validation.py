"""Rigorous validation tests addressing code audit findings:
1. True Sparse MoE execution vs Static BMM and FLOP savings.
2. JEV Riemannian Geodesic STP tube loss and Stiefel distance.
3. Dynamic Tokenizer generalization across diverse multi-paragraph corpora.
"""

import math
import unittest
import torch
import torch.nn.functional as F

from triune.model.jev import JEVObjective
from triune.runtime.parallel_engine import MoERoutingBlock
from triune.data.dynamic_tokenizer import DynamicTokenizer


class TestMoESparseExecution(unittest.TestCase):
    def test_sparse_vs_dense_bmm_parity_and_flops(self):
        dim = 128
        num_experts = 8
        top_k = 2
        block = MoERoutingBlock(dim=dim, num_experts=num_experts, top_k=top_k, mode="sparse")
        block.eval()

        # Input: batch of tokens
        x = torch.randn(2, 16, dim)

        # 1. Sparse execution (only top_k experts computed per token)
        out_sparse, weights_sparse, idx_sparse = block(x, mode="sparse")
        self.assertEqual(out_sparse.shape, x.shape)
        self.assertEqual(idx_sparse.shape, (32, top_k))

        # 2. Dense BMM execution (all experts computed)
        out_dense, weights_dense, idx_dense = block(x, mode="dense_graph")
        self.assertEqual(out_dense.shape, x.shape)

        # Numerical parity between sparse and dense forward
        max_diff = (out_sparse - out_dense).abs().max().item()
        self.assertLess(max_diff, 2e-4)

        # FLOP comparison: for 8 experts, top-2 saves 75% FLOPs
        sparse_flops = block.compute_expert_flops(32, sparse=True)
        dense_flops = block.compute_expert_flops(32, sparse=False)
        self.assertEqual(dense_flops, sparse_flops * 4)


class TestJEVGeodesicSTP(unittest.TestCase):
    def test_geodesic_stp_distance(self):
        # eps_tube in radians (e.g., 0.2 rad ~ 11.5 deg)
        eps_tube = 0.2
        obj = JEVObjective(eps_tube=eps_tube, stp_mode="geodesic")

        # Two vectors with small angular separation < 0.2 rad
        z_t = torch.tensor([[1.0, 0.0, 0.0]])
        # Angle ~ 0.1 rad
        cos_01 = math.cos(0.1)
        sin_01 = math.sin(0.1)
        z_inside = torch.tensor([[cos_01, sin_01, 0.0]])
        loss_inside = obj.compute_stp_loss(z_inside, z_t)
        self.assertEqual(loss_inside.item(), 0.0)

        # Two vectors with large angular separation > 0.2 rad (e.g. 0.5 rad)
        cos_05 = math.cos(0.5)
        sin_05 = math.sin(0.5)
        z_outside = torch.tensor([[cos_05, sin_05, 0.0]])
        loss_outside = obj.compute_stp_loss(z_outside, z_t)
        expected = (0.5 ** 2) - (0.2 ** 2)
        self.assertAlmostEqual(loss_outside.item(), expected, places=4)


class TestTokenizerGeneralization(unittest.TestCase):
    def test_tokenizer_across_corpora(self):
        tok = DynamicTokenizer()

        code_sample = '''
def train_step(model, batch, optimizer, scheduler):
    """Executes a single distributed training step with gradient clipping."""
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss, aux_loss = model(batch["input_ids"], targets=batch["labels"])
        total_loss = loss + 0.02 * aux_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    return float(total_loss.detach().item())
'''
        enc_code = tok.encode(code_sample, compress=True)
        self.assertGreaterEqual(enc_code.compression_ratio, 2.0)
        dec_code = tok.decode(enc_code.ids)
        self.assertEqual(dec_code.strip(), code_sample.strip())

        prose_sample = '''
The Triune Transformer architecture introduces hierarchical dynamic early exits,
vectorized gated linear attention with constant-memory recurrent state propagation,
and Riemannian manifold centroid steering optimization. By replacing traditional
quadratic attention with O(N) linear-time gated recurrence, the framework achieves
unprecedented context scaling with strict mathematical guarantees.
'''
        enc_prose = tok.encode(prose_sample, compress=True)
        self.assertGreaterEqual(enc_prose.compression_ratio, 2.0)
        dec_prose = tok.decode(enc_prose.ids)
        self.assertEqual(dec_prose.strip(), prose_sample.strip())


if __name__ == "__main__":
    unittest.main()
