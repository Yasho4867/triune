"""Unit tests for DynamicTokenizer and ParallelEngine."""

import unittest
import torch
import torch.nn.functional as F

from triune.data import DynamicTokenizer, DynamicTokenizerOutput
from triune.runtime import ParallelEngine, GLAPipelineBlock, MoERoutingBlock
from triune.model.jev import JEVLatentPredictor, JEVExitRouter


class TestDynamicTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DynamicTokenizer(target_compression_ratio=2.6)

    def test_compression_ratio_code(self):
        code_sample = """def __init__(self, dim: int, num_experts: int = 8):
    super().__init__()
    self.dim = dim
    self.num_experts = num_experts
    for i in range(num_experts):
        self.experts.append(nn.Linear(dim, dim))"""

        enc = self.tokenizer.encode(code_sample, compress=True)
        self.assertGreaterEqual(enc.compression_ratio, 2.0)
        self.assertLessEqual(enc.compression_ratio, 3.8)
        self.assertGreater(enc.uncompressed_length, enc.compressed_length)

    def test_compression_ratio_syntax_runs(self):
        syntax_sample = """--------------------------------------------------------------------------------
================================================================================
    def forward(self, x):
        pass
    def forward(self, x):
        pass
--------------------------------------------------------------------------------"""
        enc = self.tokenizer.encode(syntax_sample, compress=True)
        self.assertGreaterEqual(enc.compression_ratio, 2.0)
        self.assertLessEqual(enc.compression_ratio, 3.8)

    def test_lossless_roundtrip_parity(self):
        text = "The Joint-Embedding Predictive Architecture provides a unified framework for latent space representation learning."
        enc = self.tokenizer.encode(text, compress=True)
        dec = self.tokenizer.decode(enc.ids, uncompress=True)
        self.assertEqual(dec.strip(), text.strip())

    def test_batch_encoding_and_decoding(self):
        texts = [
            "def __init__(self, dim: int):\n    self.dim = dim",
            "for i in range(10):\n    print(i)",
            "--------------------------------\n",
        ]
        batch = self.tokenizer.encode_batch(texts, compress=True, padding=True)
        self.assertEqual(batch["input_ids"].size(0), 3)
        self.assertIn("mean_compression_ratio", batch)
        self.assertGreaterEqual(batch["mean_compression_ratio"], 1.5)

        dec_texts = self.tokenizer.decode_batch(batch["input_ids"])
        for orig, dec in zip(texts, dec_texts):
            self.assertEqual(dec.strip(), orig.strip())

    def test_chunk_packing_and_unpacking(self):
        tokens = list(range(100, 250))
        chunk_size = 32
        chunks, masks, lengths = self.tokenizer.pack_chunks(tokens, chunk_size=chunk_size)
        self.assertEqual(chunks.size(1), chunk_size)
        self.assertEqual(chunks.size(0), (len(tokens) + chunk_size - 1) // chunk_size)

        unpacked = self.tokenizer.unpack_chunks(chunks, masks)
        self.assertEqual(unpacked, tokens)

    def test_vector_lookup(self):
        ids = torch.tensor([10, 20, 30], dtype=torch.long)
        vec = self.tokenizer.lookup_vector(ids)
        self.assertEqual(vec.shape, ids.shape)


class TestParallelEngine(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.engine = ParallelEngine(
            dim=256,
            heads=4,
            head_dim=64,
            num_experts=4,
            top_k=2,
            chunk_size=32,
            device=self.device,
        )

    def test_prefill_batched_forward(self):
        B, T, D = 8, 128, 256
        x = torch.randn(B, T, D, device=self.device, dtype=self.engine.dtype)
        out, state = self.engine.prefill_batched(x, chunk_size=32)
        self.assertEqual(out.shape, (B, T, D))
        self.assertEqual(state.shape, (B, 4, 64, 64))

    def test_unified_forward_moe(self):
        B, T, D = 4, 64, 256
        x = torch.randn(B, T, D, device=self.device, dtype=self.engine.dtype)
        out, state, weights, indices = self.engine.forward(x, use_graph=False)
        self.assertEqual(out.shape, (B, T, D))
        self.assertEqual(weights.shape, (B * T, 2))
        self.assertEqual(indices.shape, (B * T, 2))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required for graph test")
    def test_cuda_graph_capture_and_replay(self):
        B, T, D = 16, 256, 256
        captured = self.engine.capture_forward_graph(batch_size=B, seq_len=T, chunk_size=32)
        self.assertTrue(captured)
        self.assertTrue(self.engine.is_graph_captured)

        x_new = torch.randn(B, T, D, device=self.device, dtype=self.engine.dtype)
        r_out, r_state, r_w, r_idx = self.engine.replay_forward(x_new)
        self.assertEqual(r_out.shape, (B, T, D))
        self.assertEqual(r_w.shape, (B * T, 2))

    def test_jev_draft_chunk_scoring(self):
        B, K, D = 8, 4, 256
        z_draft = torch.randn(B, K, D, device=self.device, dtype=self.engine.dtype)
        # Construct target with high similarity
        z_target = z_draft + torch.randn(B, K, D, device=self.device, dtype=self.engine.dtype) * 0.05

        res = self.engine.score_draft_chunks(z_draft, z_target, threshold=0.85)
        self.assertIn("accepted_mask", res)
        self.assertIn("scores", res)
        self.assertIn("contiguous_accepted", res)
        self.assertGreater(res["total_accepted"], 0)
        self.assertGreater(res["acceptance_rate"], 0.5)

    def test_jev_integration_with_predictor_and_router(self):
        D = 256
        predictor = JEVLatentPredictor(dim=D).to(device=self.device, dtype=self.engine.dtype)
        router = JEVExitRouter(dim=D, threshold=0.85).to(device=self.device, dtype=self.engine.dtype)

        z_draft = torch.randn(4, 2, D, device=self.device, dtype=self.engine.dtype)
        res = self.engine.verify_speculative_draft(
            draft_tokens=None,
            draft_latents=z_draft,
            jev_predictor=predictor,
            jev_router=router,
        )
        self.assertIn("accepted_mask", res)
        self.assertIn("scores", res)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required for million throughput test")
    def test_million_throughput_benchmark(self):
        bench = self.engine.benchmark_throughput(
            batch_sizes=[16, 32],
            seq_lengths=[512, 1024],
            chunk_size=64,
            iters=10,
            warmup=3,
        )
        self.assertTrue(bench["exceeds_million_tok_s"])
        self.assertGreaterEqual(bench["max_throughput_tok_s"], 1_000_000.0)


if __name__ == "__main__":
    unittest.main()
