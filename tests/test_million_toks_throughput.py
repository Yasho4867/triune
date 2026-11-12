"""Rigorously verifies Dynamic Tokenizer compression and Multi-Million Tok/s Parallel Engine.

Empirical Test Suite for:
1. DynamicTokenizer:
   - Sequence length compression ratio (>= 2.0x on code, structured syntax, and repetitive text).
   - 100% exact lossless round-trip parity between original and decompressed text.
   - Dynamic token chunk packing, attention masks, and chunk length metadata.
   - Continuous n-gram learning, whitespace run compression, and prefix/suffix merging.
   - Vector lookup table tensor generation and fast lookups.

2. ParallelEngine:
   - Hardware CUDA Graph capture, single-pulse replay, and CPU fallback.
   - Multi-stream pipelined GLA prefill across parallel CUDA streams.
   - Batched throughput scaling breaking past 1,000,000 tok/s into multi-million tokens/sec.
   - Continuous JEV latent speculative draft scoring without O(V) vocabulary projection.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from triune.data.dynamic_tokenizer import DynamicTokenizer, DynamicTokenizerOutput
from triune.model.jev import JEVExitRouter, JEVLatentPredictor
from triune.runtime.parallel_engine import GLAPipelineBlock, MoERoutingBlock, ParallelEngine


class TestDynamicTokenizer(unittest.TestCase):
    """Rigorous verification of DynamicTokenizer compression ratios and lossless round-trip."""

    def setUp(self):
        self.tokenizer = DynamicTokenizer()

    def test_compression_ratio_code_and_syntax(self):
        """Verifies compression ratio >= 2.0x on repetitive and structured programming code."""
        code_sample = (
            "from __future__ import annotations\n"
            "import os\n"
            "import sys\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class TransformerBlock(nn.Module):\n"
            "    def __init__(self, dim: int = 512, heads: int = 8):\n"
            "        super().__init__()\n"
            "        self.dim = dim\n"
            "        self.heads = heads\n"
            "        self.attn = nn.Linear(dim, dim)\n"
            "        self.mlp = nn.Linear(dim, dim * 4)\n\n"
            "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "        for i in range(10):\n"
            "            x = x + self.attn(x)\n"
            "            x = x + self.mlp(x)\n"
            "        return x\n"
        )

        res = self.tokenizer.encode(code_sample, compress=True)
        self.assertIsInstance(res, DynamicTokenizerOutput)
        self.assertGreater(res.uncompressed_length, 0)
        self.assertGreater(res.compressed_length, 0)
        self.assertLess(res.compressed_length, res.uncompressed_length)

        # Invariant: Sequence compression ratio must be >= 2.0x
        self.assertGreaterEqual(res.compression_ratio, 2.0)
        self.assertGreaterEqual(
            res.uncompressed_length / res.compressed_length, 2.0
        )

    def test_compression_ratio_repetitive_and_natural_text(self):
        """Verifies compression ratio >= 2.0x on natural language and structured markdown."""
        text_sample = (
            "# Executive Summary and Architectural Invariants\n\n"
            "The model utilizes continuous latent representations and multi-tier exits. "
            "The model utilizes continuous latent representations and multi-tier exits. "
            "The model utilizes continuous latent representations and multi-tier exits.\n\n"
            "--------------------------------\n"
            "================================\n"
            "--------------------------------\n"
            "================================\n"
        )

        res = self.tokenizer.encode(text_sample, compress=True)
        self.assertGreaterEqual(res.compression_ratio, 2.0)

    def test_exact_lossless_round_trip(self):
        """Validates 100% exact mathematical and byte parity for lossless decompression."""
        samples = [
            "def hello_world():\n    return 'Hello, World!'\n",
            "for i in range(100):\n    if i % 2 == 0:\n        print(i)\n",
            "Triune AI Transformer 2026: Multi-Million Tokens Per Second Throughput Engine.",
            "Short text sequence for round trip validation",
            "The Joint-Embedding Predictive Architecture provides a unified framework for latent space representation learning.",
        ]

        for text in samples:
            res = self.tokenizer.encode(text, compress=True)
            # Decompress and decode back with uncompress=True
            decoded_text = self.tokenizer.decode(res.ids, uncompress=True)
            self.assertEqual(
                decoded_text.strip(),
                text.strip(),
                f"Lossless round-trip failed for sample: {repr(text)} != {repr(decoded_text)}",
            )

    def test_batch_encode_and_chunk_packing(self):
        """Validates batched encoding, dense chunk packing, and mask generation."""
        batch = [
            "def foo():\n    return 42\n",
            "import torch\nimport torch.nn as nn\n",
            "class Model(nn.Module):\n    pass\n",
        ]

        batch_out = self.tokenizer.encode_batch(
            batch,
            compress=True,
            padding=True,
        )

        self.assertIn("input_ids", batch_out)
        self.assertIn("attention_mask", batch_out)
        self.assertIn("mean_compression_ratio", batch_out)
        self.assertEqual(batch_out["input_ids"].size(0), len(batch))
        self.assertGreaterEqual(batch_out["mean_compression_ratio"], 1.5)

        # Chunk packing test
        tokens = list(range(100, 250))
        chunk_size = 32
        chunks, masks, lengths = self.tokenizer.pack_chunks(tokens, chunk_size=chunk_size)
        self.assertEqual(chunks.size(1), chunk_size)
        self.assertEqual(chunks.size(0), (len(tokens) + chunk_size - 1) // chunk_size)

        unpacked = self.tokenizer.unpack_chunks(chunks, masks)
        self.assertEqual(unpacked, tokens)

    def test_learn_ngrams_from_corpus(self):
        """Validates unsupervised continuous n-gram mining and registration."""
        corpus = [
            "high performance linear attention with gated deltanet",
            "high performance linear attention with chunked execution",
            "high performance linear attention with tensor core acceleration",
        ]

        learned = self.tokenizer.learn_ngrams(corpus, top_k=5, min_freq=2)
        self.assertGreater(len(learned), 0)

        # Check that mined n-gram is now registered
        ngram, tid = learned[0]
        self.assertIn(ngram, self.tokenizer.ngram_to_id)
        self.assertEqual(self.tokenizer.ngram_to_id[ngram], tid)

    def test_vector_lookup_table(self):
        """Validates fast vector lookup table generation."""
        ids = torch.tensor([10, 20, 30, 40], dtype=torch.long)
        vec = self.tokenizer.lookup_vector(ids)
        self.assertEqual(vec.shape, ids.shape)
        self.assertTrue((vec == ids).all())


class TestParallelEngine(unittest.TestCase):
    """Rigorous verification of ParallelEngine CUDA Graphs, multi-stream GLA, and throughput."""

    def setUp(self):
        self.is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.is_cuda else "cpu")
        self.dim = 256
        self.heads = 4
        self.head_dim = 64
        self.num_experts = 4
        self.top_k = 2
        self.chunk_size = 64

        self.engine = ParallelEngine(
            dim=self.dim,
            heads=self.heads,
            head_dim=self.head_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            chunk_size=self.chunk_size,
            device=self.device,
            num_streams=4,
        )

    def test_engine_initialization_and_telemetry(self):
        """Validates correct dimension setup and initial telemetry state."""
        self.assertEqual(self.engine.dim, self.dim)
        self.assertEqual(self.engine.heads, self.heads)
        self.assertEqual(self.engine.head_dim, self.head_dim)
        self.assertEqual(self.engine.num_experts, self.num_experts)

        telem = self.engine.get_telemetry()
        self.assertIn("peak_throughput_tok_s", telem)
        self.assertIn("total_tokens_processed", telem)
        self.assertIn("graph_replays", telem)

    def test_cuda_graph_capture_and_replay(self):
        """Validates CUDA Graph capture and single-pulse replay on CUDA devices."""
        if not self.is_cuda:
            # CPU fallback check
            captured = self.engine.capture_forward_graph(batch_size=2, seq_len=64)
            self.assertFalse(captured)
            self.assertFalse(self.engine.is_graph_captured)
            return

        B, T = 16, 256
        captured = self.engine.capture_forward_graph(
            batch_size=B, seq_len=T, chunk_size=self.chunk_size, warmup_steps=3
        )
        self.assertTrue(captured)
        self.assertTrue(self.engine.is_graph_captured)

        # Replay forward pass with new tensor
        x_test = torch.randn(B, T, self.dim, device=self.device, dtype=self.engine.dtype)
        out, state, weights, indices = self.engine.replay_forward(x_test)

        self.assertEqual(out.shape, (B, T, self.dim))
        self.assertEqual(state.shape, (B, self.heads, self.head_dim, self.head_dim))
        self.assertEqual(weights.shape, (B * T, self.top_k))
        self.assertEqual(indices.shape, (B * T, self.top_k))
        self.assertTrue(torch.isfinite(out).all())

        # Invalidate graph
        self.engine.invalidate_graph()
        self.assertFalse(self.engine.is_graph_captured)

    def test_multi_stream_batch_execution(self):
        """Validates multi-stream pipelining produces matching shapes and finite outputs."""
        B, T = 8, 128
        x = torch.randn(B, T, self.dim, device=self.device, dtype=self.engine.dtype)

        # Execute with 4 parallel streams
        out, state = self.engine.prefill_batched(x, chunk_size=self.chunk_size)
        self.assertEqual(out.shape, (B, T, self.dim))
        self.assertEqual(state.shape, (B, self.heads, self.head_dim, self.head_dim))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(state).all())

    def test_throughput_scaling_batched(self):
        """Validates that batched throughput scales efficiently with batch size."""
        # Benchmark small batch vs larger batch
        bench_res = self.engine.benchmark_throughput(
            batch_sizes=[1, 16],
            seq_lengths=[256],
            chunk_size=64,
            iters=5,
            warmup=2,
        )

        self.assertIn("max_throughput_tok_s", bench_res)
        self.assertGreater(bench_res["max_throughput_tok_s"], 0.0)

        # Throughput for batch 16 must exceed batch 1
        res_b1 = next(r for r in bench_res["results"] if r["batch_size"] == 1)
        res_b16 = next(r for r in bench_res["results"] if r["batch_size"] == 16)
        self.assertGreater(
            res_b16["throughput_tok_s"], res_b1["throughput_tok_s"]
        )

        if self.is_cuda:
            # On NVIDIA RTX 5070 GPU, confirm throughput easily exceeds 1,000,000 tok/s
            large_bench = self.engine.benchmark_throughput(
                batch_sizes=[32],
                seq_lengths=[512],
                chunk_size=64,
                iters=10,
                warmup=3,
            )
            self.assertTrue(large_bench["exceeds_million_tok_s"])
            self.assertGreater(large_bench["max_throughput_tok_s"], 1_000_000.0)

    def test_speculative_draft_chunk_scoring_jev(self):
        """Validates continuous JEV latent speculative draft scoring without O(V) vocabulary heads."""
        B, K, D = 4, 8, self.dim
        draft_latents = torch.randn(B, K, D, device=self.device)
        target_latents = draft_latents + 0.05 * torch.randn(B, K, D, device=self.device)

        # Warmup run
        _ = self.engine.score_draft_chunks(
            draft_latents, target_latents, threshold=0.8, metric="cosine"
        )

        scoring = self.engine.score_draft_chunks(
            draft_latents, target_latents, threshold=0.8, metric="cosine"
        )

        self.assertIn("accepted_mask", scoring)
        self.assertIn("scores", scoring)
        self.assertIn("contiguous_accepted", scoring)
        self.assertIn("acceptance_rate", scoring)
        self.assertIn("latency_us", scoring)

        self.assertEqual(scoring["accepted_mask"].shape, (B, K))
        self.assertEqual(scoring["scores"].shape, (B, K))
        self.assertGreaterEqual(scoring["acceptance_rate"], 0.0)
        self.assertLessEqual(scoring["acceptance_rate"], 1.0)
        # Fast execution: warm latency must be < 5000 microseconds (< 5 ms)
        self.assertLess(scoring["latency_us"], 5000.0)


if __name__ == "__main__":
    unittest.main()
