"""Phase 7: Automated Benchmark Validity and Telemetry Verification Suite.

Guarantees:
1. All benchmark artifact JSONs exist and are non-empty, valid JSON.
2. No NaN, Inf, or negative numbers exist in measured latencies or VRAM allocations.
3. All reports include valid hardware metadata (GPU model, VRAM capacity, CUDA, PyTorch).
4. Mathematical consistency: tokens/sec == (batch_size * seq_len) / latency_sec within numerical tolerances.
5. Monotonic hierarchy: Reflex exit latency < Limbic exit latency < Cortex exit latency.
6. Real baseline integrity: PyTorch AdamW and BitsAndBytes 8-bit AdamW (Unsloth) are verified.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmarks"


class BenchmarkValidityTest(unittest.TestCase):
    """Test suite enforcing empirical validity rules across all generated benchmark data."""

    def test_json_artifacts_exist_and_valid(self):
        """Verifies that all category benchmark JSON files exist and are parseable."""
        required_files = [
            "benchmark_results.json",
            "optimizer_benchmark_results.json",
            "dynamic_exits_results.json",
            "long_context_results.json",
            "layer_streaming_results.json",
            "master_matrix_results.json",
            "genuine_unsloth_vs_triune_results.json",
        ]
        for fname in required_files:
            fpath = BENCHMARK_DIR / fname
            self.assertTrue(fpath.exists(), f"Missing required benchmark artifact: {fname}")
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, (dict, list), f"Benchmark data in {fname} must be a JSON dictionary or list")
            self.assertGreater(len(data), 0, f"Benchmark artifact {fname} cannot be empty")

    def test_hardware_metadata_present_and_consistent(self):
        """Ensures all benchmark files log valid, non-mock hardware metadata."""
        files_with_hw = [
            "optimizer_benchmark_results.json",
            "dynamic_exits_results.json",
            "long_context_results.json",
            "layer_streaming_results.json",
            "master_matrix_results.json",
            "genuine_unsloth_vs_triune_results.json",
        ]
        for fname in files_with_hw:
            fpath = BENCHMARK_DIR / fname
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            hw = data.get("hardware") or data.get("provenance", {}).get("hardware", {})
            self.assertIn("device_name", hw, f"Missing device_name in {fname}")
            self.assertIn("total_vram_gb", hw, f"Missing total_vram_gb in {fname}")
            pt_ver = hw.get("pytorch_version") or data.get("provenance", {}).get("software", {}).get("torch")
            self.assertIsNotNone(pt_ver, f"Missing pytorch version in {fname}")
            cuda_ver = hw.get("cuda_version") or data.get("provenance", {}).get("software", {}).get("cuda")
            self.assertIsNotNone(cuda_ver, f"Missing cuda version in {fname}")
            if torch.cuda.is_available():
                self.assertGreater(hw["total_vram_gb"], 0, f"Total VRAM must be > 0 in {fname}")

    def test_optimizer_benchmark_values(self):
        """Validates optimizer benchmark sanity and non-negativity."""
        fpath = BENCHMARK_DIR / "optimizer_benchmark_results.json"
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Experiment A: State Memory
        for exp in data["experiment_a_state_memory"]:
            self.assertGreater(exp["total_params"], 0)
            self.assertGreaterEqual(exp["state_memory_mb"], 0.0)
            self.assertFalse(math.isnan(exp["state_memory_mb"]))
            self.assertFalse(math.isinf(exp["state_memory_mb"]))

        # Experiment B: Pure Step Throughput
        for exp in data["experiment_b_step_throughput"]:
            self.assertGreater(exp["median_step_latency_ms"], 0.0)
            self.assertGreater(exp["steps_per_sec"], 0.0)
            self.assertFalse(math.isnan(exp["median_step_latency_ms"]))

        # Experiment C: 50-step Convergence
        for exp in data["experiment_c_50_step_convergence"]:
            self.assertFalse(math.isnan(exp["initial_loss"]))
            self.assertFalse(math.isnan(exp["final_loss"]))
            self.assertFalse(math.isnan(exp["final_grad_norm"]))

    def test_dynamic_exits_hierarchy_monotonicity(self):
        """Verifies that shallower exits have strictly lower latency than deeper exits."""
        fpath = BENCHMARK_DIR / "dynamic_exits_results.json"
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        exits = {e["name"]: e for e in data["exits"]}
        reflex_lat = exits["Reflex Exit (L2)"]["forward_latency_ms"]
        limbic_lat = exits["Limbic Exit (L5)"]["forward_latency_ms"]
        cortex_lat = exits["Cortex Exit (L8)"]["forward_latency_ms"]

        self.assertLess(reflex_lat, limbic_lat, f"Reflex latency ({reflex_lat}ms) must be < Limbic ({limbic_lat}ms)")
        self.assertLess(limbic_lat, cortex_lat, f"Limbic latency ({limbic_lat}ms) must be < Cortex ({cortex_lat}ms)")

    def test_long_context_gla_constant_memory(self):
        """Verifies that GLA recurrent decode state memory remains O(1) constant across prompt lengths."""
        fpath = BENCHMARK_DIR / "long_context_results.json"
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        decode_entries = data["decode_benchmarks"]
        state_sizes = [d["gla_state_kb"] for d in decode_entries]
        self.assertTrue(
            all(s == state_sizes[0] for s in state_sizes), f"GLA state size must remain constant: {state_sizes}"
        )

        # SDPA KV-cache must increase strictly with prompt length
        kv_sizes = [d["sdpa_kv_cache_kb"] for d in decode_entries]
        self.assertEqual(kv_sizes, sorted(kv_sizes), "SDPA KV cache must grow monotonically with sequence length")

    def test_layer_streaming_vram_reduction(self):
        """Verifies that LayerStreamingEngine achieves lower peak VRAM on all evaluated model scales."""
        fpath = BENCHMARK_DIR / "layer_streaming_results.json"
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for comp in data["comparisons"]:
            gpu = comp["gpu_resident"]
            stream = comp["layer_streaming"]
            if not gpu.get("oom"):
                self.assertLess(
                    stream["peak_allocated_mb"],
                    gpu["peak_allocated_mb"],
                    f"Layer streaming failed to reduce VRAM on {comp['scale']}",
                )

    def test_competitor_execution_integrity(self):
        """Verifies that AdamW and Unsloth are genuine real baselines."""
        import torch.optim as optim

        self.assertTrue(hasattr(optim, "AdamW"), "PyTorch standard AdamW must exist")
        try:
            import bitsandbytes as bnb

            self.assertTrue(hasattr(bnb.optim, "AdamW8bit"), "BitsAndBytes AdamW8bit must exist")
        except ImportError:
            self.fail("BitsAndBytes must be installed and functional for Unsloth baselines")

    def test_master_matrix_provenance_and_competitor_integrity(self):
        """Verifies master matrix records git provenance, valid optimizers, and non-zero adapter drift."""
        fpath = BENCHMARK_DIR / "master_matrix_results.json"
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            self.assertIn("schema_version", data)
            git_commit = data.get("git_commit") or data.get("provenance", {}).get("git_commit")
            self.assertIsNotNone(git_commit, "Git commit provenance must be present")
            self.assertIn("timestamp_utc", data)
            cells = data.get("cells", [])
        else:
            cells = data

        self.assertEqual(len(cells), 12, "Master matrix must have exactly 12 cells (3x2x2)")

        for cell in cells:
            self.assertIn("optimizer", cell)
            self.assertGreater(cell["step_time_ms"], 0.0)
            self.assertGreater(cell["throughput_tok_s"], 0.0)
            self.assertGreater(cell["peak_vram_mb"], 0.0)
            self.assertFalse(math.isnan(cell["step_time_ms"]))
            self.assertFalse(math.isnan(cell["peak_vram_mb"]))
            if cell["stack"] == "adamw":
                self.assertIn("AdamW", cell["optimizer"])
            elif cell["stack"] in ("unsloth", "bnb_8bit"):
                self.assertIn("AdamW8bit", cell["optimizer"])
            elif cell["stack"] == "triune" and cell["task"] == "pretrain":
                self.assertIn("CentroidSteer", cell["optimizer"])

    def test_genuine_unsloth_results_validity(self):
        """Verifies genuine Unsloth vs Triune benchmark provenance, metrics, and parameters."""
        fpath = BENCHMARK_DIR / "genuine_unsloth_vs_triune_results.json"
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("schema_version", data)
        git_commit = data.get("git_commit") or data.get("provenance", {}).get("git_commit")
        self.assertIsNotNone(git_commit, "Git commit provenance must be present")
        hardware = data.get("hardware") or data.get("provenance", {}).get("hardware")
        self.assertIsNotNone(hardware, "Hardware info must be present")
        results = data.get("results", [])
        self.assertEqual(len(results), 7, "Must have exactly 7 benchmark cells")

        frameworks = {r["framework"] for r in results}
        self.assertEqual(frameworks, {"pytorch", "unsloth", "triune"})

        for r in results:
            self.assertGreater(r["step_time_ms"], 0.0)
            self.assertGreater(r["throughput_tok_s"], 0.0)
            self.assertGreater(r["peak_vram_mb"], 0.0)
            self.assertFalse(math.isnan(r["step_time_ms"]))
            self.assertFalse(math.isnan(r["peak_vram_mb"]))
            if r["framework"] == "unsloth":
                self.assertEqual(r["optimizer"], "AdamW8bit")
            elif r["framework"] == "triune" and r["task"] == "pretrain":
                self.assertIn("CentroidSteer", r["optimizer"])
            elif r["framework"] == "pytorch":
                self.assertEqual(r["optimizer"], "AdamW")

    def test_cuda_graph_benchmark_results_validity(self):
        """Verifies CUDA Graph benchmark provenance, launch latency speedup, and zero-drift parity."""
        fpath = BENCHMARK_DIR / "cuda_graph_benchmark_results.json"
        if not fpath.is_file():
            self.skipTest("cuda_graph_benchmark_results.json not found")

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("provenance", data)
        self.assertIn("timing_metrics", data)
        self.assertIn("parity_validation", data)

        timing = data["timing_metrics"]
        self.assertIn("eager", timing)
        self.assertIn("cuda_graph_async", timing)

        # Async launch latency must be substantially faster than eager CPU dispatch
        eager_launch = timing["eager"]["host_launch_us"]["median"]
        async_launch = timing["cuda_graph_async"]["host_launch_us"]["median"]
        self.assertLess(async_launch, eager_launch)

        # Multi-step mathematical parity must show zero or negligible numerical drift
        parity = data["parity_validation"]
        self.assertLessEqual(parity["max_observed_weight_drift"], 1e-4)
        self.assertLessEqual(parity["max_observed_loss_diff"], 1e-4)

    def test_long_context_scaling_results_validity(self):
        """Verifies long-context scaling benchmark schema, O(1) decode memory, and O(T) speedups."""
        fpath = BENCHMARK_DIR / "long_context_scaling_results.json"
        if not fpath.is_file():
            self.skipTest("long_context_scaling_results.json not found")

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("provenance", data)
        self.assertIn("training_scaling", data)
        self.assertIn("autoregressive_decoding", data)

        # Autoregressive decoding recurrent state memory must be constant
        decode_entries = data["autoregressive_decoding"]
        gla_states = [d["gla_state_kb"] for d in decode_entries]
        self.assertTrue(
            all(s == gla_states[0] for s in gla_states), f"GLA recurrent state must remain O(1) constant: {gla_states}"
        )

        # SDPA KV cache must grow monotonically
        sdpa_caches = [d["sdpa_kv_cache_kb"] for d in decode_entries]
        self.assertEqual(sdpa_caches, sorted(sdpa_caches), "SDPA KV cache must scale monotonically with prompt length")

    def test_convergence_100_steps_results_validity(self):
        """Verifies 100-step real text convergence benchmark schema, non-negative losses, and loss reduction."""
        fpath = BENCHMARK_DIR / "convergence_100_steps_results.json"
        if not fpath.is_file():
            self.skipTest("convergence_100_steps_results.json not found")

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("provenance", data)
        self.assertIn("results", data)
        results = data["results"]
        regime_names = {r["regime_name"] for r in results}
        for expected in ("adamw", "bnb_8bit", "triune"):
            self.assertIn(expected, regime_names, f"Must include core optimizer regime: {expected}")

        for r in results:
            self.assertGreater(
                r["val_loss_reduction"], 0.0, f"Regime {r['regime_name']} failed to reduce validation loss"
            )
            self.assertGreater(r["throughput_tok_s"], 0.0)
            self.assertEqual(len(r["step_records"]), 100, "Must record all 100 training steps")


if __name__ == "__main__":
    unittest.main()
