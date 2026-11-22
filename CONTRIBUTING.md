# Contributing & Empirical Verification Policy

Thank you for contributing to **Lucid Machines** (`lucid-machines` / the **Triune** architecture `TriuneTransformer` / **Polysom Studio**, building toward our first flagship model **Junkie**). To maintain scientific rigor across model code, custom kernels, benchmarks, papers, and CI, every change must satisfy the engineering, taxonomy, and verification standards below.

---

## 1. Benchmark Taxonomy & Scientific Claims

Always classify and label performance numbers using clear categories so isolated microkernels are never conflated with end-to-end model throughput:

1. **Isolated Kernel Microbenchmarks**: Single-operator or single-layer sweeps (e.g., `scripts/benchmark_competitors.py`, `benchmarks/moe_ffn_sparse_vs_dense.py`). Always specify tensor dimensions, batch/sequence shape, precision (`bf16`/`fp32`), and warmup count.
2. **End-to-End Model Throughput (`90.79M` / `64.21M` active)**: Complete checkpoint forward/decode benchmarks including embeddings, MoE routing, exits, and vocabulary projection (`scripts/benchmark_full_model_e2e.py` -> `benchmarks/FULL_MODEL_E2E_REPORT.md`), separated clearly from streaming runtime HTTP/CLI decode metrics (`40.8 tok/s` warm decode / `71.3 ms` TTFT on RTX 5070).
3. **Multi-Layer Wrapper vs. Legacy Block Diagnostics**: Distinguish multi-layer eager Python sweeps (`scripts/benchmark_long_context.py`) and legacy diagnostic modules (e.g., `MoERoutingBlock`) from production fast paths (`MoE_FFN` with `use_sparse_eval=True`).
4. **Tokenizer Compression vs. Model Computation**: Report held-out text compression ratios (`compress_mode` / lossless reconstruction) separately from compute-equivalent logit verification (`compress_mode="expand"`, `0.0` max logit diff vs. base tokenization).
5. **Speculative & Early-Exit Clarity**:
   - Explicitly restrict 100% token-for-token Cortex parity claims to **greedy decoding (`T=0.0`)**.
   - Stochastic speculative decoding (`T > 0.0`) uses Leviathan et al. rejection sampling (`exact_distribution`).
   - Disclose checkpoint calibration state when reporting CALM agreement (e.g., uncalibrated single-exit checkpoint `triune_step_24501.pt` vs. jointly calibrated multi-exit runs).

---

## 2. Mathematical Invariants & Architecture Guards

Changes to core subsystems must preserve and test:
- **CentroidSteer (`lucid_machines/optim/centroid.py`, `lucid_machines/ops/centroid_steer.py`)**: `sCholQR2` orthogonality (`||Q^T Q - I||_F`), sliced Householder fallback under rank deficiency, upper-triangular projection core (`R = triu(S_bar)`), centroid-deflated Krylov, independent `steer_step` reset, and the non-refresh steering moment-zeroing invariant (`M/V` column & row `r+1` zeroed whenever `needs_steer_refresh` rotates steering axes).
- **Layer Streaming (`lucid_machines/runtime/staging.py`, `lucid_machines/runtime/streaming.py`)**: Write-once parameter forward snapshots and `is_buffer=True` device-mutation copy-back across `release_gpu(clear_snapshot=False)`.
- **Compacted Per-Token Exits (`lucid_machines/model/transformer.py`)**: True per-token / per-sample CALM margin gating (`last_calm_exit_depths`) and GLA state scatter (`_scatter_cache`) across cached autoregressive steps.
- **Production Sparse MoE (`lucid_machines/model/moe.py`)**: `use_sparse_eval=True` by default, zero-sync `N_tok == 1` GPU path, true post-SwiGLU `last_hidden_centroids` capture, and parity against dense evaluation.

---

## 3. Mandatory Pre-Push Verification & CI Checks

Before committing or pushing:

```bash
python -m ruff check --select=E9,F63,F7,F82 .
python -m compileall -q lucid_machines
pytest -q
python -m unittest discover tests
python -m unittest tests.test_pipeline_integration tests.test_self_speculative_decoding tests.test_staging_invariants tests.test_triangular_core_integrity tests.test_benchmark_validity
python -m build --sdist --wheel
```

All checks and every GitHub Actions matrix job (`Python 3.10`, `3.11`, `3.12`, `3.13`) must pass completely (`conclusion == "success"`). Never weaken assertions, skip real failures, or check in untracked private scratch/agent directives (keep `WORKSPACE_RULES.md` untracked and gitignored).
