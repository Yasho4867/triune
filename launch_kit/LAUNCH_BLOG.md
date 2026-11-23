# Introducing Lucid Machines `v1.0.0`: The Triune Architecture, Multi-Directional CentroidSteer, Polysom Studio & Inaugural Flagship Model Junkie

**Release Date:** November 2026  
**Company / Package / CLI:** `Lucid Machines` (`lucid-machines` / `lucid_machines` SDK, inaugural release `v1.0.0`)  
**Core Neural Architecture:** `Triune` (`TriuneTransformer` / `TriuneModel` / `TriuneConfig`, templates `models/configs/triune-*.json`)  
**Inaugural Flagship Models To Train:** **Junkie** (`models/configs/junkie-mini.json`, `junkie-1b.json`, `junkie-3b.json`, `junkie-7b.json`, `junkie.json`)  
**Desktop Workbench:** `Polysom Studio v1.0.0` (`polysom` CLI, `PolysomStudio.exe`, `PolysomStudioSetup.exe`)  

Today we are launching **Lucid Machines v1.0.0** (`lucid-machines`), our inaugural public release. Lucid Machines pairs an open-source C/AVX2/CUDA/Triton deep learning framework built around **the Triune architecture by Lucid Machines** (`TriuneTransformer` / `TriuneConfig`) with **Polysom Studio v1.0.0**, a native Windows desktop workbench designed to train, steer, quantize, and serve multi-exit Mixture-of-Experts models within the memory envelope of consumer and single-GPU cloud hardware—establishing the validated substrate for **Junkie**, our inaugural flagship model to be trained.

This inaugural `v1.0.0` release delivers five foundational pillars:
1. **R1 Locked 4-Layer Brand & Compatibility Shims**: Screened `Lucid Machines` (`lucid-machines` company/package/CLI), `Triune` (`TriuneTransformer` core neural architecture), `Junkie` (inaugural flagship model to be trained), and `Polysom Studio` (`polysom` workbench) across PyPI, npm, GitHub, DNS, and web search (`docs/RELEASE_NAME_REPORT.md`), while providing seamless compatibility via the `import triune` shim, `triune-*` presets, and `TRIUNE_*` env vars.
2. **R2 Multi-Directional CentroidSteer 5-Seed Validation vs. Scientific Baselines**: An Exact Orthogonal Direct-Sum ($G = G_{\parallel C_e} \oplus G_{\perp C_e}$) Stiefel-manifold MoE optimizer validated across `N=5` seeds (`[42, 1337, 2026, 777, 1024]`) against **`No-Steer Low-Rank`**, **`Memory-Matched Random Subspace`**, and **`Full-Rank AdamW`**.
3. **R3 5-Tier Empirical Benchmark Suite, Cloud Colab L4 Kit & Native `.NET 10` Polysom Studio v1.0.0**: Strict separation between isolated Triton/CUDA microkernels, end-to-end `90.79M` checkpoint throughput (`383,135 tok/s` peak prefill; `40.8 tok/s` warm decode / `71.3 ms` TTFT on RTX 5070), multi-layer sweeps, lossless n-gram compression, and conformal CALM / speculative decoding, plus 6 editorial Studio themes (`Hypnagogic`, `Phosphene`, `Pareidolia`, `Afterimage`, `Mirage`, `Euphoria`) and the **`Junkie-3B` Colab L4 FP8** 3-stage curriculum kit.
4. **R4 Rock-Solid Security Hardening**: Defense-in-depth across `weights_only=True` checkpoint loading, symlink-resolved path confinement, `shell=False` command execution, proxy-aware bearer authentication, Origin/CSRF/SSRF guards, Ed25519 plugin verification (`trusted=False` default), and Windows Job Object sandboxing.
5. **R5 Project Website & `v1.0.0` Release Packaging**: React/Vite project website (`website/`) and portable Windows Studio distribution (`dist/PolysomStudio-1.0.0-win-x64.zip`).

---

## 1. R2 Deep Dive: Multi-Directional CentroidSteer vs. Scientific Baselines

Mixture-of-Experts (MoE) models face two coupled bottlenecks on single-GPU hardware: **expert optimizer state explosion** (FP32 AdamW requires `8.0 B/p` for first and second moments) and **expert representation under-specialization** when low-rank gradient projections discard activation-aligned subspace directions.

In **Lucid Machines v1.0.0**, `MoE_FFN` and `CentroidSteerOptimizer` combine:
- **Exact Orthogonal Direct-Sum ($G = G_{\parallel C_e} \oplus G_{\perp C_e}$) & Block-Oja EMA Subspaces ($k_c \in [4, 8]$)**: Captures `87.36%` gradient energy in Suite 1 diagnostics (`20.48x` vs. random) and **`92.23%`** in Suite 3 FineWeb-Edu LM (`51.24x` vs. random) while deflating the activation subspace before Krylov iteration on the complementary residual subspace (`<Delta_W_parallel, Delta_W_perp>_F = 0.00e+00`).
- **True Post-SwiGLU Hidden Centroids (`last_hidden_centroids`)**: `MoE_FFN` captures true post-activation hidden centroids (`[E, H]`) for `e[2]` down-projection steering alongside input activation statistics (`last_centroids`, `[E, D]`).
- **Grassmann Parallel Transport, Batched `sCholQR2` & BFloat16 Stochastic Rounding**: Transports preconditioned moments across subspace rotations (`15.13x` activation-space step norm amplification) and maintains $\|Q^T Q - I\|_F < 10^{-4}$ with sliced Householder fallback, independent `steer_step` reset, and unbiased BF16 weight updates.
- **5-Seed Scientific Baseline Validation (`[42, 1337, 2026, 777, 1024]`)**:
  - **Suite 2 (Multi-Domain MoE Specialization, `N=5` seeds, `0.550 B/p`)**: Improves validation loss from **`1.2243 ± 0.0468`** (`No-Steer Low-Rank`) to **`1.1992 ± 0.0444`** (Wilcoxon 1-sided **`p = 0.0312 < 0.05`**, won `5/5` seeds) while capturing **`61.88%` subspace energy** (`0.6188 ± 0.0243` vs. `0.0188 ± 0.0009` random, **`32.9x`**, `p = 6.320e-07`).
  - **Suite 3 (Real FineWeb-Edu Cold-Start LM — 4-Layer Pure-Routed MoE, `N=5` seeds, `0.264 B/p`)**: Achieves **`2.2012 ± 0.0178`** val loss (`9.04 ± 0.16` PPL) vs. **`2.2472 ± 0.0069`** (`9.46 ± 0.07` PPL) **`No-Steer Low-Rank`** (`Delta = -0.0460`, **`paired t-test p = 0.0112 < 0.05`**, won `5/5` seeds) and **`2.2585 ± 0.0131`** (`9.57 ± 0.13` PPL) **`Memory-Matched Random Subspace`** (`Delta = -0.0573`, **`paired t-test p = 0.0006 < 0.001`**, won `5/5` seeds), capturing **`92.23%` gradient energy** at **`0.264 B/p`** (`96.70%` state reduction / `30.3x` compression vs. `8.000 B/p` FP32 AdamW).
  - **Suite 4 (Real FineWeb-Edu 75.8M Checkpoint Continuation from `step_36500`, `N=5` seeds, `0.201 B/p`)**: Achieves **`4.9934 ± 0.0003`** val loss (`147.43 ± 0.04` PPL) vs. **`4.9945 ± 0.0001`** `No-Steer Low-Rank` (**`paired t-test p = 0.0004 < 0.001`**), **`4.9945 ± 0.0001`** `Memory-Matched Random Subspace` (**`paired t-test p = 0.0005 < 0.001`**), and **`4.9954 ± 0.0008`** `Full-Rank AdamW` (**`paired t-test p = 0.0038 < 0.01`**, won `5/5` seeds across all baselines) at **`0.201 B/p`** (`97.49%` reduction; `0.1118 B/p` / `3.78 MB` vs. `135.21 MB` on the `90.79M` Tier-2 model — **98.4% reduction**).

---

## 2. Inaugural `Junkie` Configurations & The Colab Pro L4 200-CU Sweet Spot

We ship four hardware-matched configurations for **Junkie** on `TriuneTransformer` (`cloud/README.md`):
- **`Junkie-Mini`** (`models/configs/junkie-mini.json`): **535,101,975** (`535.10M`) total / **230.81M** expected active (`12L / 1024D / 4E+1S`, `0.1992 B/p`, `3.12 GiB` BF16 static).
- **`Junkie-1B`** (`models/configs/junkie-1b.json`): **1,162,535,483** (`1.16B`) total / **374.99M** expected active (`22L / 1024D / 4E+1S`, `0.1992 B/p`, `4.79 GiB` FP8 / `5.87 GiB` BF16 static).
- **`Junkie-3B` — #1 Colab Pro L4 200-CU Sweet Spot** (`models/configs/junkie-3b.json`): **2,990,355,155** (`2.99B`) total / **395.74M** MoE-trunk Cortex active / **206.73M** average dynamic-exit active (`16L / 1024D / 16E+1S`, `top_k=1`, `0.1992 B/p`). Fits in **`10.10 GiB` FP8 static VRAM** (`< 20 GiB` total resident on Colab L4 24GB, whereas FP32 AdamW requires `33.42 GiB` static and OOMs). Delivers ~3x more tokens per compute unit than A100 (**`~21B–23B` real tokens / `~35B+` AdamW-equivalent tokens** across 200 compute units) using our 3-Stage Decontaminated Curriculum (`50%` FineWeb-Edu + `30%` Code + `20%` FineMath -> High-density textbook/code WSD annealing -> Clean SFT + Verifiable GRPO RL).
- **`Junkie-7B` / `Junkie`** (`models/configs/junkie-7b.json` & `junkie.json`): **7,396,845,163** (`7.40B`) total / **1.95B** expected active (`32L / 2048D / 8E+1S`, `0.3678 B/p`, `34.49 GiB` BF16 static on a single 80GB A100 vs. `82.67 GiB` OOM under FP32 AdamW).

---

## 3. 5-Tier Empirical Benchmark Summary (`CONTRIBUTING.md §1`)

### Tier 1: Isolated Kernel Microbenchmarks
- **`VectorisedGLA` Recurrent Scan**: Constant **`48.0 KiB`** state cache at $H=6, L=16,384$, achieving **`10.07x` prefill** and **`7.31x` decode** speedup over Flash/SDPA attention, while `Gated DeltaNet` achieves **100.0% standard and 100.0% overwrite MQAR associative recall**.
- **Production Sparse `MoE_FFN(use_sparse_eval=True)`**: Zero-sync single-token (`N=1, d=384`) stacked BMM achieves **`1.648x` speedup** (`0.0` max diff vs. dense), scaling to **`7.518x` speedup** at `d=384, E=8, k=2, N=4096` (`4.774x` top-1 and `5.450x` top-2 at `d=1536`).
- **Chunked Vocabulary Cross-Entropy (`chunked_cross_entropy`)**: Saves **`98.5%` peak transient VRAM** vs. standard PyTorch and **`94.1%` vs. genuine Unsloth (`2026.10.10`)** at `V=32,000, N=8,192`.
- **Fused `FastSwiGLU`**: **`1.95x` speedup** over eager PyTorch.

### Tier 2: End-to-End 90.79M Baseline Model (`64.21M` Active) & Layer Streaming
- **Full-Checkpoint Prefill & Decode (`step_40555`)**: Sustains **`383,135 tok/s`** peak batched prefill (`B=32, T=1024`), **`200,463 tok/s`** (`B=1, T=2048`), **`242,029 tok/s`** (`B=1, T=8192`), **`297,753 tok/s`** (`B=4, L=16,384`), **`157.0 tok/s`** sequential tensor decode (`6.363 ms/tok`), and **`40.8 tok/s` warm decode / `71.3 ms` TTFT** in streaming runtime inference on RTX 5070.
- **LoRA Fine-Tuning**: Sustains **`25,973 tok/s`**, outperforming genuine installed Unsloth (`24,344 tok/s`) by **`+6.7%`**.
- **Depth-Invariant `LayerStreamingEngine`**: Cuts `102M` peak VRAM by **`76.43%`** (`1,033.6 MB`), `271M` by **`83.76%`** (`1,498.1 MB`, **`1.97x` faster** than Windows WDDM paging spill), and trains **`695M` (`2,115.7 MB`)** and **`1.52B + CentroidSteer` (`3,132.6 MB`)** on an `8 GB` consumer GPU.
- **4-Tier Quantization (`NVFP4` & Hadamard W4A16)**: Compresses `96` `NVFP4Linear` blocks (`78.26M` params) by **`3.20x`** (`46.65 MB`), with Hadamard-rotated W4A16 + protected layers reducing 4-bit perplexity to **`119.106`**.

### Tier 3: Multi-Layer Sweeps vs. Legacy Diagnostics
- Multi-layer long-context sweeps up to $L=16,384$ maintain a **`64.0 KiB`** state cache (**99.8% reduction** vs. Softmax SDPA) with **`4.02x` prefill** and **`4.13x` decode** speedups.
- Production `MoE_FFN` (`1.648x–7.518x`) is explicitly separated from the legacy diagnostic `MoERoutingBlock` (`0.47x`).

### Tier 4: Lossless Tokenizer Compression vs. Model Computation
- `DynamicTokenizer` achieves **`2.65x` aggregate lossless text compression** (`100%` string round-trip reconstruction) across held-out Python, Math Prose, JSON, and Log splits.
- `embed_compressed(compress_mode="expand")` achieves **`0.0` max logit difference** against uncompressed base token IDs, while `order_pool` applies orthogonal RoPE phase rotation to preserve intra-span token order at compressed sequence lengths.

### Tier 5: Self-Speculative Decoding & Conformal CALM Calibration
- **Monotonic Multi-Exit Perplexity**: On the KL-rescued `90.79M` checkpoint (`step_40555`), validation perplexity improves monotonically across **Reflex L2 (`111.80`) -> Limbic L5 (`111.18`) -> Cortex L8 (`110.92`)**, with a **`4.31x` Reflex vs. Cortex prefill speedup** (`741,070 tok/s` vs. `172,099 tok/s` at `B=4, T=512`).
- **Self-Speculative Decoding**: Greedy `T=0.0` self-speculation achieves **`91.62 tok/s`** (`51.28%` acceptance) with **100% exact Cortex token parity**; stochastic `T=0.7` Leviathan et al. rejection sampling achieves **`90.74 tok/s`** (`63.64%` acceptance, Pearson $\chi^2$ verified).
- **Held-Out Conformal CALM (`benchmarks/calm_calibration.py`)**: Across `N=87,141` calibration and `N=85,503` evaluation tokens, softmax-margin gating reaches `AUROC = 0.8159` (Reflex) and `0.8319` (Limbic), certifying **`95.73%` Reflex agreement** at $\tau=0.20$ (`31.34%` coverage, **`20.35%` transformer FLOPs saved** at `+0.14%` PPL delta).

---

## 4. R4 Security Hardening & Polysom Studio (`v1.0.0`)

Lucid Machines v1.0.0 ships with comprehensive security hardening verified by `tests/test_security_regression.py` and the Tiers 1–4 E2E suite:
- **Safe Deserialization**: `torch.load(..., weights_only=True)` enforced across fine-tuning, checkpoint rescue, and residual verification tools.
- **Symlink-Resolved Path Confinement & `shell=False`**: All `/v1/models/*`, `/v1/checkpoints/*`, `/v1/datasets/*`, `/v1/finetune/*`, `/v1/workspace/*`, and DAG file nodes enforce `Path.resolve(strict=False).is_relative_to(...)` and reject `..`, UNC, and symlink traversal; all subprocess invocations enforce `shell=False`.
- **Proxy-Aware Bearer Auth, CSRF & SSRF Guards**: Requests carrying `Forwarded`, `X-Forwarded-For`, or `X-Real-IP` headers (or non-loopback connections) require a valid `LUCID_MACHINES_API_TOKEN` bearer token; state-changing HTTP routes and `/ws/telemetry` validate `Origin`/`Host`; remote dataset and module URLs pass `validate_http_url` SSRF checks.
- **Cryptographic Plugin Trust & Credential Hygiene**: External workspace plugins default to `trusted=False` (requiring Ed25519 signature verification), vLLM defaults to `trust_remote_code=False`, Gemini keys use the `x-goog-api-key` HTTP header exclusively, and user credentials reside in the OS keyring (`Polysom Studio` / `Triune Studio`).

### Get Started Today

```bash
pip install -e ".[all]"
lucid-machines doctor
polysom --port 8000
```

Or launch the self-contained Windows desktop application and setup wizard in `studio\publish\PolysomStudioSetup.exe`.
