# Release Notes — Lucid Machines `v1.0.0` (Inaugural Public Release: Triune Architecture, Junkie & Polysom Studio)

**Release Version:** `1.0.0` (Inaugural Public Release)  
**Company / Python Package:** **Lucid Machines** (`lucid-machines` distribution, `lucid_machines` Python SDK, `lucid-machines` CLI)  
**Core Neural Architecture:** **Triune** (`TriuneTransformer` / `TriuneModel` / `TriuneConfig`, HF `"model_type": "triune"`, size templates `models/configs/triune-*.json`)  
**Inaugural Flagship Models To Train:** **Junkie** (`models/configs/junkie-mini.json`, `junkie-1b.json`, `junkie-3b.json`, `junkie-7b.json`, and `junkie.json` — built on the Triune architecture)  
**Desktop Workbench:** **Polysom Studio v1.0.0** (`polysom` CLI, `PolysomStudio.exe`, `PolysomStudioSetup.exe`)  
**Branch / Baseline:** `release/lucid-machines-v1.0.0`  
**Release Date:** November 2026  

---

## Overview

**Lucid Machines v1.0.0** is our inaugural public release of the security-hardened, empirically verified open-source deep learning platform (`lucid-machines`) built around **the Triune architecture by Lucid Machines** (`TriuneTransformer` / `TriuneModel` / `TriuneConfig`), the validated hardware-matched configurations and 3-stage training curriculum for **Junkie** (our inaugural flagship model to be trained), and the native desktop workbench **Polysom Studio v1.0.0**.

This inaugural release delivers five core engineering pillars (**R1–R5**):
1. **R1 — Locked 4-Layer Naming Boundary & Compatibility Shims** (`Lucid Machines` company / `Triune` core neural architecture / `Junkie` inaugural flagship model / `Polysom Studio` desktop workbench)
2. **R2 — Multi-Directional CentroidSteer Optimizer & Controlled 5-Seed Scientific Baseline Validation** (compared against `No-Steer Low-Rank`, `Memory-Matched Random Subspace`, and `Full-Rank AdamW`)
3. **R3 — Core Package, Native C/AVX2/CUDA Engine, `.NET 10` Polysom Studio v1.0.0 Build, Cloud Colab L4 Kit & Asset Preservation Guardrails**
4. **R4 — Defense-in-Depth Security Hardening & Regression Suite**
5. **R5 — Project Website (`website/`), Local Launch Kit (`launch_kit/`) & `v1.0.0` Distribution Packaging**

---

## 1. Locked 4-Layer Naming Boundary & Compatibility Contract (`R1`)

- **Conflict-Screened 4-Layer Brand (`docs/RELEASE_NAME_REPORT.md`)**: Locked **Lucid Machines** (`lucid-machines` / `lucid_machines` company, package, CLI, and release brand), **Triune** (`TriuneTransformer` / `TriuneModel` / `TriuneConfig` core neural architecture), **Junkie** (inaugural flagship model to be trained), and **Polysom Studio** (`polysom` / `polysom-studio` workbench) via automated registry queries across PyPI, npm, GitHub, DNS, and web search alongside an explicit non-legal/trademark disclaimer (`docs/RELEASE_NAME_REPORT.md`).
- **Unified Naming Surface (`v1.0.0`)**:
  - Python distribution & import package: `lucid-machines` / `lucid_machines` (`pyproject.toml` `version = "1.0.0"`).
  - CLI entrypoints: `lucid-machines` (`studio`, `serve`, `train`, `finetune`, `chat`, `doctor`, `benchmark`, `plan-memory`) and `polysom` (Polysom Studio launcher).
  - Model configs: 12 Triune architecture size templates in `models/configs/triune-*.json` plus 4 inaugural **Junkie** training configurations (`models/configs/junkie-mini.json`, `junkie-1b.json`, `junkie-3b.json`, `junkie-7b.json` / `junkie.json`).
  - Native C/AVX2/CUDA shared library: `lucid_machines_core.dll` / `liblucid_machines_core.so` / `liblucid_machines_core.dylib` (`1.0.0-native-avx2-cuda`, with `triune_core.*` fallback).
  - Windows Desktop App & Installer: `PolysomStudio.exe` and `PolysomStudioSetup.exe` (`dist/PolysomStudio-1.0.0-win-x64.zip`).
  - Environment variable prefix: `LUCID_MACHINES_*` (`LUCID_MACHINES_API_TOKEN`, `LUCID_MACHINES_REQUIRE_AUTH`, `LUCID_MACHINES_PLUGIN_TRUSTED_KEYS`, `LUCID_MACHINES_ALLOW_PIP_INSTALL`, `LUCID_MACHINES_ALLOW_REMOTE_SANDBOX`, `LUCID_MACHINES_ALLOW_AGENT_WRITES`).
- **Backward-Compatibility Shims**:
  - `import triune` delegates transparently to `lucid_machines`, preserving `TriuneTransformer`, `TriuneModel`, `TriuneConfig`, `TriuneForCausalLM`, and `NativeTriuneEngine`.
  - `_LEGACY_PRESET_ALIASES` maps shorthand aliases (`"triune-micro"`, `"triune-1.5b"`, `"triune-32b"`, etc.) cleanly to canonical `triune-*` architecture templates.
  - All `LUCID_MACHINES_*` environment lookups fall back to `TRIUNE_*`, OS keyring lookups check `"Polysom Studio"` with `"Triune Studio"` fallback, and `triune_tokenizer.json` resolves natively.

---

## 2. Multi-Directional CentroidSteer & 5-Seed Scientific Baseline Validation (`R2`)

### Exact Orthogonal Direct-Sum & Multi-Directional Subspace Steering
`CentroidSteerOptimizer` (`lucid_machines/optim/centroid.py`, `lucid_machines/ops/centroid_steer.py`) compresses Mixture-of-Experts optimizer states to `< 0.6 B/p` (`0.201–0.550 B/p` across validation suites; `0.1992–0.3678 B/p` across `Junkie` configs) while preserving high-energy activation-aligned gradient updates:
1. **Exact Orthogonal Direct-Sum Decomposition ($G = G_{\parallel C_e} \oplus G_{\perp C_e}$)**: Projects expert gradients onto a multi-directional ($k_c \in [4, 8]$) Block-Oja EMA activation subspace (capturing `87.36%` gradient energy in Suite 1 diagnostics and `92.23%` in Suite 3 FineWeb-Edu LM, with `<Delta_W_parallel, Delta_W_perp>_F = 0.00e+00`) and deflates the activation component prior to Krylov iteration on the complementary residual subspace.
2. **True Post-SwiGLU Hidden Centroids (`last_hidden_centroids`)**: `MoE_FFN` captures true post-activation hidden centroids (`[E, H]`) for `e[2]` down-projection steering alongside input activation statistics (`last_centroids`, `[E, D]`).
3. **Grassmann Parallel Transport, Batched `sCholQR2` & BFloat16 Stochastic Rounding**: Transports preconditioned moments across subspace rotations without resets (`15.13x` activation-space step amplification over unsteered low-rank cores), maintains strict Stiefel manifold orthonormality ($\|Q^T Q - I\|_F < 10^{-4}$, Stiefel isometry error `9.10e-07`) with sliced Householder fallback and BFloat16 `cholesky_ex` autocast safety, independent `steer_step` reset, and unbiased BF16 stochastic rounding.

### Controlled 5-Seed Validation vs. Scientific Baselines (`[42, 1337, 2026, 777, 1024]`)
Across `N=5` matched random seeds on an NVIDIA RTX 5070 Laptop GPU (`benchmarks/CENTROIDSTEER_UPGRADE_VALIDATION_REPORT.md`), **CentroidSteer** is evaluated against three scientific baselines:
- **`No-Steer Low-Rank` (`nosteer`)**: Pure low-rank Krylov/GaLore projection without activation subspace steering.
- **`Memory-Matched Random Subspace` (`random_same_mem`)**: Identical memory footprint (`r + k_c` columns) using random orthonormal border directions instead of Block-Oja activation centroids.
- **`Full-Rank AdamW` (`adamw`)**: Uncompressed `8.000 B/p` FP32 (`4.000 B/p` BF16) first/second moment baseline.
- **Measured Results (`N=5` Seeds)**:
  - **Suite 1 (Mechanistic Diagnostics)**: Direct-sum relative error `4.50e-07` (`W_1`) / `7.90e-07` (`W_2`), `<Delta_W_parallel, Delta_W_perp>_F = 0.00e+00`, Block-Oja gradient energy capture **`87.36%` vs. `4.27%` random (`20.48x`)**, and **`15.13x`** activation-space step norm amplification (`1.5446x` vs. `0.1021x` unsteered).
  - **Suite 2 (Multi-Domain MoE Specialization, `steps=120`, `rank=8`, `k_c=4`, `0.550 B/p`)**: Validation loss **`1.1992 ± 0.0444`** (PPL **`3.32 ± 0.15`**) vs. **`1.2243 ± 0.0468`** `nosteer` (Wilcoxon 1-sided **`p = 0.0312 < 0.05`**, won `5/5` seeds) and **`1.2044 ± 0.0446`** `random_same_mem`, with **`61.88%` subspace energy capture** (`0.6188 ± 0.0243` vs. `0.0188 ± 0.0009` random, **`32.9x`**, `p = 6.320e-07`) at `93.12%` state reduction vs. FP32 AdamW.
  - **Suite 3 (Real FineWeb-Edu Cold-Start LM Pretraining — 4-Layer Pure-Routed MoE, `steps=300`, `rank=4`, `k_c=8`, `0.264 B/p`)**: Validation loss **`2.2012 ± 0.0178`** (PPL **`9.04 ± 0.16`**) vs. **`2.2472 ± 0.0069`** (PPL `9.46 ± 0.07`) `nosteer` (`Delta = -0.0460`, **`paired t-test p = 0.0112 < 0.05`**, Wilcoxon 1-sided `p = 0.0312`, won `5/5` seeds) and **`2.2585 ± 0.0131`** (PPL `9.57 ± 0.13`) `random_same_mem` (`Delta = -0.0573`, **`paired t-test p = 0.0006 < 0.001`**, Wilcoxon 1-sided `p = 0.0312`, won `5/5` seeds), capturing **`92.23%` gradient energy** (`51.24x` vs. random, `p = 2.126e-08`) at **`0.264 B/p`** (`96.70%` reduction / `30.3x` compression vs. FP32 AdamW).
  - **Suite 4 (Real FineWeb-Edu 75.8M Checkpoint Continuation from `step_36500`, `steps=100`, `rank=8`, `k_c=8`, `0.201 B/p`)**: Validation loss **`4.9934 ± 0.0003`** (PPL **`147.43 ± 0.04`**) vs. **`4.9945 ± 0.0001`** `nosteer` (**`paired t-test p = 0.0004 < 0.001`**), **`4.9945 ± 0.0001`** `random_same_mem` (**`paired t-test p = 0.0005 < 0.001`**), and **`4.9954 ± 0.0008`** `adamw` (**`paired t-test p = 0.0038 < 0.01`**, won `5/5` seeds across all three baselines) with **`70.54%` subspace energy capture** (`57.82x` vs. random, `p = 1.478e-09`) at **`0.201 B/p`** (`97.49%` reduction / `39.8x` compression vs. FP32 AdamW).

---

## 3. Inaugural `Junkie` Configurations & Colab Pro L4 Sweet Spot (`cloud/`)

All parameter counts are verified on `torch.device("meta")` (`vocab_size=32000`, `EXPERT_HIDDEN_MULTIPLIER=6`, `shared_expert=true`):

| Model Config | Total Params | Active Params (`top_k=1`: Reflex / Limbic / Cortex / Expected / Dynamic 50-30-20) | Expert Tier (`%`) | CentroidSteer Expert State | Static VRAM (FP8 / BF16) | FP32 AdamW Static VRAM |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| `junkie-mini.json` (`12L / 1024D / 4E+1S`) | **535,101,975** (`535.10M`) | `161.30M` / `252.59M` / `280.65M` (**230.81M** / **212.56M**) | `251.66M` (`47.0%`) | **`0.1992 B/p`** (`rank=64`) | **`2.62 GiB`** / **`3.12 GiB`** | `5.98 GiB` |
| `junkie-1b.json` (`22L / 1024D / 4E+1S`) | **1,162,535,483** (`1.16B`) | `174.94M` / `387.95M` / `568.15M` (**374.99M** / **317.48M**) | `704.64M` (`60.6%`) | **`0.1992 B/p`** (`rank=64`) | **`4.79 GiB`** / **`5.87 GiB`** | `12.99 GiB` |
| **`junkie-3b.json` (`16L / 1024D / 16E+1S`)** | **2,990,355,155** (`2.99B`) | `106.75M` / `198.07M` / `469.69M` (**395.74M** MoE-trunk / **206.73M** dynamic) | `2,617.25M` (`87.5%`) | **`0.1992 B/p`** (`rank=64`) | **`10.10 GiB` FP8** / **`12.89 GiB` BF16** (`< 20 GiB` resident on L4 24GB!) | **`33.42 GiB` (OOM on L4 24GB)** |
| `junkie-7b.json` & `junkie.json` (`32L / 2048D / 8E+1S`) | **7,396,845,163** (`7.40B`) | `1.17B` / `2.02B` / `2.68B` (**1.95B** / **1.73B**) | `5,234.49M` (`70.8%`) | **`0.3678 B/p`** (`rank=256`) | **`27.60 GiB`** / **`34.49 GiB`** | **`82.67 GiB` (OOM on 80GB A100)** |

- **Colab Pro 200-CU Sweet Spot (`cloud/lucid_machines_colab_runner.ipynb`)**: Running **`Junkie-3B` on Colab L4 (`--use_fp8 --depth_mode dynamic`)** delivers ~3x more tokens per compute unit than A100 (**`~21B–23B` real tokens / `~35B+` AdamW-equivalent tokens** across 200 compute units) inside `< 20 GiB` resident VRAM using the 3-Stage Decontaminated Curriculum (`50%` FineWeb-Edu + `30%` Code + `20%` FineMath -> High-density textbook/code WSD annealing -> Clean SFT + Verifiable GRPO RL).

---

## 4. 5-Tier Empirical Benchmarks, Native Kernels & Polysom Studio v1.0.0 (`R3`)

All performance metrics are verified and reported under the strict **5-Tier Benchmark Taxonomy** (`CONTRIBUTING.md §1`):

| Tier | Category | Key Verified Metrics (RTX 5070 Laptop GPU) |
| :-: | :--- | :--- |
| **Tier 1** | **Isolated Kernel Microbenchmarks** | `VectorisedGLA` (`H=6, L=16384`): **`48.0 KiB`** state cache, **`10.07x`** prefill / **`7.31x`** decode vs. SDPA; `Gated DeltaNet`: **100.0% standard & overwrite MQAR recall**; Sparse `MoE_FFN(use_sparse_eval=True)`: **`1.648x`** zero-sync at `N=1` (`0.0` max diff) and **`7.518x`** at `d=384, E=8, k=2, N=4096` (`4.774x` top-1 / `5.450x` top-2 at `d=1536`); `chunked_cross_entropy`: **`98.5%` transient VRAM saved** vs. PyTorch (**`94.1%`** vs. Unsloth `2026.10.10`) at `V=32,000, N=8,192`; `FastSwiGLU`: **`1.95x`** vs. PyTorch. |
| **Tier 2** | **End-to-End 90.79M Model (`64.21M` Active)** | Rescued `step_40555`: **`383,135 tok/s`** peak prefill (`B=32, T=1024`), **`200,463 tok/s`** (`B=1, T=2048`), **`242,029 tok/s`** (`B=1, T=8192`), **`297,753 tok/s`** (`B=4, L=16384`), **`157.0 tok/s`** sequential tensor decode (`6.363 ms/tok`), **`40.8 tok/s` warm decode / `71.3 ms` TTFT** streaming inference, **`25,973 tok/s` LoRA fine-tuning** (**`+6.7%`** over Unsloth `24,344 tok/s`); `LayerStreamingEngine`: cuts `102M` VRAM by **`76.43%`** (`1,033.6 MB`), `271M` by **`83.76%`** (`1,498.1 MB`), trains **`1.52B + CentroidSteer` in `3,132.6 MB` VRAM**; `NVFP4`: **`3.20x`** compression (`46.65 MB` across 96 blocks), Hadamard W4A16 4-bit PPL **`119.106`**. |
| **Tier 3** | **Multi-Layer Wrapper vs. Legacy Diagnostics** | Multi-layer long-context sweep (`L=16384`): **`64.0 KiB`** cache (**99.8% reduction** vs. SDPA), **`4.02x`** prefill / **`4.13x`** decode; production `MoE_FFN` (`1.648x–7.518x`) explicitly separated from legacy diagnostic `MoERoutingBlock` (`0.47x`). |
| **Tier 4** | **Tokenizer Compression vs. Model Compute** | Held-out lossless text compression: **`2.65x` aggregate** (`100%` reconstruction); compute-equivalent `compress_mode="expand"`: **`0.0` max logit diff** vs. base IDs; `order_pool` orthogonal RoPE phase rotation. |
| **Tier 5** | **Speculative & Conformal CALM Exits** | Monotonic exit PPL (`111.80 -> 111.18 -> 110.92`), **`4.31x` Reflex prefill speedup** (`741,070` vs. `172,099 tok/s`); `T=0.0` greedy self-speculative decode: **`91.62 tok/s`** (`51.28%` acceptance, 100% exact Cortex parity); `T=0.7` Leviathan rejection sampling: **`90.74 tok/s`** (`63.64%` acceptance, Pearson $\chi^2$ verified); Conformal CALM (`N=85,503` eval tokens, $\tau=0.20$): **`95.73%` Reflex agreement** (`31.34%` coverage, **`20.35%` FLOPs saved**). |

### Polysom Studio v1.0.0 Themes, Workspace IDE, Agent Loops & Preservation Guardrails
- **6 Editorial UI Themes (`studio/src/`)**: `Hypnagogic` (default dark), `Phosphene` (pitch black + phosphor), `Pareidolia` (cold slate + silver/ice), `Afterimage` (crimson/amber retinal burn), `Mirage` (clean daylight light theme), and `Euphoria` (warm, elated, luminous editorial palette).
- **Project Workspace IDE & Multi-Mode Agent Loops**: Full confined Workspace IDE (`/v1/workspace/*` file tree, tabbed code editor, unified diff viewer, regex search, git status/diff, allowlisted `shell=False` terminal) and SSE-streamed Agentic Loops (`ReAct`, `PEV`, `Multi-Agent Explorer -> Implementer -> Verifier`, and `Autonomous Training Supervisor` with HITL diff approval gates, stuck-loop detection, and context compaction).
- **Native Windows Studio (`studio/windows/`)**: Self-contained `.NET 10` WPF + `WebView2` desktop app (`PolysomStudio.exe`) and installer (`PolysomStudioSetup.exe`) with `Version=1.0.0`, Lucid Eye `polysom.ico`, HKCU `Uninstall` registry registration, `--verify`, `--yes --no-launch`, and `--uninstall` CLI flags, Desktop/Start Menu shortcuts, and single-archive `dist/` packaging with `.sha256` digests.
- **Preservation Guardrails**: All pre-existing `checkpoints/*.pt` (`session_state.json` at `step: 40555`, `triune_studio_step_36500_rescued.pt`), `data/*.jsonl`, `exports/*`, `papers/*`, and `triune_tokenizer.json` are preserved intact; `WORKSPACE_RULES.md` remains untracked.

---

## 5. Defense-in-Depth Security Hardening (`R4`)

- **Checkpoint Deserialization (`weights_only=True`)**: Enforced across `finetune.py`, `rescue_checkpoint.py`, `verify_triangular_qr_residuals.py`, and API loaders with CPU-first `map_location="cpu"`.
- **Strict Path Confinement & `shell=False`**: All `/v1/models/load`, `/v1/checkpoints/load`, `/v1/datasets/*`, `/v1/models/export`, `/v1/finetune/*`, `/v1/workspace/*`, and DAG file nodes enforce `Path.resolve(strict=False).is_relative_to(...)` and reject `..`, UNC, and symlink escape; all subprocess invocations enforce `shell=False`.
- **Proxy-Aware Bearer Auth, CSRF & Security Headers**: Requests with `Forwarded`, `X-Forwarded-For`, or `X-Real-IP` headers (or non-loopback hosts) require a valid `LUCID_MACHINES_API_TOKEN` / `TRIUNE_API_TOKEN` bearer token; state-changing HTTP routes and `/ws/telemetry` validate `Origin`/`Host`; responses include `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, and `Content-Security-Policy`.
- **Plugin Cryptography, Sandbox & Provider Hygiene**: External workspace plugins default to `trusted=False` (Ed25519 signature required); `PythonSandbox` enforces strict memory/job limits and gates `allow_writes`; vLLM bridge defaults to `trust_remote_code=False`; Google Gemini provider sends API keys exclusively via the `x-goog-api-key` HTTP header.

---

## 6. Release Artifacts & Verification Checksums (`R5`)

| Artifact Path | Description |
| :--- | :--- |
| `dist/lucid_machines-1.0.0-py3-none-any.whl` | Python Wheel Package (`python -m build --wheel`) |
| `dist/lucid_machines-1.0.0.tar.gz` | Python Source Distribution (`python -m build --sdist`) |
| `dist/PolysomStudio-1.0.0-win-x64.zip` (and `.sha256`) | Self-Contained Windows x64 Desktop App & Installer Bundle |
| `dist/SHA256SUMS.txt` | Consolidated SHA-256 Checksum Manifest |
| `cloud/lucid_machines_colab_runner.ipynb` | Colab Pro Runner (`Junkie-3B` L4 FP8 200-CU Sweet Spot + 3-Stage Curriculum) |
| `website/` | React/Vite Project Website (`index.html`, `src/*`, `public/*`, `dist/*`) |
| `launch_kit/` | Local Launch & Advertising Kit (`README.md`, `HERO_MESSAGING.md`, `LAUNCH_BLOG.md`, `SOCIAL_POSTS.md`, `AD_COPY_VARIANTS.md`, SVG cards) |
| `docs/RELEASE_NAME_REPORT.md` & `docs/WORKFLOW_2.md` | R1 Conflict-Check Matrix, Non-Legal Disclaimer & Workflow 2 Compliance Guide |
| `benchmarks/CENTROIDSTEER_UPGRADE_VALIDATION_REPORT.md` | Multi-Directional CentroidSteer 5-Seed Scientific Baseline Validation Report |
