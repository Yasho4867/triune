# Lucid Machines (`v1.0.0`) — Triune Architecture, Junkie & Polysom Studio

**Website:** [https://lucidmachines.co](https://lucidmachines.co) · **GitHub:** [https://github.com/lucid-machines/triune](https://github.com/lucid-machines/triune)

**Lucid Machines** (`lucid-machines` / `lucid_machines`) is a modular, hardware-aware deep learning ecosystem built around **the Triune architecture by Lucid Machines** (`TriuneTransformer` / `TriuneModel` / `TriuneConfig` 3-tier multi-exit Mixture-of-Experts architecture, plus `UniversalTransformer`), **Polysom Studio v1.0.0** (a self-contained native desktop workbench), and **Junkie** (`models/configs/junkie.json`, `junkie-7b.json`, `junkie-3b.json`, `junkie-1b.json`, `junkie-mini.json`) — Lucid Machines' inaugural flagship model configurations to be trained on the Triune architecture. All existing checkpoints (`checkpoints/*.pt`), datasets (`data/*.jsonl`), exports (`exports/*`), papers (`papers/*`), `triune-*` architecture templates (`models/configs/triune-*.json`), and `triune_tokenizer.json` are preserved natively.

---

## Key Capabilities & Pillars

- **Gated Linear Attention (`VectorisedGLA`) & Gated DeltaNet**: $O(N)$ linear-time prefill and strictly constant $O(1)$ recurrent state memory (`48.0 KiB` isolated kernel at $H=6$ with `10.07x` prefill / `7.31x` decode vs. SDPA; `64.0 KiB` across multi-layer sweeps up to $L=16,384$, **99.8% cache reduction** vs. Softmax SDPA, `4.02x` prefill / `4.13x` decode speedup at $L=16,384$) with fused chunked Triton scan (`FusedRecurrentScan`) and Gated DeltaNet achieving **100.0% standard and 100.0% overwrite MQAR associative recall**.
- **Sparse Mixture of Experts (`MoE_FFN`)**: Default sparse scatter-gather evaluation (`use_sparse_eval=True`) with zero-sync `N_tok=1` stacked BMM (**`1.648x` speedup** at `d=384, N=1`, **0.0 max diff** vs. dense) and sorted multi-token sparse slices (**`7.518x` speedup** at checkpoint width `d=384, E=8, k=2, N=4096`; **4.774x** top-1 and **5.450x** top-2 at `d=1536`), alongside always-active shared experts, contrastive activation statistics (`last_centroids`, `[E, D]`), and true post-SwiGLU hidden centroids (`last_hidden_centroids`, `[E, H]`).
- **3-Tier Hierarchical Exits, Conformal CALM Calibration & Self-Speculative Decoding**: Intermediate exits (**Reflex L2**, **Limbic L5**) and terminal **Cortex L8** on the KL-rescued `90.79M` checkpoint (`checkpoints/triune_studio_step_36500_rescued.pt`, `step_40555`, strictly monotonic validation perplexity **`111.80 -> 111.18 -> 110.92` PPL**), **`4.31x` Reflex vs. Cortex prefill speedup** (`2.764 ms / 741,070 tok/s` vs `11.90 ms / 172,099 tok/s` at `B=4, T=512`), `T=0.0` **exact-match greedy self-speculative decoding** (**`91.62 tok/s`**, **`51.28%` acceptance**), Pearson $\chi^2$-verified Leviathan et al. **rejection sampling** at `T=0.7` (**`90.74 tok/s`**, **`63.64%` acceptance**), and streaming runtime inference at **`40.8 tok/s` warm decode / `71.3 ms` TTFT** on RTX 5070. Held-out conformal CALM calibration over `N=87,141` calibration / `N=85,503` FineWeb-Edu evaluation tokens (`benchmarks/calm_calibration.py`) yields `AUROC = 0.8159` (Reflex) and `0.8319` (Limbic), certifying `95.73%` Reflex agreement at $\tau=0.20$ (`31.34%` coverage, **`20.35%` transformer FLOPs saved** at `+0.14%` PPL delta).
- **Lossless Dynamic N-Gram Compression (`compress_mode`)**: Explicit separation between lossless text compression (**2.65x aggregate** across held-out Python, Math Prose, JSON, and Repetitive Log splits with `100%` round-trip reconstruction) and transformer computation (`compress_mode="expand"` achieves **0.0 max logit diff** vs. base tokenization; `compress_mode="order_pool"` preserves intra-span token ordering via orthogonal RoPE phase rotation; `compress_mode="mean_pool"` and `extend_vocabulary_for_tokenizer()` enable compressed sequence training).
- **4-Tier Weight Quantization (BF16 / FP8 / NVFP4 / Hadamard W4A16) & CUDA Graph Decode**: Packed 4-bit E2M1 + 16-element E4M3 block scales via `convert_to_nvfp4` across **`96` `NVFP4Linear` blocks** (`78.26M` params) on the `90.79M` checkpoint (`46.65 MB` / `3.20x` unrotated NVFP4 at `126.887` PPL; **`78.88 MB` / `1.89x` Hadamard-rotated W4A16 + protected layers** reducing 4-bit PPL to **`119.106`**; `74.63 MB` / `2.00x` FP8 E4M3 at `101.560` PPL; `149.26 MB` BF16 at `97.504` PPL), direct GPU GGUF-to-NVFP4 streaming, fused RMSNorm, CUDA Graph decode capture, and `ParameterStager(use_fp4=True)` (`7 NVFP4StagingBuffer` transfers = `5.94 MB`).
- **3-Tier Partitioned Optimizer (`CentroidSteerOptimizer`)**:
  - **Tier 1 (`Muon`)**: Batched 5th-order Newton-Schulz polar orthogonalization with fused `baddbmm` polynomial evaluation and multi-tensor `_foreach` updates for 2D hidden projections ($2.0\text{ bytes/param}$).
  - **Tier 2 (`CentroidSteer`)**: Exact Orthogonal Direct-Sum ($G = G_{\parallel C_e} \oplus G_{\perp C_e}$) and multi-directional ($k_c \in [4, 8]$) activation-subspace steering on the Universal Triangular Core with Block-Oja EMA subspaces (`87.36%` gradient energy capture in Suite 1, `92.23%` in Suite 3 FineWeb-Edu LM), Grassmann parallel transport (`15.13x` activation-space step amplification), centroid-deflated Krylov, batched `sCholQR2` ($||Q^T Q - I||_F < 10^{-4}$), sliced Householder fallback, true post-SwiGLU hidden centroid capture (`last_hidden_centroids` `[E, H]` for `e[2]` down-projections), independent `steer_step` reset, non-refresh moment zeroing, and Triton-fused upper-triangular AdamW. Validated across `N=5` seeds (`[42, 1337, 2026, 777, 1024]`) against scientific baselines (**`No-Steer Low-Rank`**, **`Memory-Matched Random Subspace`**, and **`Full-Rank AdamW`**) with statistically significant FineWeb-Edu validation loss & perplexity improvements (`Suite 3` cold-start LM at `0.264 B/p`: `2.2012` vs `2.2472` `nosteer`, `p = 0.0112`, and `2.2585` `random_same_mem`, `p = 0.0006`; `Suite 4` 75.8M continuation at `0.201 B/p`: `4.9934` vs `4.9945` `nosteer`, `p = 0.0004`, `4.9945` `random_same_mem`, `p = 0.0005`, and `4.9954` `adamw`, `p = 0.0038`; `0.1118 B/p` / `3.78 MB` vs `135.21 MB` on 90.79M Tier-2, **98.4% reduction**; `0.1992 B/p` on `Junkie-Mini` / `Junkie-1B` / `Junkie-3B`; `0.3678 B/p` on `Junkie-7B`).
  - **Tier 3 (`AdamW` / `8-bit AdamW`)**: Fused adaptive moments for 1D vectors, norms, router gates, and vocabulary embeddings.
- **Activation-Checkpointed Layer Streaming & Hugging Face Bridge (`LayerStreamingEngine`, `hf_compat`)**: Depth-invariant host-to-GPU layer virtualization with `activation_checkpointing=True`, `_is_bwd_staged` reentry guards, and non-parameter buffer copy-back cuts `102M` peak allocated VRAM by **`76.43%`** (`1,033.6 MB`), cuts `271M` VRAM by **`83.76%`** (`1,498.1 MB`, running **`1.97x` faster** than Windows WDDM paging spill), and trains **`695M` (`2,115.7 MB`)** and **`1.52B + CentroidSteer` (`3,132.6 MB`)** inside an `8 GB` consumer GPU.
- **Native C/AVX2/CUDA/Triton Operator Engine (`csrc/` & `lucid_machines.ops`)**: Zero-copy `ctypes` / P/Invoke C ABI (`lucid_machines_core.dll` / `liblucid_machines_core.so`, with `triune_core.*` fallback) and Triton/Tensor Core kernels for RMSNorm, RoPE, GLA recurrent decode, chunked vocabulary-projection Cross-Entropy (**`98.5%` peak transient VRAM saved vs. PyTorch, `94.1%` vs. genuine Unsloth at `V=32,000, N=8,192`**), SwiGLU (**`1.95x` vs. PyTorch**), and Neuron Scope activation telemetry.
- **Polysom Studio (`PolysomStudio.exe` / `PolysomStudioSetup.exe`) & R4 Security Architecture**: Self-contained native Windows desktop application (`.NET 10` + `WebView2`), WPF installer, 6 editorial themes (`Hypnagogic`, `Phosphene`, `Pareidolia`, `Afterimage`, `Mirage`, `Euphoria`), 40-node visual DAG IDE, Obsidian 3D brain topology inspector, PC model scanner (Ollama, HF, vLLM, Hermes), and R4 defense-in-depth security (`weights_only=True` checkpoint hydration, symlink-resolved path confinement, proxy-aware `LUCID_MACHINES_API_TOKEN` bearer authentication, Origin/CSRF/SSRF guards, Ed25519 cryptographic plugin verification with `trusted=False` default, and OS keyring credential storage).

---

## 5-Tier Empirical Benchmark Summary (`CONTRIBUTING.md §1`)

| Tier | Benchmark Category | Hardware-Verified Metrics (RTX 5070 Laptop GPU) |
| :-: | :--- | :--- |
| **Tier 1** | **Isolated Kernel Microbenchmarks** | `VectorisedGLA` (`H=6, L=16384`): **`48.0 KiB`** state cache, **`10.07x`** prefill / **`7.31x`** decode vs. SDPA; `Gated DeltaNet`: **100.0% standard & overwrite MQAR recall**; Sparse `MoE_FFN(use_sparse_eval=True)`: **`1.648x`** zero-sync at `N=1` (`0.0` max diff) and **`7.518x`** at `d=384, E=8, k=2, N=4096` (`4.774x` top-1 / `5.450x` top-2 at `d=1536`); `chunked_cross_entropy`: **`98.5%` transient VRAM saved** vs. PyTorch (**`94.1%`** vs. Unsloth `2026.10.10`) at `V=32,000, N=8,192`; `FastSwiGLU`: **`1.95x`** vs. PyTorch. |
| **Tier 2** | **End-to-End 90.79M Model (`64.21M` Active)** | Rescued `step_40555`: **`383,135 tok/s`** peak prefill (`B=32, T=1024`), **`200,463 tok/s`** (`B=1, T=2048`), **`242,029 tok/s`** (`B=1, T=8192`), **`297,753 tok/s`** (`B=4, L=16384`), **`157.0 tok/s`** sequential tensor decode (`6.363 ms/tok`), **`40.8 tok/s` warm decode / `71.3 ms` TTFT** streaming inference, **`25,973 tok/s` LoRA fine-tuning** (**`+6.7%`** over Unsloth `24,344 tok/s`); `LayerStreamingEngine`: trains **`1.52B + CentroidSteer` in `3,132.6 MB` VRAM**; `NVFP4`: **`3.20x`** compression (`46.65 MB` across 96 blocks), Hadamard W4A16 4-bit PPL **`119.106`**. |
| **Tier 3** | **Multi-Layer Wrapper vs. Legacy Diagnostics** | Multi-layer long-context sweep (`L=16384`): **`64.0 KiB`** cache (**99.8% reduction** vs. SDPA), **`4.02x`** prefill / **`4.13x`** decode; production `MoE_FFN` (`1.648x–7.518x`) explicitly separated from legacy diagnostic `MoERoutingBlock` (`0.47x`). |
| **Tier 4** | **Tokenizer Compression vs. Model Compute** | Held-out lossless text compression: **`2.65x` aggregate** (`100%` reconstruction); compute-equivalent `compress_mode="expand"`: **`0.0` max logit diff** vs. base IDs; `order_pool` orthogonal RoPE phase rotation. |
| **Tier 5** | **Speculative & Conformal CALM Exits** | Monotonic exit PPL (`111.80 -> 111.18 -> 110.92`), **`4.31x` Reflex prefill speedup** (`741,070` vs. `172,099 tok/s`); `T=0.0` greedy self-speculative decode: **`91.62 tok/s`** (`51.28%` acceptance, 100% exact Cortex parity); `T=0.7` Leviathan rejection sampling: **`90.74 tok/s`** (`63.64%` acceptance, Pearson $\chi^2$ verified); Conformal CALM (`N=85,503` eval tokens, $\tau=0.20$): **`95.73%` Reflex agreement** (`31.34%` coverage, **`20.35%` FLOPs saved**). |

---

## Repository Structure

```text
LucidMachines/
├── lucid_machines/             # Core Python SDK, Kernels, Optimizers, Runtime, API & CLI
│   ├── ops/                    # Unified Operator API (gla, rmsnorm, rope, cross_entropy, centroid_steer, muon_ns)
│   ├── kernels/                # Triton & PyTorch Tensor Core kernels (fast_gla, fast_muon, fast_triangular, fast_ce, fast_nvfp4)
│   ├── model/                  # TriuneTransformer (TriuneModel alias), TriuneConfig, VectorisedGLA, MoE_FFN, GumbelSoftmaxRouter, FP8/FP4
│   ├── optim/                  # 3-Tier Optimizers: Muon, Multi-Directional CentroidSteer, AdamW
│   ├── runtime/                # LayerStreamingEngine, ParameterStager, DynamicResourceManager, PythonSandbox
│   ├── trainer/                # Trainer, TrainingEngine, LoRA/QLoRA, Checkpointing
│   ├── execution/ & plugins/   # 40-Node Typed DAG Engine & Ed25519 Cryptographic Plugin Registry
│   ├── api/                    # Hardened FastAPI server, OpenAI-compatible REST API, WebSocket telemetry
│   └── cli.py                  # Unified CLI (`kcx` primary; `polysom` / `lucid-machines` compat aliases)
├── triune/                     # Backward-compatibility Python shim delegating to lucid_machines
├── csrc/                       # Native C/AVX2/OpenMP & CUDA Engine (lucid_machines_core.dll / liblucid_machines_core.so)
├── studio/                     # Polysom Studio Frontend (studio/src/), .NET 10 WPF Host (studio/windows/) & Installers
├── models/                     # Triune architecture size templates (`models/configs/triune-*.json`) & `junkie.json` flagship model target
├── website/                    # React/Vite Project Website (index.html, src/, public/, SVG assets)
├── launch_kit/                 # Local Launch & Advertising Kit (messaging, blog, social drafts, SVG cards)
├── benchmarks/                 # Hardware-verified empirical benchmark reports + M1 CentroidSteer validation report
├── papers/                     # Companion LaTeX manuscripts & compiled PDFs (Paper 1: CentroidSteer, Paper 2: Systems)
├── tests/                      # Unit, mathematical invariant, security regression, and Tiers 1–4 E2E suites (tests/e2e/)
└── docs/                       # Architecture, mathematical foundations, installation, and RELEASE_NAME_REPORT.md
```

---

## 1-Click Installation & Launching Polysom Studio

### Windows (Native Installer, Double-Click, or PowerShell)
```cmd
:: Run the native Windows setup wizard or automated batch installer:
studio\publish\PolysomStudioSetup.exe
studio\installer\install_studio.bat
```

### Linux / WSL2 / macOS
```bash
chmod +x studio/installer/install_studio.sh
./studio/installer/install_studio.sh
```

### Rebuilding the Native Windows App & Installer (`scripts/publish_studio_windows.ps1`)
```powershell
python csrc/build_native.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/publish_studio_windows.ps1
```

---

## Quickstart (Python SDK & Unified CLI)

### Python SDK
```python
import torch
import lucid_machines

# Load a Triune architecture template ("triune-2.5b") or first flagship target ("junkie")
model = lucid_machines.load_model("triune-2.5b")

# Or upgrade any PyTorch / Hugging Face model with FastRMSNorm, LoRA & CentroidSteer
engine = lucid_machines.upgrade(model, lora_rank=16)

# Forward pass returning all 3 exit head logits
input_ids = torch.randint(0, 32000, (1, 128))
reflex_logits, limbic_logits, cortex_logits, route_logits = model.forward_all_exits(input_ids)
```

### CLI (`kcx` — Polysom KCX)
```bash
# Launch Polysom Studio Desktop App (`polysom` / `lucid-machines studio` aliases also supported)
kcx studio --port 8000

# Start embedded OpenAI-compatible API & UI server
kcx serve --port 8000

# Run hardware & CUDA diagnostics
kcx doctor

# Plan memory requirements for target GPU VRAM
kcx plan-memory --vram-gb 8.0
```

---

## Triune Architecture Templates & Inaugural `Junkie` Configurations (`models/configs/`)

**Junkie** is Lucid Machines' inaugural flagship model to be trained on the **Triune** architecture (`TriuneTransformer`), provided in four hardware-matched configurations:
- **`junkie-mini`** (`models/configs/junkie-mini.json`): **535,101,975** (`535.10M`) total / **230.81M** expected active (`12L / 1024D / 8H / 4E+1S`, `3.12 GiB` BF16 static VRAM).
- **`junkie-1b`** (`models/configs/junkie-1b.json`): **1,162,535,483** (`1.16B`) total / **374.99M** expected active (`22L / 1024D / 8H / 4E+1S`, `5.87 GiB` BF16 / `4.79 GiB` FP8 static VRAM).
- **`junkie-3b`** (`models/configs/junkie-3b.json`): **2,990,355,155** (`2.99B`) total / **395.74M** MoE-trunk Cortex active / **206.73M** dynamic-exit active (`16L / 1024D / 8H / 16E+1S`, **`10.10 GiB` FP8 static** — `< 20 GiB` resident on Colab L4 24GB FP8, #1 200-CU sweet spot!).
- **`junkie-7b` / `junkie`** (`models/configs/junkie-7b.json` & `junkie.json`): **7,396,845,163** (`7.40B`) total / **1.95B** expected active (`32L / 2048D / 16H / 8E+1S`, `34.49 GiB` BF16 static on single 80GB A100 vs `82.67 GiB` OOM under FP32 AdamW).

Triune architecture size templates live in `models/configs/triune-*.json`:

| Architecture Template | Legacy Alias | Layers | Hidden Dim | Heads | Experts | Exit Layers |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `triune-tiny` | `triune-micro` | 4 | 192 | 3 | 2 + 1 shared | L2, L3 |
| `triune-edge` | — | 6 | 256 | 4 | 4 + 1 shared | L2, L4 |
| `triune-nano` | — | 8 | 512 | 4 | 4 + 1 shared | L3, L6 |
| `triune-base` | — | 8 | 384 | 6 | 4 + 1 shared | L2, L5 |
| `triune-small` | — | 14 | 1024 | 8 | 4 + 1 shared | L5, L10 |
| `triune-pro` | `triune-large` | 16 | 768 | 12 | 8 + 1 shared | L5, L10 |
| `triune-1b` | `triune-1.5b` | 24 | 1024 | 16 | 8 + 1 shared | L7, L16 |
| `triune-2.5b` | — | 18 | 1280 | 10 | 8 + 1 shared | L5, L13 |
| `triune-large` | — | 28 | 1536 | 12 | 8 + 1 shared | L8, L19 |
| `triune-moe` | — | 24 | 1280 | 10 | 16 + 1 shared | L6, L16 |
| `triune-7b` | — | 32 | 2048 | 16 | 8 + 1 shared | L8, L22 |
| `triune-frontier` | `triune-32b` | 48 | 2560 | 20 | 16 + 1 shared | L11, L30 |

---

## Documentation Index

- [Release Name Selection & Conflict Report (`R1`)](docs/RELEASE_NAME_REPORT.md): Timestamped PyPI/npm/GitHub/DNS/web conflict matrix, 4-layer split (`Lucid Machines` company / `Triune` architecture / `Junkie` first model / `Polysom Studio` workbench), and non-legal disclaimer.
- [Architecture Overview (`docs/ARCHITECTURE.md`)](docs/ARCHITECTURE.md): 6-layer system architecture, 3-Tier CALM pipeline, and R2 multi-directional CentroidSteer design.
- [Installation Guide (`docs/INSTALL.md`)](docs/INSTALL.md) & [Windows Setup (`docs/WINDOWS_INSTALLATION.md`)](docs/WINDOWS_INSTALLATION.md): Native Windows `.NET 10` Polysom Studio installer, Python/CUDA bootstrap, and cross-platform setup.
- [Mathematical Foundations (`docs/MATH_FOUNDATIONS.md`)](docs/MATH_FOUNDATIONS.md): Formulations and proofs for batched `sCholQR2`, multi-directional CentroidSteer, Muon, conformal CALM gating, and Leviathan speculative rejection sampling.
- [Framework Guide (`docs/FRAMEWORK.md`)](docs/FRAMEWORK.md) & [Dynamic Modular Guide (`docs/DYNAMIC_MODULAR_GUIDE.md`)](docs/DYNAMIC_MODULAR_GUIDE.md): Subsystems, runtime layer streaming, DAG nodes, and configuration parameters.
- [Release Notes (`RELEASE_NOTES.md`)](RELEASE_NOTES.md) & [Deployment Guide (`DEPLOYMENT_GUIDE.md`)](DEPLOYMENT_GUIDE.md): Complete `v1.0.0` inaugural release notes and step-by-step publishing runbook.
- [Workflow 2 (`docs/WORKFLOW_2.md`)](docs/WORKFLOW_2.md) & [Cloud Runner Guide (`cloud/README.md`)](cloud/README.md): High-efficiency token & rules compliance workflow plus Colab Pro `Junkie-3B` L4 FP8 3-stage curriculum guide.
- [Security Policy (`SECURITY.md`)](SECURITY.md) & [Contributing Policy (`CONTRIBUTING.md`)](CONTRIBUTING.md): R4 security boundaries and 5-tier empirical verification rules.
