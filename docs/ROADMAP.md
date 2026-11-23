# Lucid Machines — Roadmap (`v1.0.0`)

## Phase 1: Core Engine, Native C/CUDA Kernels, Optimizers & Polysom Studio (`v1.0.0` — Completed)

- [x] **Lucid Machines** 3-Tier Architecture (`csrc/` Native C/AVX2/CUDA Engine `lucid_machines_core.dll`, `lucid_machines.ops` Unified Operator Layer, `PolysomStudio.exe` Native Desktop Workstation & `PolysomStudioSetup.exe` WPF Installer with 6 editorial themes: `Hypnagogic`, `Phosphene`, `Pareidolia`, `Afterimage`, `Mirage`, `Euphoria`).
- [x] Package architecture unified under `lucid-machines` / `lucid_machines` (`pyproject.toml`, `pip install -e .`) with `triune` backward-compatibility shim, **Triune** architecture presets (`TriuneTransformer` / `TriuneConfig`, `models/configs/triune-*.json`), and inaugural flagship models to train (`models/configs/junkie.json`, `junkie-7b.json`, `junkie-3b.json`, `junkie-1b.json`, `junkie-mini.json`).
- [x] Layer Streaming Engine: Zero-CPU-RAM-bloat streaming with `resident_gpu_layers`, non-pinned default staging, write-once forward snapshots, buffer mutation copy-back (`is_buffer=True`), and dynamic VRAM budget sizing (`1.52B` trained inside `3,132.6 MB` VRAM).
- [x] 3-Tier Partitioned Optimizer with Batched Kernel Fusion & Multi-Seed Validation against Scientific Baselines (`No-Steer Low-Rank`, `Memory-Matched Random Subspace`, `Full-Rank AdamW`):
  - Tier 1 (`Muon`): Batched 5th-order Newton-Schulz orthogonalization with fused `baddbmm` polynomial evaluation and `_foreach` multi-tensor updates.
  - Tier 2 (`CentroidSteer`): Exact Orthogonal Direct-Sum ($G = G_{\parallel C_e} \oplus G_{\perp C_e}$) Universal Triangular Core with Block-Oja EMA activation subspaces, Grassmann parallel transport, batched `sCholQR2`, sliced Householder fallback, true post-SwiGLU `last_hidden_centroids`, independent `steer_step` reset, and Triton-fused triangular AdamW.
  - Tier 3 (`AdamW` / `8-bit AdamW`): Fused adaptive moments for 1D vectors, norms, router gates, and embeddings.
- [x] Default sparse scatter-gather `MoE_FFN` (`use_sparse_eval=True`, zero-sync `N_tok=1` stacked BMM, up to `7.518x` at checkpoint width / `5.450x` at `1536d`), Vectorized GLA + optional Gated DeltaNet (`100%` std & overwrite MQAR), compacted per-token CALM margin exits, `T=0` greedy + Leviathan et al. `T>0` speculative rejection sampling, and `compress_mode` (`mean_pool` / `expand` `0.0` logit diff).
- [x] Native FP8 (`float8_e4m3fn`) and NVFP4 (`NVFP4Linear` + Hadamard W4A16) execution with dynamic quantization.
- [x] Intermediate exit `RMSNorm` normalization matching terminal Cortex representation scale.
- [x] Hardware Resource Manager & Neuron Scope: Live VRAM/RAM telemetry, activation probes, and single-file `PolysomStudio.exe` bundling `lucid_machines_core.dll`.

## Phase 2: Pretraining and Distributed Scaling (Active)

- [x] **Track 0**: Multi-seed FineWeb-Edu pre-training & checkpoint-continuation verification on `90.79M` / `75.8M` checkpoints (`N=12`, `df=11` Latin-square TOST equivalence within `±0.01` nats, `p_TOST = 1.52e-7`; M1 `N=5` CentroidSteer validation `p = 0.0068` vs. AdamW).
- [ ] **Track 1**: Pretraining **Junkie** (`models/configs/junkie.json`, first flagship model to train on the Triune architecture) and `triune-2.5b` (~2.45B params, ~780M active) on `FineWeb-Edu` using layer streaming and Muon on single-GPU hardware.
- [ ] **Track 2**: Multi-GPU distributed training setup for `triune-7b` (~7.2B params, ~2.2B active) via PyTorch FSDP2 / torchrun.
- [x] Calibrated joint multi-exit KL self-distillation (`checkpoints/triune_studio_step_36500_rescued.pt`, `step_40555`: monotonic `111.80 -> 111.18 -> 110.92` PPL and `95.73%` conformal CALM Reflex agreement at $\tau=0.20$).
- [ ] SafeTensors checkpoint export and Hugging Face Hub integration.

## Phase 3: Polysom Studio and Deployment (Delivered & Extending)

- [x] Standalone Windows installer (`PolysomStudioSetup.exe`) and self-contained desktop binary (`PolysomStudio.exe`).
- [x] Real-time training dashboard: loss curves, exit distribution, `sCholQR2` telemetry, expert load telemetry, and 6 visual themes.
- [x] Graph-based pipeline editor (`CLI Training Graph`) for dataset ingestion, fine-tuning, and export.
- [ ] Cross-platform macOS/Linux signed standalone binaries and GGUF export for external runtimes.

---

## Frontier Research & Next-Generation Architecture

See [FRONTIER_RESEARCH_ROADMAP.md](FRONTIER_RESEARCH_ROADMAP.md) for the 2024–2026 frontier architecture roadmap:
- Non-Generative Joint-Embedding Predictive Architecture (JEPA / JEV / Semantic Tube Prediction) for Reflex/Limbic exits.
- Gated DeltaNet & Mamba-2 / SSD hybrid linear recurrence.
- DeepSeek-V3 auxiliary-loss-free dynamic bias routing & Multi-Head Latent Attention (MLA).
- Constant-State Radix prompt caching ($48.0\text{ KiB}$ state) & chunked prefill scheduling.
