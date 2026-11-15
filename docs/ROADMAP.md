# Triune AI — Roadmap

## Phase 1: Core Engine, Native C/CUDA Kernels & Optimizers (Completed)

- [x] **Triune AI** 3-Tier Refactored Architecture (`csrc/` Native C/AVX2/CUDA Engine, `triune.ops` Unified Operator Layer, `TriuneStudio.exe` Native Desktop Workstation & `TriuneStudioSetup.exe` WPF Installer).
- [x] Package architecture unified under `triune` root (`pyproject.toml`, `pip install -e .`) with Minimalist Public API (`triune.Model`, `triune.Config`, `triune.CentroidSteer`, `triune.Muon`, `triune.ops`).
- [x] Layer Streaming Engine: Zero-CPU-RAM-bloat streaming with `resident_gpu_layers`, non-pinned default staging, write-once forward snapshots, buffer mutation copy-back (`is_buffer=True`), and dynamic VRAM budget sizing.
- [x] 3-Tier Partitioned Optimizer with Batched Kernel Fusion:
  - Tier 1 (`Muon`): Batched 5th-order Newton-Schulz orthogonalization with fused `baddbmm` polynomial evaluation and `_foreach` multi-tensor updates.
  - Tier 2 (`CentroidSteer`): Universal Triangular Core with batched `sCholQR2`, sliced Householder fallback, Kahan-Parlett centroid steering, non-refresh moment-zeroing invariant, and Triton-fused triangular AdamW (`0.128 B/p`).
  - Tier 3 (`AdamW` / `8-bit AdamW`): Fused adaptive moments for 1D vectors, norms, router gates, and embeddings.
- [x] Default sparse scatter-gather `MoE_FFN` (`use_sparse_eval=True`, zero-sync `N_tok=1` stacked BMM, up to `7.58x` at checkpoint width / `5.45x` at `1536d`), Vectorized GLA + optional Gated DeltaNet (`100%` std & overwrite MQAR), compacted per-token CALM margin exits, `T=0` greedy + Leviathan et al. `T>0` speculative rejection sampling, and `compress_mode` (`mean_pool` / `expand` `0.0` logit diff).
- [x] Native FP8 (`float8_e4m3fn`) and NF4 (`FP4Linear`) execution with dynamic quantization.
- [x] Intermediate exit `RMSNorm` normalization matching terminal Cortex representation scale.
- [x] Hardware Resource Manager & Neuron Scope: Live VRAM/RAM telemetry, activation probes, and single-file `TriuneStudio.exe` bundling `triune_core.dll`.

## Phase 2: Pretraining and Distributed Scaling (Active)

- [x] **Track 0**: Multi-seed FineWeb-Edu pre-training & checkpoint-continuation verification on `triune` 43.5M–90.79M checkpoints (`N=12`, `df=11` Latin-square TOST equivalence within `±0.01` nats, `p_TOST = 1.52e-7`).
- [ ] **Track 1**: Pretraining `triune-2.5b` (~2.45B params, ~780M active) on `FineWeb-Edu` using layer streaming and Muon on single-GPU hardware.
- [ ] **Track 2**: Multi-GPU distributed training setup for `triune-7b` (~7.2B params, ~2.2B active) via PyTorch FSDP2 / torchrun.
- [ ] Calibrated joint multi-exit training for higher Reflex-Cortex CALM agreement (uncalibrated single-exit checkpoint currently measures 9.4% high-confidence agreement).
- [ ] SafeTensors checkpoint export and Hugging Face Hub integration.

## Phase 3: Studio and Deployment (Delivered & Extending)

- [x] Standalone Windows installer (`TriuneStudioSetup.exe`) and self-contained desktop binary (`TriuneStudio.exe`).
- [x] Real-time training dashboard: loss curves, exit distribution, sCholQR2 telemetry, and expert load telemetry.
- [x] Graph-based pipeline editor (`CLI Training Graph`) for dataset ingestion, fine-tuning, and export.
- [ ] Cross-platform macOS/Linux signed standalone binaries and GGUF export for external runtimes.

---

## Frontier Research & Next-Generation Architecture

See [FRONTIER_RESEARCH_ROADMAP.md](FRONTIER_RESEARCH_ROADMAP.md) for the 2024–2026 frontier architecture roadmap:
- Non-Generative Joint-Embedding Predictive Architecture (JEPA / JEV / Semantic Tube Prediction) for Reflex/Limbic exits.
- Gated DeltaNet & Mamba-2 / SSD hybrid linear recurrence.
- DeepSeek-V3 auxiliary-loss-free dynamic bias routing & Multi-Head Latent Attention (MLA).
- Constant-State Radix prompt caching ($48.0\text{ KiB}$ state) & chunked prefill scheduling.

