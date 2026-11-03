# Triune AI — Roadmap

## Phase 1: Core Engine, Native C/CUDA Kernels & Optimizers (Completed)

- [x] **Triune AI** 3-Tier Refactored Architecture (`csrc/` Native C/AVX2/CUDA Engine, `triune.ops` Unified Operator Layer, `TriuneStudio.exe` Native Desktop Workstation).
- [x] Package architecture unified under `triune` root (`pyproject.toml`, `pip install -e .`) with Minimalist Public API (`triune.Model`, `triune.Config`, `triune.CentroidSteer`, `triune.Muon`, `triune.ops`).
- [x] Layer Streaming Engine: Zero-CPU-RAM-bloat streaming with `resident_gpu_layers`, non-pinned default staging, and dynamic VRAM budget sizing.
- [x] 3-Tier Partitioned Optimizer with Batched Kernel Fusion:
  - Tier 1 (`Muon`): Batched 5th-order Newton-Schulz orthogonalization with fused `baddbmm` polynomial evaluation and `_foreach` multi-tensor updates.
  - Tier 2 (`CentroidSteer`): Universal Triangular Core with batched Cholesky-QR subspace iteration, batched dual-orthonormal centroid steering, and Triton-fused triangular AdamW.
  - Tier 3 (`AdamW` / `8-bit AdamW`): Fused adaptive moments for 1D vectors, norms, router gates, and embeddings.
- [x] Fused QKVG / SwiGLU projections, factored intra-chunk & batched inter-chunk Gated Linear Attention (`fast_chunk_gla`), and batched 3D BMM MoE routing.
- [x] Native FP8 (`float8_e4m3fn`) and NF4 (`FP4Linear`) execution with dynamic quantization.
- [x] Intermediate exit `RMSNorm` normalization matching terminal Cortex representation scale.
- [x] Hardware Resource Manager & Neuron Scope: Live VRAM/RAM telemetry, activation probes, and single-file `TriuneStudio.exe` bundling `triune_core.dll`.

## Phase 2: Pretraining and Distributed Scaling (Active)

- [ ] **Track 1**: Pretraining `triune-2.5b` (~2.45B params, ~780M active) on `FineWeb-Edu` using layer streaming and Muon on single-GPU hardware.
- [ ] **Track 2**: Multi-GPU distributed training setup for `triune-7b` (~7.2B params, ~2.2B active) via PyTorch FSDP2 / torchrun.
- [ ] Calibrated routing loss for intermediate exit decision thresholds.
- [ ] SafeTensors checkpoint export and Hugging Face Hub integration.

## Phase 3: Studio and Deployment (Upcoming)

- [ ] Standalone installer packages for Triune Studio.
- [ ] Real-time training dashboard: loss curves, exit distribution, and expert load telemetry.
- [ ] Graph-based pipeline editor for dataset ingestion, fine-tuning, and export.
- [ ] GGUF export for local inference runtimes.
