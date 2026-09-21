# Roadmap

## Phase 1: Core Engine and Optimizers (Completed)

- [x] Package architecture unified under `triune` root (`pyproject.toml`, `pip install -e .`).
- [x] Layer Streaming Engine: Sub-350 MB peak VRAM training for 2.5B+ architectures.
- [x] 3-Tier Partitioned Optimizer:
  - Tier 1 (Muon): 5th-order Newton-Schulz orthogonalization for 2D hidden projections.
  - Tier 2 (CentroidSteer): Dual-sided SVD GaLore with centroid steering for routed MoE experts.
  - Tier 3 (AdamW): Standard adaptive moments for 1D vectors and norms.
- [x] Native FP8 (`float8_e4m3fn`) scaled GEMM execution with dynamic quantization.
- [x] Intermediate exit RMSNorm normalization matching terminal Cortex scale.
- [x] Hardware Resource Manager: VRAM telemetry, allocation planning, and `--force` override.
- [x] FastAPI server with OpenAI-compatible chat completions and WebSocket telemetry.

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
