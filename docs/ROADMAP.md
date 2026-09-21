# Triune Ecosystem Roadmap

## Phase 1 – Framework Engine & Algorithmic Breakthroughs (Completed ✅)
- ✅ Core `triune` package unified under single root packaging (`pyproject.toml`, `pip install -e .`).
- ✅ **AirLLM-Style Layer Streaming Engine**: Sub-300 MB peak VRAM execution for 2.5B–5B architectures on 8 GB laptop GPUs.
- ✅ **3-Tier Optimizer Architecture**:
  - **Tier 1 (Muon)**: 5th-order Newton-Schulz matrix orthogonalization for non-expert 2D hidden projections.
  - **Tier 2 (CentroidSteer)**: Dual-sided SVD GaLore with semantic activation centroid steering for routed MoE experts.
  - **Tier 3 (AdamW)**: Adaptive moment estimation for 1D vectors and norms.
- ✅ **Native Hardware FP8 (E4M3) Scaled GEMM**: Hardware acceleration on modern GPUs (Ada Lovelace, Blackwell) with dynamic quantization.
- ✅ **Variance-Matched Exit Normalization**: Intermediate `RMSNorm` at Reflex and Limbic layers eliminating early-exit gradient explosions.
- ✅ **Dynamic Hardware Resource Manager**: Automated feasibility probing, live VRAM telemetry, and strict architecture immutability with `--force` authority.
- ✅ **Embedded FastAPI & WebSocket Telemetry Server**: OpenAI-compatible `/v1/chat/completions` and streaming telemetry.

## Phase 2 – Pretraining & Distributed Scaling (Active Focus 🚀)
- 🔲 **Track 1 (Laptop Pretraining)**: Pretrain native **`triune-2.5b`** (~2.45B params, ~780M active) on `HuggingFaceFW/fineweb-edu` using Layer Streaming + Muon on RTX 5070 (8 GB).
- 🔲 **Track 2 (Cloud Distributed Scaling)**: Multi-GPU cluster pretraining for **`triune-7b`** (~7.2B params, ~2.2B active) using distributed PyTorch FSDP2 / torchrun.
- 🔲 Dynamic routing calibration: Platt scaling / ECE calibrated loss on router logits to achieve Jev-style high-confidence System 1 early returns.
- 🔲 Checkpoint export pipeline: SafeTensors and Hugging Face Hub direct publishing (`triune/triune-2.5b-base`).

## Phase 3 – Triune Studio & Ecosystem (Upcoming 🌟)
- 🔲 One-click standalone Windows installer for Triune Studio.
- 🔲 Real-time Training Visualizer: Interactive loss curves, router exit distribution charts, and per-expert load balancing graphs.
- 🔲 DAG Workflow Builder: Drag-and-drop node graph for dataset ingestion, fine-tuning, and model export.
- 🔲 GGUF & llama.cpp export for ultra-fast local inference across Apple Silicon and CPU/GPU runtimes.
