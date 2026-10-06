# Contributing to Triune Transformer

Thank you for your interest in contributing to Triune Transformer! We welcome contributions ranging from bug fixes, architecture optimizations, documentation, and DAG plugins to new research recipes.

---

## 1. Architecture Overview

Triune Transformer is a hierarchical multi-exit neural architecture with gated linear attention and consumer-GPU layer streaming:

- **Core Hierarchy**:
  - **Reflex Tier** (Exits at early layer, e.g., Layer 6): Fast low-latency generation for routine tokens.
  - **Limbic Tier** (Exits at intermediate layer, e.g., Layer 16): Contextual reasoning.
  - **Cortex Tier** (Final exit, Layer 24): Full multi-head capacity and deep MoE representation.
- **Attention & FFN**:
  - Gated Linear Attention (`VectorisedGLA`) with recurrent $O(1)$ state stepping during inference.
  - Sparse Mixture of Experts (`MoE_FFN`) with top-$k$ routing, centroid steering, and auxiliary load balancing.
- **Runtime & Execution**:
  - `LayerStreamingEngine`: CPU master parameter storage with GPU prefetching and D2H gradient transfer, allowing 7B models to train on 8GB VRAM consumer GPUs.
  - `DAGParser` & `ExecutionEngine`: Strongly typed port DAG execution graph connecting tokenizer, dataloader, model blocks, losses, optimizers, and exporters.

---

## 2. Setting Up Your Development Environment

Triune Transformer is developed primarily on Linux / WSL2 (Ubuntu 22.04 / 24.04) with Python 3.10 or 3.11.

### Prerequisites
- Python 3.10+
- PyTorch 2.2+ (CUDA 12.1+ recommended for GPU training)
- Node.js 18+ (for Studio UI development)

### Clone and Install
```bash
git clone https://github.com/Yasho4867/TriuneTransformer.git
cd TriuneTransformer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies in editable mode
pip install -e .
```

---

## 3. Running the Test Suite

We maintain a comprehensive suite of unit, integration, numerical parity, and streaming lifecycle tests:

```bash
# Run all unit tests
python -m unittest discover tests

# Run specific test modules
python -m unittest tests/test_ecosystem.py
python -m unittest tests/test_model_semantics_and_checkpoint.py
python -m unittest tests/test_streaming_training_parity.py
```

Before submitting a PR, make sure:
1. All unit tests pass cleanly without errors or regressions.
2. No mock or simulated values are returned from active computation nodes in the DAG engine.
3. Memory safety invariants are preserved (e.g. pinned memory buffers, autograd graph continuity).

---

## 4. Coding Standards

- **Python Style**: Follow PEP 8 guidelines. Type hints are encouraged across all core modules (`from __future__ import annotations`).
- **Autograd Integrity**: Avoid in-place modifications (`.index_copy_()`, `.copy_()`) on tensors that are tracked in the computation graph. Always verify backward gradient flow.
- **Device & Precision Safety**:
  - Always stage weights to CPU before saving checkpoints or during initial hydration (`map_location="cpu"`).
  - Explicitly verify hardware capabilities before invoking vendor-specific operations (e.g., `cublas_scaled_mm` requires Ada Lovelace / Compute Capability 8.9+).
- **DAG Ports & Contracts**:
  - Nodes must declare explicit input/output port types (`PortType`) and execution contracts (`NodeExecutionType`: `execution` vs `specification`).
  - Do NOT hardcode fake throughput, synthetic delay loops, or fabricated loss values in the UI or DAG handlers.

---

## 5. Pull Request Guidelines

1. **Branch Naming**: Use descriptive branch names: `feature/gla-chunking`, `fix/streaming-prefetch`, `docs/byok-setup`.
2. **Commit Messages**: Write concise, informative commit messages explaining *why* a change was made.
3. **Tests**: Include unit tests covering new features or reproducing any bugs being fixed.
4. **Documentation**: Update relevant docs or docstrings when modifying public APIs or configuration schemas.

---

## 6. Questions & Community

- Open an issue on GitHub for bug reports or feature proposals.
- For architectural discussions, refer to `audit_report_final.md` and related design documents in the repository.
