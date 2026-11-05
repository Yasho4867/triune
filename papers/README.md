# Triune AI — Research Publications

This directory contains publication-grade research papers, vector figures, and LaTeX manuscripts documenting the architectural, mathematical, and systems innovations introduced in **Triune AI**.

---

## Published Papers

### 1. CentroidSteer: Tri-Tier Subspace Optimization and Polar Orthogonalization for Dynamic Mixture-of-Experts
- **Track**: Machine Learning Optimization & Scalable Pre-Training
- **Manuscript**: [`paper1_centroidsteer/main.tex`](paper1_centroidsteer/main.tex)
- **Compiled PDF**: [`paper1_centroidsteer/centroidsteer.pdf`](paper1_centroidsteer/centroidsteer.pdf)
- **Primary Artifacts**:
  - Multi-seed FineWeb-Edu convergence evaluation ($N=3$ seeds: 42, 1337, 2026).
  - Tri-tier parameter partitioning: Muon polar orthogonalization (Tier 1), CentroidSteer low-rank router subspace projection (Tier 2), and Fused AdamW base (Tier 3).
  - Scale-invariant Cholesky-QR Gram orthogonalization ($6.8\times$ faster than SVD, $3.1\times$ lower transient VRAM).
  - Persistent optimizer state memory reduced to **$1.00$ byte/param** ($87.5\%$ reduction vs.\ standard AdamW).
  - Highest convergence efficiency ($0.1916$ loss/s) on 90.8M Mixture-of-Experts.

```bibtex
@article{bhardwaj2026centroidsteer,
  title={CentroidSteer: Tri-Tier Subspace Optimization and Polar Orthogonalization for Dynamic Mixture-of-Experts},
  author={Bhardwaj, Yash},
  journal={arXiv preprint},
  year={2026}
}
```

---

### 2. Triune: A Sub-Quadratic, Multi-Exit Transformer Architecture with Fused Logit-Free Kernels and Asynchronous Pipelining
- **Track**: Efficient Deep Learning Systems & Hardware-Aware Machine Learning
- **Manuscript**: [`paper2_triune_systems/main.tex`](paper2_triune_systems/main.tex)
- **Compiled PDF**: [`paper2_triune_systems/triune_architecture.pdf`](paper2_triune_systems/triune_architecture.pdf)
- **Primary Artifacts**:
  - Gated Linear Attention (GLA) with fused Triton recurrent scan: constant $\mathcal{O}(1)$ decode memory ($64.0\text{ KB}$ fixed cache, $99.8\%$ compression vs.\ Softmax SDPA at 16k context, up to $7.94\times$ faster decode).
  - Hierarchical Dynamic Exits (Reflex L2, Limbic L5, Cortex L8) with learned entropy-gated routing ($-74.4\%$ latency on low-entropy tokens).
  - Chunked Logit-Free Cross-Entropy: $-91.3\%$ peak transient memory and $1.76\times$ speedup at vocabulary size $V=32,000$.
  - Multi-Level Parallelism: Asynchronous PCIe DMA prefetching ($1.28\times$ throughput speedup, $812,956\text{ tok/s}$), Tensor Core GEMM projection fusion ($6.14\times$ speedup), and asynchronous CUDA Graph replay ($-99.0\%$ CPU launch latency with bitwise optimizer parity).
  - Head-to-head competitive benchmark vs.\ genuine installed Unsloth (2026.10.10): $+19.7\%$ throughput on LoRA fine-tuning with native Mixture-of-Experts execution.

```bibtex
@article{bhardwaj2026triune,
  title={Triune: A Sub-Quadratic, Multi-Exit Transformer Architecture with Fused Logit-Free Kernels and Asynchronous Pipelining},
  author={Bhardwaj, Yash},
  journal={arXiv preprint},
  year={2026}
}
```

---

## Reproduction & Build Commands

To recompile all figures and LaTeX manuscripts to PDF from source:

```bash
# 1. Regenerate publication vector figures
python papers/scripts/generate_figures.py

# 2. Build both research paper PDFs
python papers/scripts/build_papers.py
```
