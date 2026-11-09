# Triune AI — Research Manuscripts & Preprints

This directory contains research preprints, vector figures, and LaTeX manuscripts documenting the architectural, mathematical, and systems innovations introduced in **Triune AI**.

---

## Research Preprints (Working Manuscripts)

### 1. CentroidSteer: Activation-Guided Low-Rank Subspace Optimization for Sparse Mixture-of-Experts
- **Track**: Machine Learning Optimization & Scalable Pre-Training
- **Manuscript**: [`paper1_centroidsteer/main.tex`](paper1_centroidsteer/main.tex)
- **Compiled PDF**: [`paper1_centroidsteer/centroidsteer.pdf`](paper1_centroidsteer/centroidsteer.pdf)
- **Primary Artifacts**:
  - Theorem 1 proving upper-triangular Krylov projection core identity $\bar{S} = \bar{Q}_L^T G \bar{Q}_R = \begin{bmatrix} R_L & Q_L^T G \tilde{v}_\perp \\ \mathbf{0} & \tilde{c}_\perp^T G \tilde{v}_\perp \end{bmatrix}$.
  - Proposition 1 establishing basis isometricity and robust fallback to effective rank $r$ under degeneracy.
  - Multi-seed FineWeb-Edu convergence evaluation ($N=3$ seeds from scratch; $N=5$ seeds, $200$ steps from a 75.8M checkpoint with seed-specific batch permutations, Latin-square counterbalancing, empirical minimum equivalence margin $\Delta_{\min}$, and multi-margin Schuirmann TOST).
  - Paired per-seed steering ablation analysis with confidence intervals and hypothesis testing.
  - Two-pass shifted Cholesky-QR (`sCholQR2`) with synchronization-free sliced Householder fallback ($15.2\times$ faster than Householder QR on full-rank matrices, $84.75\%$ fast-path survival across $4,905$ live training QR calls).
  - Exact memory accounting: **0.128 bytes/param** on Tier 2 expert weights ($98.4\%$ persistent state reduction vs.\ AdamW), $4.54$ bytes/param model-wide with standard Tier 3 AdamW ($1.27$ bytes/param with 8-bit AdamW), and $12.0\%$ incremental peak training VRAM reduction ($119.9\text{ MB}$ saved on the 75.8M checkpoint).

```bibtex
@article{bhardwaj2026centroidsteer,
  title={CentroidSteer: Activation-Guided Low-Rank Subspace Optimization for Sparse Mixture-of-Experts},
  author={Bhardwaj, Yash},
  note={Working paper / preprint under review},
  url={https://github.com/Yasho4867/triune},
  year={2026}
}
```

---

### 2. Triune: A Memory-Virtualizing and Constant-State Transformer Architecture for Long-Context Autoregressive Models
- **Track**: Efficient Deep Learning Systems & Hardware-Aware Machine Learning
- **Manuscript**: [`paper2_triune_systems/main.tex`](paper2_triune_systems/main.tex)
- **Compiled PDF**: [`paper2_triune_systems/triune_architecture.pdf`](paper2_triune_systems/triune_architecture.pdf)
- **Primary Artifacts**:
  - Gated Linear Attention (GLA) with fused Triton recurrent scan: constant $\mathcal{O}(1)$ decode memory ($48.0\text{ KiB}$ fixed cache for $H=6$ heads, $64.0\text{ KiB}$ for $H=8$ heads; $99.8\%$ compression vs.\ Softmax SDPA at 16k context, up to $7.94\times$ faster decode).
  - Hierarchical Dynamic Exits (Reflex L2, Limbic L5, Cortex L8) with learned entropy-gated routing ($-35.8\%$ forward latency reduction, $56.8\%$ layer-skip reduction, $1.49\times$ decode speedup vs Cortex).
  - Chunked Vocabulary-Projection Cross-Entropy: $-91.3\%$ peak transient memory and $1.76\times$ speedup at vocabulary size $V=32,000$, with exact online $\nabla_W \mathcal{L}$ gradient accumulation.
  - Multi-Level Parallelism: Asynchronous PCIe DMA prefetching ($1.28\times$ throughput speedup, $812,956\text{ tok/s}$), Tensor Core GEMM projection fusion ($6.14\times$ micro-kernel speedup), and asynchronous CUDA Graph replay ($-99.0\%$ CPU launch latency with bitwise optimizer parity).
  - Head-to-head competitive benchmark vs.\ genuine installed Unsloth (2026.10.10): $+19.7\%$ throughput on LoRA fine-tuning with native Mixture-of-Experts execution.

```bibtex
@article{bhardwaj2026triune,
  title={Triune: A Memory-Virtualizing and Constant-State Transformer Architecture for Long-Context Autoregressive Models},
  author={Bhardwaj, Yash},
  note={Working paper / preprint under review},
  url={https://github.com/Yasho4867/triune},
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
