# Triune AI — Research Manuscripts & Preprints

This directory contains research preprints, vector figures, and LaTeX manuscripts documenting the architectural, mathematical, and systems innovations introduced in **Triune AI**.

---

## Research Preprints (Working Manuscripts)

### 1. CentroidSteer: Activation-Guided Low-Rank Subspace Optimization on the Stiefel Manifold for Sparse Mixture-of-Experts
- **Track**: Machine Learning Optimization & Scalable Pre-Training
- **Manuscript**: [`paper1_centroidsteer/main.tex`](paper1_centroidsteer/main.tex)
- **Compiled PDF**: [`paper1_centroidsteer/centroidsteer.pdf`](paper1_centroidsteer/centroidsteer.pdf)
- **Primary Artifacts**:
  - Riemannian Geometry on the Stiefel Manifold $\operatorname{St}(r, d)$, including canonical tangent projections $\mathcal{P}_{T_Q}$, exact geodesic exponential mapping, polar Cholesky retraction $\mathcal{R}_Q^{\text{polar}}$, and Cayley transform retraction $\mathcal{R}_W^{\text{Cayley}}$.
  - Theorem 1 proving upper-triangular Krylov projection core identity $\bar{S} = \bar{Q}_L^T G \bar{Q}_R = \begin{bmatrix} R_L & Q_L^T G \tilde{v}_\perp \\ \mathbf{0} & \tilde{c}_\perp^T G \tilde{v}_\perp \end{bmatrix}$.
  - Proposition 1 establishing basis isometricity on $\operatorname{St}(r+1, d)$ and robust fallback to effective rank $r$ under degeneracy.
  - Theorem 2 establishing the routing entropy concentration bound, linking token gating entropy $\mathcal{H}(p)$ to intra-cluster trace dispersion $\operatorname{tr}(\Sigma_e)$ and residual steering perturbation $\|c_\perp\|_2$.
  - Theorem 3 proving unconditional finite-precision orthogonal stability of two-pass shifted Cholesky-QR (`sCholQR2`) under shift regularization and non-convex ergodic convergence rate $\mathcal{O}(1/\sqrt{T})$ on the Stiefel manifold.
  - Multi-seed FineWeb-Edu convergence evaluation ($N=3$ seeds from scratch; $N=12$ seeds, $df=11$, $200$ steps from a 75.8M checkpoint at primary step $36,500$ [$222.7\text{M}$ tokens] plus supplementary stage-consistency verification across $N=3$ seeds at step $35,500$ [$157.2\text{M}$ tokens] sampling from a $1,000$-batch / $512,000$-token pool across balanced $3 \times 3$ Latin squares, evaluated on a disjoint 64-batch / $32,768$-token held-out validation set with empirical minimum equivalence margin $\Delta_{\min} = \pm 0.0047$ nats and exact upper-tail Schuirmann TOST $p_{\text{TOST}} = 1.52 \times 10^{-7}$ at $\pm 0.01$ nats; accompanied by extended-horizon continuation up to 600 steps/seed tracking intermediate validation curves at steps 0, 200, 400, and 600 where the paired gap narrows from $+0.0051$ to $+0.0031$ nats).
  - Paired per-seed steering ablation analysis with confidence intervals and hypothesis testing.
  - Two-pass shifted Cholesky-QR (`sCholQR2`) with sliced Householder fallback ($15.2\times$ faster than Householder QR on full-rank matrices, $83.76\%$ fast-path survival across $11,832$ live training QR calls at step $36,500$ and $84.64\%$ across $3,066$ live calls at step $35,500$, with zero host-device syncs on the $95\%–99.5\%$ non-refresh steps).
  - Exact memory accounting: **0.128 bytes/param** on Tier 2 expert weights ($98.4\%$ persistent state reduction vs.\ AdamW), $4.54$ bytes/param model-wide with standard Tier 3 AdamW ($1.27$ bytes/param with 8-bit AdamW), $12.2\%$ incremental same-process peak VRAM reduction ($121.8\text{ MB}$ saved), and $11.5\%$ separate-OS-subprocess peak PyTorch-allocated tensor memory reduction ($122.4\text{ MB}$ saved across 36 fresh child processes spanning 4 complete $3 \times 3$ Latin squares at $N=12$, $df=11$ from an unallocated $0.00\text{ MB}$ PyTorch tensor baseline, $942.1\text{ MB}$ allocated / $1,342.8\text{ MB}$ reserved vs $1,064.5\text{ MB}$ allocated / $1,308.8\text{ MB}$ reserved, with physical CUDA driver overhead residing outside the PyTorch allocator pool, at matched full-step latency).

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
  - Theorem 1 establishing bounded spatial memory complexity $\Theta(1)$ and constant per-step decode FLOPs $\Theta(H d_k d_v)$ for Vectorized Gated Linear Attention (GLA) with fused Triton recurrent scan ($48.0\text{ KiB}$ to $64.0\text{ KiB}$ per layer across sequences up to 16,384 tokens; $99.8\%$ compression vs.\ Softmax SDPA at 16k context, up to $7.94\times$ faster decode).
  - Proposition 1 establishing the exact mathematical gradient equivalence of Streaming Chunked Vocabulary Cross-Entropy, reducing transient memory by $91.3\%$ on $V=32,000$ with online rank-$B_c$ GEMM accumulation and numerical discrepancy bounded by $\mathcal{O}(\lceil N/B_c \rceil \epsilon_{\text{mach}} \|H\|_F \|W_{\text{head}}\|_F)$.
  - Theorem 2 proving asymptotic communication bubble elimination ($\beta(K) \to 0$) under Pipelined Asynchronous DMA Streaming on dedicated CUDA streams ($1.28\times$ throughput speedup, $812,956\text{ tok/s}$).
  - Theorem 3 proving distributed throughput scaling limits and Roofline operational intensity bounds ($I_{\text{GLA}} \gg I^*$, compute-bound execution) with dynamic exit routing ($\bar{\tau}_{\text{depth}} \approx 0.432$, delivering up to $2.31\times$ aggregate throughput scaling).
  - Hierarchical Dynamic Exits (Reflex L2, Limbic L5, Cortex L8) with learned entropy-gated routing ($-35.8\%$ forward latency reduction, $56.8\%$ layer-skip reduction, $1.49\times$ decode speedup vs Cortex).
  - Multi-Level Parallelism: Asynchronous PCIe DMA prefetching, Tensor Core GEMM projection fusion ($6.14\times$ micro-kernel speedup), and asynchronous CUDA Graph replay ($-99.0\%$ CPU launch latency with bitwise optimizer parity).
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
