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
  - Theorem 2 proving **Second-Moment ANOVA Energy Conservation & Total Covariance Dispersion** ($\Sigma_{\text{total}} = \Sigma_{\text{between}} + \mathbb{E}[\Sigma_e]$) alongside Cauchy--Schwarz centered gradient bounds, verified empirically across $N=8$ MoE seeds ($270.99 + 113.01 = 384.00 = d$, $46.86\%$ within-expert variance reduction).
  - Theorem 3 proving unconditional finite-precision orthogonal stability of two-pass shifted Cholesky-QR (`sCholQR2`) under shift regularization and non-convex ergodic convergence rate $\mathcal{O}(1/\sqrt{T})$ on the Stiefel manifold.
  - Multi-seed FineWeb-Edu convergence evaluation ($N=8$ MoE seeds across 6 regimes comparing BAdam, AdamW, Muon+AdamW, GaLore, CentroidSteer No-Muon, and Triune MoE with exact paired/Welch $p$-values [$p=0.0082$ vs. AdamW, $p=0.7316$ vs. GaLore] and $N=8$ steering ablation cleanly distinguishing short-horizon subspace gains from centroid-vs-unsteered loss [$p=0.1188$]; $N=12$ seeds, $df=11$, $200$ steps from a 75.8M checkpoint at primary step $36,500$ [$222.7\text{M}$ tokens] plus supplementary stage-consistency verification across $N=3$ seeds at step $35,500$ [$157.2\text{M}$ tokens] sampling from a $1,000$-batch / $512,000$-token pool across balanced $3 \times 3$ Latin squares, evaluated on a disjoint 64-batch / $32,768$-token held-out validation set with empirical minimum equivalence margin $\Delta_{\min} = \pm 0.0047$ nats and exact upper-tail Schuirmann TOST $p_{\text{TOST}} = 1.52 \times 10^{-7}$ at $\pm 0.01$ nats; accompanied by extended-horizon continuation up to 600 steps/seed where the paired gap narrows from $+0.0051$ to $+0.0031$ nats).
  - Paired per-seed steering ablation analysis with ANOVA covariance telemetry and Welch/paired confidence bounds.
  - Two-pass shifted Cholesky-QR (`sCholQR2`) with sliced Householder fallback ($15.2\times$ faster than Householder QR on full-rank matrices, $83.76\%$ fast-path survival across $11,832$ live training QR calls at step $36,500$ and $84.64\%$ across $3,066$ live calls at step $35,500$, with zero host-device syncs on the $95\%–99.5\%$ non-refresh steps).
  - Exact memory accounting: **0.128 bytes/param** analytical on Tier 2 expert weights (`3.78 MB` / **0.1118 bytes/param** measured in 90.79M MoE vs. `135.21 MB` AdamW and `6.68 MB` GaLore), $4.54$ bytes/param model-wide with standard Tier 3 AdamW ($1.27$ bytes/param with 8-bit AdamW), and $11.5\%$ ($122.4\text{ MB}$) separate-OS-subprocess peak PyTorch-allocated tensor memory reduction ($942.09 \pm 7.81\text{ MB}$ vs.\ $1,064.49 \pm 6.83\text{ MB}$ across $N=12$ fresh child processes).

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
  - Theorem 1 establishing bounded spatial memory complexity $\Theta(1)$ and constant per-step decode FLOPs $\Theta(H d_k d_v)$ for stabilized Vectorized Gated Linear Attention (GLA) (`48.0\text{ KiB}` isolated $H=6$ / `64.0\text{ KiB}` multi-head $H=8$ up to 16,384 tokens; $99.8\%$ reduction vs.\ Softmax SDPA, $4.02\times$ kernel prefill / $4.13\times$ decode at 16k; **$181,150\text{ tok/s}$ ($B=1$) and $297,753\text{ tok/s}$ ($B=4$) full 90.79M 16k prefill**, $157\text{ tok/s}$ decode), plus `fast_gla.py` bounded-decay `use_factored_intra` & 1-chunk OOM fix.
  - Proposition 1 establishing exact gradient equivalence of Streaming Chunked Vocabulary Cross-Entropy ($-98.5\%$ peak VRAM vs.\ PyTorch, $-94.1\%$ vs.\ `unsloth 2026.10.10` at $V=32,000, N=8,192$, grad diff $\le 4.47\times 10^{-8}$).
  - Theorem 2 & Theorem 3 proving asymptotic DMA bubble elimination ($\beta(K) \to 0$, $812,956\text{ tok/s}$) and Roofline operational bounds ($I_{\text{GLA}} \gg I^*$), plus Proposition 2 bounding activation-checkpointed `LayerStreamingEngine` VRAM ($\Theta(1)$ in layer count: $-76.43\%$ at $102\text{M}$, $-83.76\%$ and **$1.97\times$ WDDM crossover speedup** at $271\text{M}$, fitting **$695\text{M}$ ($2.12\text{ GB}$)** and **$1.52\text{B}+\text{CentroidSteer}$ ($3.13\text{ GB}$)** in $8\text{ GB}$ VRAM) alongside the HuggingFace bridge (`TriuneForCausalLM`, `WeightTiedExitProjection`).
  - Rescued Monotonic 90.79M Exits (`checkpoints/triune_studio_step_36500_rescued.pt`, `step_40555`: $111.80 \to 111.18 \to 110.92$ PPL), $4.31\times$ Reflex vs.\ Cortex speedup, $91.62\text{ tok/s}$ greedy ($51.28\%$ acc) & Pearson $\chi^2$-verified $90.74\text{ tok/s}$ rejection sampling ($T=0.7$, $63.64\%$ acc), conformal CALM calibration over $N=87,141$ cal / $85,503$ eval tokens (Reflex AUROC $0.8159$, Limbic AUROC $0.8319$, saving $20.35\%\text{--}41.11\%$ FLOPs while candidly reporting row-level compaction timing `0.294x–0.723x`), 4-tier quantization (`96 NVFP4Linear`: BF16 $97.504$ / FP8 $101.56$ / Unrotated NVFP4 $126.887$ / Hadamard W4A16 $119.106$ PPL), default sparse `MoE_FFN` (`5.450x` @ `d=1536, k=2`, `7.518x` @ `d=384`), $100.0\%$ std / overwrite MQAR on Gated DeltaNet, $2.65\times$ held-out lossless text compression (`expand` `0.0` diff + `order_pool`), $+6.7\%$ LoRA throughput vs.\ `unsloth 2026.10.10`, and $-99.0\%$ CUDA Graph host launch latency.

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
