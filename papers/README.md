# Lucid Machines (`v1.0.0`) — Research Manuscripts & Preprints

This directory contains research preprints, BibTeX bibliographies, vector figures, build/staging automation, and LaTeX manuscripts documenting the architectural, mathematical, and systems innovations introduced in **Lucid Machines `v1.0.0`** (**Triune** architecture `TriuneTransformer`, **Junkie** inaugural flagship model configurations, **CentroidSteer** optimizer, and **Polysom Studio** workbench).

> **Licensing**: Manuscripts and figures in `papers/` are licensed under **CC BY 4.0** (Creative Commons Attribution 4.0 International). Reference software, kernel implementations, and benchmark scripts in `lucid_machines/` and `papers/scripts/` are licensed under the **Apache License 2.0** (see root [`LICENSE`](../LICENSE)).
> **Contact**: Paper author correspondence: **[`founder@lucidmachines.co`](mailto:founder@lucidmachines.co)** · General inquiries & support: **[`contact@lucidmachines.co`](mailto:contact@lucidmachines.co)**
> **Publication Workflow**: See [`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md) for pre-flight hygiene gates, sequential arXiv submission order (Paper 1 -> Paper 2), peer-reviewed venue `.sty`/`.bst` migration, reproducibility alignment, and website metadata.

---

## Research Preprints (Working Manuscripts)

### 1. CentroidSteer: Activation-Guided Low-Rank Subspace Optimization on the Stiefel Manifold for Sparse Mixture-of-Experts
- **Track**: Machine Learning Optimization & Scalable Pre-Training
- **Author & Correspondence**: Yash Bhardwaj (`founder@lucidmachines.co`)
- **Manuscript & Bibliography**: [`paper1_centroidsteer/main.tex`](paper1_centroidsteer/main.tex), [`paper1_centroidsteer/references.bib`](paper1_centroidsteer/references.bib)
- **Compiled PDF**: [`paper1_centroidsteer/centroidsteer.pdf`](paper1_centroidsteer/centroidsteer.pdf) *(recompile with `tectonic` via `papers/scripts/build_papers.py --strict`)*
- **Primary Artifacts**:
  - Riemannian Geometry on the Stiefel Manifold $\operatorname{St}(r, d)$, including canonical tangent projections $\mathcal{P}_{T_Q}$, exact geodesic exponential mapping, polar Cholesky retraction $\mathcal{R}_Q^{\text{polar}}$, and Cayley transform retraction $\mathcal{R}_W^{\text{Cayley}}$.
  - **Exact Orthogonal Direct-Sum Decomposition** ($G = G_{\parallel C_e} \oplus G_{\perp C_e}$, max relative orthogonality error `4.50e-07` / `7.90e-07`, `<Delta_W_parallel, Delta_W_perp>_F = 0.00e+00`, Stiefel isometry `9.10e-07`) with **Uncentered Matrix-Free Block-Oja EMA** activation subspaces ($k_c \in [4, 8]$), true post-SwiGLU hidden centroids (`last_hidden_centroids`, $[E, H]$), **Grassmann Parallel Transport** across subspace rotations (continuous `steer_step` `10/10 uninterrupted rotations` with zero moment resets and **`15.13x`** activation-space step norm amplification), and **BFloat16 Stochastic Rounding**.
  - Theorem 1 proving upper-triangular Krylov projection core identity $\bar{S} = \bar{Q}_L^T G \bar{Q}_R = \begin{bmatrix} R_L & Q_L^T G \tilde{v}_\perp \\ \mathbf{0} & \tilde{c}_\perp^T G \tilde{v}_\perp \end{bmatrix}$, Proposition 1 establishing basis isometricity on $\operatorname{St}(r+k_c, d)$ with rank-deficient fallback, Theorem 2 proving **Second-Moment ANOVA Energy Conservation & Total Covariance Dispersion** ($\Sigma_{\text{total}} = \Sigma_{\text{between}} + \mathbb{E}[\Sigma_e]$), and Theorem 3 proving `sCholQR2` stability ($\|Q^T Q - I\|_F < 10^{-4}$, `83.76%` fast-path survival across $11,832$ live calls on the 8-layer `step_36500` Studio checkpoint) & non-convex ergodic convergence $\mathcal{O}(1/\sqrt{T})$.
  - **Controlled 5-Seed Scientific Baseline Validation (`[42, 1337, 2026, 777, 1024]` on RTX 5070, `< 0.6 B/p` Expert State, `benchmarks/centroidsteer_upgrade_results.json`)**:
    - **Suite 1 (Mechanistic Diagnostics)**: `87.36%` gradient energy capture vs. `4.27%` random (`20.48x`), `60.53%` activation variance capture (`18.16x`), continuous Grassmann transport across `10/10 uninterrupted rotations`, `15.13x` step amplification.
    - **Suite 2 (Multi-Domain MoE Specialization, `0.550 B/p`)**: Val loss **`1.1992 +/- 0.0444`** vs. `1.2243 +/- 0.0468` `No-Steer Low-Rank` (Wilcoxon 1-sided `p = 0.0312 < 0.05`, won `5/5` seeds) with `61.88%` subspace energy capture (`32.9x` vs. random, `p = 6.320e-07`).
    - **Suite 3 (Real FineWeb-Edu Cold-Start LM, 4-Layer Pure-Routed MoE, `0.264 B/p`)**: Val loss **`2.2012 +/- 0.0178`** (`9.04 +/- 0.16` PPL) vs. `2.2472 +/- 0.0069` (`9.46 +/- 0.07` PPL) **`No-Steer Low-Rank`** (paired $t$-test `p = 0.0112 < 0.05`, won `5/5` seeds) and `2.2585 +/- 0.0131` (`9.57 +/- 0.13` PPL) **`Memory-Matched Random Subspace`** (paired $t$-test `p = 0.0006 < 0.001`, won `5/5` seeds), capturing **`92.23%` gradient energy** (`51.24x` vs. random, `p = 2.126e-08`).
    - **Suite 4 (Real FineWeb-Edu 75.8M-Labeled / 76.00M Actual 8-Layer Checkpoint Continuation from `step_36500`, `0.201 B/p`)**: Val loss **`4.9934 +/- 0.0003`** (`147.43 +/- 0.04` PPL) vs. `4.9945 +/- 0.0001` `No-Steer Low-Rank` (`p = 0.0004`), `4.9945 +/- 0.0001` `Memory-Matched Random Subspace` (`p = 0.0005`), and `4.9954 +/- 0.0008` `Full-Rank AdamW` (`p = 0.0038`, won `5/5` seeds across all three baselines) with `70.54%` subspace energy capture (`57.82x` vs. random).
  - Exact memory accounting across scales: **0.128 B/p** analytical (`384 x 1024, r=16`), **0.1118 B/p** (`3.78 MB` vs. `135.21 MB` AdamW and `6.68 MB` GaLore in $N=8$ `90.79M` MoE), **0.1992 B/p** (`Junkie-Mini/1B/3B`), **0.3678 B/p** (`Junkie-7B`), and $11.5\%$ ($122.4\text{ MB}$) separate-OS-subprocess peak tensor memory reduction ($N=12$ fresh child processes; TOST equivalence within $\pm 0.01$ nats, $p_{\text{TOST}} = 1.52 \times 10^{-7}$).

```bibtex
@misc{bhardwaj2026centroidsteer,
  title={CentroidSteer: Activation-Guided Low-Rank Subspace Optimization on the Stiefel Manifold for Sparse Mixture-of-Experts},
  author={Bhardwaj, Yash},
  note={Lucid Machines Preprint (CC BY 4.0)},
  url={https://github.com/lucid-machines/triune},
  year={2026}
}
```

---

### 2. Triune: A Memory-Virtualizing and Constant-State Transformer Architecture for Long-Context Autoregressive Models
- **Track**: Efficient Deep Learning Systems & Hardware-Aware Machine Learning
- **Author & Correspondence**: Yash Bhardwaj (`founder@lucidmachines.co`)
- **Manuscript & Bibliography**: [`paper2_triune_systems/main.tex`](paper2_triune_systems/main.tex), [`paper2_triune_systems/references.bib`](paper2_triune_systems/references.bib)
- **Compiled PDF**: [`paper2_triune_systems/triune_architecture.pdf`](paper2_triune_systems/triune_architecture.pdf) *(recompile with `tectonic` via `papers/scripts/build_papers.py --strict`)*
- **Primary Artifacts**:
  - **Unified Lucid Machines `v1.0.0` Suite Framing**: Documents the **Triune** architecture (`TriuneTransformer` / `TriuneConfig`), inaugural **Junkie** flagship target configurations measured on `torch.device("meta")` (`Junkie-Mini` `535.10M`, `Junkie-1B` `1.16B`, `Junkie-3B` `2.99B` 16-expert `10.10 GiB` FP8 static Colab L4 sweet spot, and `Junkie-7B` `7.40B` — clearly separated from trained baselines), the `lucid_machines` Python SDK & `kcx` (**Polysom KCX**) CLI (`polysom` / `lucid-machines` aliases), and the **Polysom Studio** workbench.
  - Theorem 1 establishing bounded spatial memory complexity $\Theta(1)$ and constant per-step decode FLOPs $\Theta(H d_k d_v)$ for stabilized Vectorized Gated Linear Attention (GLA) (`48.0\text{ KiB}` isolated $H=6$ / `64.0\text{ KiB}` multi-head $H=8$ up to 16,384 tokens; $99.8\%$ reduction vs.\ Softmax SDPA, $4.02\times$ kernel prefill / $4.13\times$ decode at 16k; **$181,150\text{ tok/s}$ ($B=1$) and $297,753\text{ tok/s}$ ($B=4$) full 90.79M 16k prefill**, $157\text{ tok/s}$ sequential tensor decode, and **$40.8\text{ tok/s}$ warm streaming decode / $71.3\text{ ms}$ TTFT** on RTX 5070), plus `fast_gla.py` bounded-decay `use_factored_intra` & 1-chunk OOM fix.
  - Proposition 1 establishing exact gradient equivalence of Streaming Chunked Vocabulary Cross-Entropy ($-98.5\%$ peak VRAM vs.\ PyTorch, $-94.1\%$ vs.\ `unsloth 2026.10.10` at $V=32,000, N=8,192$, grad diff $\le 4.47\times 10^{-8}$).
  - Theorem 2 & Theorem 3 proving asymptotic DMA bubble elimination ($\beta(K) \to 0$, $812,956\text{ tok/s}$) and Roofline operational bounds ($I_{\text{GLA}} \gg I^*$), plus Proposition 2 bounding activation-checkpointed `LayerStreamingEngine` VRAM ($\Theta(1)$ in layer count: $-76.43\%$ at $102\text{M}$, $-83.76\%$ and **$1.97\times$ WDDM crossover speedup** at $271\text{M}$, fitting **$695\text{M}$ ($2.12\text{ GB}$)** and **$1.52\text{B}+\text{CentroidSteer}$ ($3,132.62\text{ MB}$ allocated / $6,248\text{ MB}$ reserved)** in $8\text{ GB}$ VRAM) alongside the HuggingFace bridge (`TriuneForCausalLM`, `WeightTiedExitProjection`).
  - Rescued Monotonic 90.79M Exits (`checkpoints/triune_studio_step_36500_rescued.pt`, `step_40555`: $111.80 \to 111.18 \to 110.92$ PPL), $4.31\times$ Reflex vs.\ Cortex speedup, $91.62\text{ tok/s}$ greedy ($51.28\%$ acc) & Pearson $\chi^2$-verified $90.74\text{ tok/s}$ rejection sampling ($T=0.7$, $63.64\%$ acc), conformal CALM calibration over $N=87,141$ cal / $85,503$ eval tokens (Reflex AUROC $0.8159$, Limbic AUROC $0.8319$, saving $20.35\%\text{--}41.11\%$ FLOPs while candidly reporting row-level compaction timing `0.294x–0.723x`), 4-tier quantization (`96 NVFP4Linear`: BF16 $97.504$ / FP8 $101.56$ / Unrotated NVFP4 $126.887$ / Hadamard W4A16 $119.106$ PPL), default sparse `MoE_FFN` (`5.450x` @ `d=1536, k=2`, `7.518x` @ `d=384`), $100.0\%$ std / overwrite MQAR on Gated DeltaNet, $2.65\times$ held-out lossless text compression (`expand` `0.0` diff + `order_pool`), $+6.7\%$ LoRA throughput vs.\ `unsloth 2026.10.10`, and $-99.0\%$ CUDA Graph host launch latency.

```bibtex
@misc{bhardwaj2026triune,
  title={Triune: A Memory-Virtualizing and Constant-State Transformer Architecture for Long-Context Autoregressive Models},
  author={Bhardwaj, Yash},
  note={Lucid Machines Preprint (CC BY 4.0)},
  url={https://github.com/lucid-machines/triune},
  year={2026}
}
```

---

## Reproduction, Build, & Local Staging Commands

To recompile figures, run strict PDF audits, sync `website/papers/`, and stage local arXiv source tarballs (no external upload):

```powershell
# 1. Regenerate publication vector figures from benchmarks/*.json
.\studio\studio_env\Scripts\python.exe papers/scripts/generate_figures.py

# 2. Strict build + log audit + clean intermediates + sync website/papers/
#    Flags: --paper {all,paper1,paper2} --with-figures --strict --max-overfull-pt 2.0 --keep-intermediates --sync-website
.\studio\studio_env\Scripts\python.exe papers/scripts/build_papers.py --strict --sync-website

# 3. Stage local-only self-contained arXiv source tarballs (papers/dist/arxiv/*.tar.gz) & verify self-compile
.\studio\studio_env\Scripts\python.exe papers/scripts/stage_arxiv.py
```
