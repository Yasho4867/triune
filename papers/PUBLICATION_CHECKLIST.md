# Lucid Machines (`v1.0.0`) — Publication Readiness Checklist

> **SAFETY BANNER — NO UNAUTHORIZED EXTERNAL SUBMISSION**
> Do **NOT** submit manuscripts to arXiv, OpenReview, conferences, workshops, Zenodo, or email, and do **NOT** execute `git push` or modify live release/website deployments without explicit user confirmation. All build, verification, and tarball staging workflows in `papers/` are strictly local until authorized.

---

## Gate 1 — Local Pre-Flight & Hygiene Gates

Run all three local commands from the repository root (`c:\Users\yashb_f1ls\OneDrive\Documents\TriuneTransformer`):

```powershell
.\studio\studio_env\Scripts\python.exe papers/scripts/generate_figures.py
.\studio\studio_env\Scripts\python.exe papers/scripts/build_papers.py --strict --sync-website
.\studio\studio_env\Scripts\python.exe papers/scripts/stage_arxiv.py
```

Verify every gate criterion passes:
- [ ] **Zero compiler/log warnings**: `0` `Overfull \hbox > 2.0pt`, `0` `Package hyperref Warning`, `0` `Invalid UTF-8` warnings, `0` `Object .* already defined` warnings, `0` undefined references/citations across both `paper1_centroidsteer` (14 pages) and `paper2_triune_systems` (15 pages).
- [ ] **Zero source/PDF leaks**: Zero raw Markdown `**...**` or backtick `` `...` `` leaks, zero stale `triune` shim claims, zero `10..10` shorthands, zero `6-layer` mislabelings of the 8-layer `step_36500` Studio checkpoint, and zero `??` unresolved reference markers in extracted PDF text.
- [ ] **Clean intermediates**: Zero `.aux`, `.out`, `.bbl`, `.blg`, `.log`, or `.xdv` files left inside `papers/`.
- [ ] **Metadata & Disclosures**:
  - Verify author correspondence email is set to `founder@lucidmachines.co` in `\author{...}` (`papers/paper1_centroidsteer/main.tex`, `papers/paper2_triune_systems/main.tex`) and general support/public contact is set to `contact@lucidmachines.co` (`README.md`, `papers/README.md`).
  - Verify split licensing notice is intact: **CC BY 4.0** (manuscripts & figures) and **Apache 2.0** (code & benchmark drivers in `lucid_machines/`, matching root `LICENSE`).
  - Verify `Ethics, Broader Impact, & Compute Disclosure` section is intact (`NVIDIA GeForce RTX 5070 Laptop GPU, 7.93 GiB VRAM`, `<2` GPU-hours / sub-hour microbenchmarks + `61.04 s` KL rescue, public `HuggingFaceFW/fineweb-edu` `sample-10BT` provenance, zero PII/human-subjects collection).
- [ ] **Staged arXiv tarballs**: `papers/dist/arxiv/paper1_centroidsteer-arxiv.tar.gz` and `papers/dist/arxiv/paper2_triune_systems-arxiv.tar.gz` (gitignored via `.gitignore:20:dist/`) contain ONLY `main.tex` (with generated `.bbl` bibliography inlined for `pdflatex` + Tectonic compatibility) and referenced `figures/*.pdf`, with self-compile verification passing at exit `0`.
- [ ] **Website artifact sync**: `website/papers/{centroidsteer.pdf, triune_architecture.pdf, paper1_centroidsteer.tex, paper2_triune_systems.tex}` match `papers/` SHA-256 hashes identically.

---

## Gate 2 — Sequential arXiv Preprint Submission (Order Matters)

Submit **Paper 1 before Paper 2** so Paper 2 can cite Paper 1's real arXiv identifier instead of a preprint placeholder:

1. **Step 2A — Submit Paper 1 (`CentroidSteer`)**:
   - Archive: `papers/dist/arxiv/paper1_centroidsteer-arxiv.tar.gz`
   - Primary Category: `cs.LG` (Machine Learning); Cross-list: `math.OC` (Optimization and Control)
   - License: `CC BY 4.0`
   - Action: Upload tarball, preview compiled PDF in arXiv's AutoTeX staging viewer (confirm all 14 pages, equations, tables, and 5 figures render cleanly), then finalize after user confirmation. Record assigned `arXiv:26xx.xxxxx` ID.
2. **Step 2B — Backfill Paper 1 ID into Paper 2**:
   - Update `@misc{bhardwaj2026centroidsteer, ...}` in `papers/paper2_triune_systems/references.bib` (`note = {arXiv preprint arXiv:26xx.xxxxx}` and `eprint = {26xx.xxxxx}`).
   - Re-run `build_papers.py --strict --sync-website` and `stage_arxiv.py` to regenerate `triune_architecture.pdf` and `paper2_triune_systems-arxiv.tar.gz`.
3. **Step 2C — Submit Paper 2 (`Triune`)**:
   - Archive: `papers/dist/arxiv/paper2_triune_systems-arxiv.tar.gz`
   - Primary Category: `cs.LG` (Machine Learning); Cross-list: `cs.CL` (Computation and Language), `cs.DC` (Distributed, Parallel, and Cluster Computing)
   - License: `CC BY 4.0`
   - Action: Preview all 15 pages and 5 figures in arXiv AutoTeX viewer, finalize after user confirmation, and record assigned `arXiv:26xx.yyyyy` ID.
4. **Step 2D — Post-arXiv Repo & Website Sync**:
   - Update BibTeX blocks in `papers/README.md` and paper links in `website/src/App.tsx` with both real arXiv IDs.

---

## Gate 3 — Peer-Reviewed Venue Targets & `.sty`/`.bst` Migration Steps

### Paper 1 (`CentroidSteer`)
- **Target Venues**:
  - Main conferences: **ICML**, **NeurIPS**, **ICLR**
  - Workshops: **OPT** (Optimization for Machine Learning), **WANT** (Workshop on Advancing Neural Network Training), Sparsity / Mixture-of-Experts workshops
- **Template Migration**:
  1. Replace `\documentclass[10pt,twocolumn,letterpaper]{article}` and `geometry` with target venue style (`icml2026.sty`, `neurips_2026.sty`, or `iclr2027_conference.sty`).
  2. Switch `\usepackage{cite}` + `\bibliographystyle{IEEEtran}` to `natbib` (`\citep`/`\citet`) + venue `.bst` (`icml2026.bst`, `plainnat`, `iclr2027_conference.bst`) pointing directly to `papers/paper1_centroidsteer/references.bib`.
  3. **Double-blind anonymization**: Remove/redact `Yash Bhardwaj`, `Lucid Machines`, `founder@lucidmachines.co`, `lucidmachines.co`, `github.com/lucid-machines/triune`, and identifying `Junkie` / `lucid_machines` paths during anonymous review (restore for camera-ready).
  4. Verify main-text page budget (typically 8–9 pages; move secondary proof details or per-seed tables to Appendix).

### Paper 2 (`Triune`)
- **Target Venues**:
  - Main conferences: **MLSys**, **EuroSys**, **ICLR** / **NeurIPS** (Efficient Systems tracks)
  - Workshops: **ES-FoMo**, **ENLSP**, Edge-LLM / Hardware-Aware ML workshops
- **Template Migration**:
  1. Replace preamble with `mlsys2026.sty`, `acmart` (EuroSys), or `neurips_2026.sty`.
  2. Switch bibliography to venue `.bst` using `papers/paper2_triune_systems/references.bib`.
  3. **Double-blind anonymization**: Redact author, affiliation, URLs, and identifying `Polysom Studio` / `Junkie` / repo paths for blind submission; verify page limits and move extended tables/proofs to Appendix if required.

---

## Gate 4 — Reproducibility & Artifact Release Alignment

Ensure the `v1.0.0` release tag (`pyproject.toml`, `dist/SHA256SUMS.txt`) aligns with all cited experimental artifacts:
- **Core Benchmark JSONs (`benchmarks/`)**:
  - Paper 1: `centroidsteer_upgrade_results.json`, `CENTROIDSTEER_UPGRADE_VALIDATION_REPORT.md`, `optimizer_benchmark_results.json`, `convergence_100_steps_results.json`, `moe_convergence_100_steps_results.json`, `triangular_qr_residual_results.json`
  - Paper 2: `checkpoint_rescue_results.json`, `calm_calibration.json`, `full_model_e2e.json`, `layer_streaming_results.json`, `genuine_unsloth_vs_triune_results.json`, `benchmark_results.json`, `long_context_results.json`, `parallelism_benchmark_results.json`, `moe_ffn_sparse_vs_dense.json`, `mqar_associative_recall_results.json`
- **Deterministic Seeds & Checkpoints**:
  - Multi-seed validation: `[42, 1337, 2026, 777, 1024]` (5-seed Suites 1–4); `[42, 1337, 2026, 7, 99, 314, 777, 1024]` (8-seed MoE & ablation); 12-seed TOST Latin squares.
  - Preserved checkpoints: `checkpoints/triune_studio_step_36500_rescued.pt` (`step_40555`), `checkpoints/session_state.json`.
- **Verified Software & Hardware Environment**:
  - `NVIDIA GeForce RTX 5070 Laptop GPU` (`7.93 GiB` VRAM, CC `12.0`), `Python 3.14.7`, `PyTorch 2.14.1+cu132`, `CUDA 13.2`.
- **Archival DOI (Optional)**: Mint a Zenodo archival DOI tied to the tagged GitHub Release `v1.0.0` after external authorization.

---

## Gate 5 — M5 -> M6 Website Metadata Contract (`website/src/` & `website/public/papers/`)

M6 copies the rebuilt PDFs from `website/papers/` into `website/public/papers/` and surfaces both papers in `website/src/App.tsx`:

### Official Lucid Machines Contact Emails
- **Paper Author Contact / Manuscript Correspondence**: `founder@lucidmachines.co`
- **General Support / Public Contact (Website & Installers)**: `contact@lucidmachines.co`

### Synced Files (`website/papers/`)
- `website/papers/centroidsteer.pdf` (`papers/paper1_centroidsteer/centroidsteer.pdf`)
- `website/papers/triune_architecture.pdf` (`papers/paper2_triune_systems/triune_architecture.pdf`)
- `website/papers/paper1_centroidsteer.tex` (`papers/paper1_centroidsteer/main.tex`)
- `website/papers/paper2_triune_systems.tex` (`papers/paper2_triune_systems/main.tex`)

### Paper 1 Metadata
- **Title**: `CentroidSteer: Activation-Guided Low-Rank Subspace Optimization on the Stiefel Manifold for Sparse Mixture-of-Experts`
- **Author**: `Yash Bhardwaj (Lucid Machines)`
- **Author Contact**: `founder@lucidmachines.co`
- **Track**: `Machine Learning Optimization & Scalable Pre-Training`
- **License**: `CC BY 4.0 (manuscript & figures) / Apache 2.0 (code)`
- **PDF Filename**: `centroidsteer.pdf` | **Source `.tex`**: `paper1_centroidsteer.tex` | **Pages**: `14`
- **Key Contributions**:
  - Exact Orthogonal Direct-Sum (`G = G_{\parallel C_e} \oplus G_{\perp C_e}`) + Uncentered Block-Oja EMA (`87.36%`–`92.23%` gradient energy capture)
  - Grassmann Parallel Transport (`10/10` uninterrupted rotations, `15.13x` step amplification) + BFloat16 Stochastic Rounding
  - Two-pass shifted Cholesky-QR (`sCholQR2`) stability, augmented Krylov triangularization, and exact second-moment ANOVA conservation
  - `<0.6 B/p` 5-seed FineWeb-Edu wins (`p = 0.0112` vs No-Steer, `p = 0.0006` vs Random, `5/5` seeds) and `0.1118 B/p` (`3.78 MB`) on `90.79M` MoE (`-98.4%` vs FP32 AdamW)
- **Abstract**:
  > First-order adaptive optimizers such as AdamW maintain persistent first and second gradient moments that consume 8.0 bytes of high-bandwidth memory (HBM) per parameter in FP32 (4.0 B/p in BF16), constituting a primary bottleneck in scaling modern Mixture-of-Experts (MoE) models. While low-rank gradient projection methods mitigate optimizer memory, standard single-sided projections ignore the dynamic activation geometry of routed expert tokens and suffer moment resets when projection bases rotate. In this paper, we present CentroidSteer (Lucid Machines v1.0.0), an activation-guided, tri-tier partitioned optimizer operating on the compact Stiefel and Grassmann manifolds for sparse MoE architectures. CentroidSteer partitions parameters into: (1) Tier 1 (Muon), applying 5th-order Newton-Schulz polar orthogonalization to 2D attention and shared-expert matrices; (2) Tier 2 (CentroidSteer), decomposing expert gradients via an Exact Orthogonal Direct-Sum combining an uncentered matrix-free Block-Oja EMA activation subspace (including true post-SwiGLU hidden centroids for down-projections) with a complementary two-sided Krylov core retracted onto St(r, d) via two-pass shifted Cholesky-QR (sCholQR2), transporting preconditioned moments across rotations via Grassmann parallel transport with unbiased BF16 stochastic rounding; and (3) Tier 3 (Base AdamW / 8-bit AdamW), preserving 1D normalizations, router gates, and vocabulary embeddings. Across controlled N=5 matched-seed evaluations on an NVIDIA RTX 5070 against scientific baselines at <0.6 B/p persistent expert state, CentroidSteer captures 87.36% of gradient energy in mechanistic diagnostics (20.48x vs. random), outperforms No-Steer Low-Rank on Multi-Domain MoE (1.1992 vs. 1.2243, p=0.0312), achieves statistically significant validation gains on FineWeb-Edu cold-start LM pretraining (2.2012 vs. 2.2472 No-Steer, p=0.0112; vs. 2.2585 Random, p=0.0006; 92.23% gradient capture at 0.264 B/p, winning 5/5 seeds), and improves 75.8M checkpoint continuation against all three baselines (p <= 0.0038), while cutting 90.79M Tier-2 expert state by 98.4% vs. FP32 AdamW to 0.1118 B/p and preserving N=12 continuation equivalence within +/-0.01 nats (p_TOST = 1.52e-7).
- **BibTeX**:
```bibtex
@misc{bhardwaj2026centroidsteer,
  title={CentroidSteer: Activation-Guided Low-Rank Subspace Optimization on the Stiefel Manifold for Sparse Mixture-of-Experts},
  author={Bhardwaj, Yash},
  note={Lucid Machines Preprint (CC BY 4.0)},
  url={https://github.com/lucid-machines/triune},
  year={2026}
}
```

### Paper 2 Metadata
- **Title**: `Triune: A Memory-Virtualizing and Constant-State Transformer Architecture for Long-Context Autoregressive Models`
- **Author**: `Yash Bhardwaj (Lucid Machines)`
- **Author Contact**: `founder@lucidmachines.co`
- **Track**: `Efficient Deep Learning Systems & Hardware-Aware Machine Learning`
- **License**: `CC BY 4.0 (manuscript & figures) / Apache 2.0 (code)`
- **PDF Filename**: `triune_architecture.pdf` | **Source `.tex`**: `paper2_triune_systems.tex` | **Pages**: `15`
- **Key Contributions**:
  - Constant-state GLA (`48–64 KiB` state, `99.8%` KV reduction up to `16,384` tokens, `181k`/`298k tok/s` 16k prefill) & `100.0%` key-overwrite MQAR Gated DeltaNet
  - Conformal CALM (`85,503` eval tokens, `20.35%–41.11%` FLOPs saved) & Self-Speculative Decoding (`91.62`/`90.74 tok/s`, monotonic `step_40555` exits)
  - Sparse `MoE_FFN` (`5.45x`–`7.52x`), Chunked Vocabulary Cross-Entropy (`-98.5%` VRAM vs PyTorch, `-94.1%` vs Unsloth), `+6.7%` LoRA throughput vs Unsloth
  - Activation-checkpointed `LayerStreamingEngine` (`1.52B + CentroidSteer` inside `8 GB` VRAM) & 4-tier FP8/NVFP4/Hadamard W4A16 quantization
- **Abstract**:
  > Standard transformer architectures encounter two critical memory scaling limits: during autoregressive generation, softmax attention incurs linear O(L) Key-Value (KV) cache expansion, which consumes gigabytes of high-bandwidth memory (HBM) and degrades token-generation throughput at long contexts; during pre-training and fine-tuning on consumer hardware, full-model parameter storage, activation graphs, transient vocabulary logit materialization in cross-entropy loss, and host-device pipeline stalls strictly cap trainability inside 8 GB VRAM. In this paper, we present Triune as the core neural architecture of Lucid Machines v1.0.0 (lucid_machines Python framework, kcx CLI with polysom/lucid-machines compatibility aliases, Polysom Studio desktop workbench, and the inaugural Junkie target model family Junkie-Mini/1B/3B/7B sized on torch.device("meta")). Triune integrates: (1) Vectorized Gated Linear Attention (GLA) and Gated DeltaNet with numerically stabilized chunked Tensor Core Triton scans, maintaining a strictly bounded Theta(1) state footprint (48.0 KiB single-layer / 64.0 KiB multi-head up to 16,384 tokens, 99.8% cache reduction vs. Softmax SDPA) while reaching 4.40M tok/s kernel prefill (4.02x vs. SDPA) and 4.13x decode speedup at L=16,384; (2) Hierarchical Dynamic Exits, Conformal CALM Calibration, Self-Speculative Decoding, and 4-Tier Quantization on a KL-rescued monotonic 90.79M checkpoint (step_40555, 111.80 -> 111.18 -> 110.92 PPL); (3) Default Sparse Scatter-Gather MoE (MoE_FFN) achieving up to 7.518x speedup; (4) Chunked Vocabulary-Projection Cross-Entropy reducing peak loss VRAM by 98.5% vs. PyTorch (94.1% vs. Unsloth) and reaching 6.7% higher LoRA throughput than Unsloth; and (5) Activation-Checkpointed LayerStreamingEngine fitting 695M and 1.52B + CentroidSteer pre-training inside an 8 GB laptop GPU. On the complete 90.79M checkpoint, Triune sustains 181,150 tok/s (B=1) and 297,753 tok/s (B=4) full-model prefill at L=16,384 and 157 tok/s sequential tensor decode.
- **BibTeX**:
```bibtex
@misc{bhardwaj2026triune,
  title={Triune: A Memory-Virtualizing and Constant-State Transformer Architecture for Long-Context Autoregressive Models},
  author={Bhardwaj, Yash},
  note={Lucid Machines Preprint (CC BY 4.0)},
  url={https://github.com/lucid-machines/triune},
  year={2026}
}
```
