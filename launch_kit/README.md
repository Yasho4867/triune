# Lucid Machines `v1.0.0` Local Launch & Advertising Kit (`R5`)

> **SAFETY & COMPLIANCE NOTICE (R5 Guardrail):**  
> All documents and visual cards in `launch_kit/` are **local-only release marketing and developer messaging assets** for Lucid Machines' inaugural **`v1.0.0`** public release. Zero automated or live posts have been executed on any external social media, forum, or advertising account. All quantitative metrics strictly adhere to the **5-Tier Benchmark Taxonomy** defined in `CONTRIBUTING.md §1` and the **5-Seed CentroidSteer Scientific Baseline Validation Report** (`benchmarks/CENTROIDSTEER_UPGRADE_VALIDATION_REPORT.md`).

---

## Kit Inventory

| File | Purpose | Target Audience / Channel |
| :--- | :--- | :--- |
| [`README.md`](README.md) | Kit index, empirical claims policy, and asset manifest | Release Engineering & Marketing Review |
| [`HERO_MESSAGING.md`](HERO_MESSAGING.md) | Brand positioning (`Lucid Machines` company, `Triune` architecture `TriuneTransformer`, inaugural flagship model `Junkie`, `Polysom Studio v1.0.0` workbench), elevator pitches, and landing/README hero copy | GitHub README, Website Hero, Product Hunt, Press |
| [`LAUNCH_BLOG.md`](LAUNCH_BLOG.md) | Long-form technical launch blog covering `v1.0.0` architecture, multi-directional CentroidSteer 5-seed validation vs. scientific baselines, 3-Tier CALM MoE, `Junkie-3B` Colab L4 FP8 sweet spot, NVFP4 streaming, and R4 security hardening | Engineering Blog, Medium, Substack, Dev.to |
| [`SOCIAL_POSTS.md`](SOCIAL_POSTS.md) | Ready-to-review local launch threads and community posts | X (Twitter), LinkedIn, Hacker News (`Show HN`), Reddit (`r/LocalLLaMA`, `r/MachineLearning`) |
| [`AD_COPY_VARIANTS.md`](AD_COPY_VARIANTS.md) | Paid search, developer newsletter sponsorships, and social card copy variants | Google/Bing Developer Search, GitHub Sponsors/Newsletters, Technical Social |
| [`benchmark-card.svg`](benchmark-card.svg) | `1200x630` OpenGraph / Social Benchmark Summary Card (solid dark styling, zero gradients/emojis) | Social OpenGraph preview & blog figures |
| [`architecture-card.svg`](architecture-card.svg) | `960x360` 3-Tier CALM & LayerStreaming Architecture Card | Technical threads & documentation embeds |

---

## Empirical Integrity & 5-Tier Taxonomy Checklist

Every claim across `launch_kit/` is cross-checked against hardware-verified repository artifacts (`NVIDIA GeForce RTX 5070 Laptop GPU`):
1. **Tier 1 (Isolated Kernel Microbenchmarks)**:
   - `VectorisedGLA`: `48.0 KiB` isolated recurrent state at `H=6, L=16,384` (`10.07x` prefill / `7.31x` decode vs. SDPA).
   - `MoE_FFN(use_sparse_eval=True)`: `1.648x` single-token zero-sync speedup (`N=1, d=384`, `0.0` max diff) and `7.518x` batch prefill speedup (`N=4096, d=384, E=8, k=2`).
   - `chunked_cross_entropy`: `98.5%` transient VRAM reduction vs. PyTorch and `94.1%` vs. Unsloth (`2026.10.10`) at `V=32,000, N=8,192`.
   - `FastSwiGLU`: `1.95x` speedup vs. eager PyTorch.
2. **Tier 2 (End-to-End 90.79M Baseline Model Throughput — `64.21M` active & Runtime Inference)**:
   - Peak E2E prefill: `383,135 tok/s` (`B=32, T=1024`); standard prefill: `200,463 tok/s` (`B=1, T=2048`), `242,029 tok/s` (`B=1, T=8192`), `297,753 tok/s` (`B=4, L=16384`).
   - Sequential tensor decode: `157.0 tok/s` (`6.363 ms/tok`); runtime streaming inference: **`40.8 tok/s` warm decode / `71.3 ms` TTFT** on RTX 5070.
   - LoRA fine-tuning: `25,973 tok/s` (`+6.7%` vs. genuine Unsloth `24,344 tok/s`).
   - `LayerStreamingEngine`: Cuts `102M` VRAM by `76.43%` (`1,033.6 MB`), `271M` by `83.76%` (`1,498.1 MB`, `1.97x` faster than WDDM paging), and trains `1.52B + CentroidSteer` in `3,132.6 MB` VRAM on an `8 GB` GPU.
3. **Tier 3 (Multi-Layer Sweeps vs. Legacy Diagnostics)**:
   - Multi-layer long-context wrapper (`L=16,384`): `64.0 KiB` cache (`99.8%` reduction vs. SDPA), `4.02x` prefill / `4.13x` decode.
   - Production `MoE_FFN` (`1.648x–7.518x`) is explicitly distinguished from the legacy diagnostic `MoERoutingBlock` (`0.47x`).
4. **Tier 4 (Tokenizer Compression vs. Model Computation)**:
   - Held-out text compression: `2.65x` aggregate lossless compression (`100%` reconstruction).
   - Compute-equivalent expansion (`compress_mode="expand"`): `0.0` max logit diff vs. base tokenization.
5. **Tier 5 (Speculative & Conformal CALM Early Exits)**:
   - `T=0.0` greedy self-speculative decode: `91.62 tok/s` (`51.28%` acceptance, 100% exact Cortex parity).
   - `T=0.7` Leviathan rejection sampling: `90.74 tok/s` (`63.64%` acceptance, Pearson $\chi^2$ verified).
   - Held-out conformal CALM (`N=87,141` cal / `N=85,503` eval): Reflex AUROC `0.8159`, Limbic AUROC `0.8319`, `95.73%` Reflex agreement at $\tau=0.20` (`31.34%` coverage, `20.35%` transformer FLOPs saved).
6. **CentroidSteer 5-Seed Scientific Baseline Validation (`N=5` Seeds: `42, 1337, 2026, 777, 1024`)**:
   - Evaluated against scientific baselines **`No-Steer Low-Rank` (`nosteer`)**, **`Memory-Matched Random Subspace` (`random_same_mem`)**, and **`Full-Rank AdamW` (`adamw`)**.
   - Mechanistic Diagnostics (Suite 1): Direct-sum relative error `4.50e-07` (`W_1`) / `7.90e-07` (`W_2`), `<Delta_W_parallel, Delta_W_perp>_F = 0.00e+00`, **`87.36%` gradient energy capture** vs `4.27%` random (`20.48x`), and **`15.13x`** activation-space step norm amplification.
   - Multi-Domain MoE (Suite 2, `0.550 B/p`): **`1.1992 ± 0.0444` vs `1.2243 ± 0.0468`** `nosteer` val loss (Wilcoxon 1-sided `p = 0.0312 < 0.05`, won `5/5` seeds) and **`61.88%` subspace energy capture** (`32.9x` vs random `0.0188`, `p = 6.320e-07`).
   - Real FineWeb-Edu Cold-Start LM (Suite 3, 4-Layer Pure-Routed MoE, `0.264 B/p`): **`2.2012 ± 0.0178`** val loss (`9.04 ± 0.16` PPL) vs **`2.2472 ± 0.0069`** (`9.46 ± 0.07` PPL) `nosteer` (**`paired t-test p = 0.0112 < 0.05`**, won `5/5` seeds) and **`2.2585 ± 0.0131`** (`9.57 ± 0.13` PPL) `random_same_mem` (**`paired t-test p = 0.0006 < 0.001`**, won `5/5` seeds), capturing **`92.23%` gradient energy** (`51.24x` vs random, `p = 2.126e-08`) at **`0.264 B/p`** (`96.70%` reduction vs `8.0 B/p` FP32 AdamW).
   - Real FineWeb-Edu 75.8M Checkpoint Continuation (Suite 4, `0.201 B/p`): **`4.9934 ± 0.0003`** val loss (`147.43` PPL) vs **`4.9945 ± 0.0001`** `nosteer` (`p = 0.0004`), **`4.9945 ± 0.0001`** `random_same_mem` (`p = 0.0005`), and **`4.9954 ± 0.0008`** `adamw` (`p = 0.0038`, won `5/5` seeds across all three baselines) with **`70.54%` subspace energy capture** (`57.82x` vs random) while preserving $\|Q^T Q - I\|_F < 10^{-4}$ (`0.1118 B/p` / `3.78 MB` vs `135.21 MB` on `90.79M` Tier-2, **98.4% reduction**).
   - **Junkie** (`junkie-mini` `535.10M`, `junkie-1b` `1.16B`, `junkie-3b` `2.99B` 16-expert Colab L4 FP8 sweet spot, and `junkie-7b` / `junkie` `7.40B` flagship) is Lucid Machines' inaugural flagship model prepared for training on this validated substrate (no fabricated large-scale training results claimed).
