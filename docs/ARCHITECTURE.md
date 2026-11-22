# Lucid Machines v1.0.0 — System Architecture Guide

**Lucid Machines** (`lucid_machines`) is organized into six modular layers spanning native C/AVX2/CUDA kernels, **the Triune architecture by Lucid Machines** (`TriuneTransformer` / `TriuneModel` / `TriuneConfig` 3-tier multi-exit Mixture-of-Experts architecture — substrate for our inaugural flagship model to train **Junkie**), Riemannian Stiefel optimizers, depth-invariant layer streaming, a 40-node visual DAG execution engine, and the **Polysom Studio v1.0.0** native desktop workbench.

---

## 1. Six-Layer System Hierarchy

| Layer | Modules & Binaries | Responsibility |
| :--- | :--- | :--- |
| **Layer 0 — Native & Kernels** | `csrc/engine/{api.h,engine.cpp}`, `csrc/build_native.py`, `lucid_machines/ops/`, `lucid_machines/kernels/` | Native C/AVX2 shared library (`lucid_machines_core.dll` / `liblucid_machines_core.so`, with `triune_core.*` fallback) with dynamic `cudart64` telemetry & `sCholQR2` helpers; Triton + PyTorch Tensor Core autograd kernels (`fast_gla`, `fast_triangular`, `fast_rmsnorm`, `fast_rope`, `fast_swiglu`, `fast_cross_entropy`, `fast_fp8`, `fast_nvfp4`, `fast_muon`, `fast_lora`). |
| **Layer 1 — Configs, Model & Data** | `lucid_machines/configs/`, `lucid_machines/model/`, `lucid_machines/data/` | `TriuneTransformer` 3-tier hierarchical architecture (**Reflex L2 / Limbic L5 / Cortex L8** early exits via conformal CALM), `VectorisedGLA` ($O(1)$ `48.0 KiB` isolated kernel), `MoE_FFN` (with contrastive input `last_centroids` `[E, D]` and true post-SwiGLU `last_hidden_centroids` `[E, H]`), `GumbelSoftmaxRouter`, 4-tier quantization (`BF16`/`FP8`/`NVFP4`/Hadamard W4A16), `DynamicTokenizer` (`2.65x` lossless compression, `expand` `0.0` max logit diff). |
| **Layer 2 — Optimizers, Runtime & Trainer** | `lucid_machines/optim/`, `lucid_machines/runtime/`, `lucid_machines/trainer/`, `lucid_machines/export/`, `universal.py` | 3-Tier `CentroidSteerOptimizer` (`Muon` 2.0 B/p + multi-directional `CentroidSteer` `0.1118–0.3678 B/p` + `8-bit AdamW`), `LayerStreamingEngine` (trains `1.52B` in `3,132.6 MB` VRAM), `TrainingEngine`, `FineTuner`, `hf_compat.py`, and `upgrade(model)`. |
| **Layer 3 — DAG, Plugins, Modules & Agents** | `lucid_machines/plugins/`, `lucid_machines/execution/dag.py`, `lucid_machines/modules/manager.py`, `lucid_machines/agents/` | 40 typed DAG nodes, Ed25519/SHA-256/AST-verified module manager (`trusted=False` default for external plugins), `PythonSandbox` with Windows Job Object / resource limits, and self-learning multi-agent harness. |
| **Layer 4 — Serving, CLI & Desktop Bridge** | `lucid_machines/api/{server.py,routes.py,srsm.py}`, `cli.py`, `desktop.py`, `inference.py` | 92 hardened FastAPI routes + `/ws/telemetry` + `/healthz`, unified `lucid-machines` & `polysom` CLI subcommands, proxy-aware bearer auth (`LUCID_MACHINES_API_TOKEN` / `TRIUNE_API_TOKEN`), Origin/CSRF/SSRF guards, and symlink-resolved path confinement. |
| **Layer 5 — Polysom Studio, Website & Release** | `studio/src/`, `studio/windows/`, `studio/installer/`, `website/`, `launch_kit/`, `dist/` | Zero-build vendored React 18 UMD Polysom Studio UI (6 themes: `Hypnagogic`, `Phosphene`, `Pareidolia`, `Afterimage`, `Mirage`, `Euphoria`; zero emojis, zero CSS gradients), `.NET 10` WPF/WebView2 Windows desktop app (`PolysomStudio.exe`) & installer (`PolysomStudioSetup.exe`), static project website (`website/`), and local launch kit (`launch_kit/`). |

---

## 2. 3-Tier Cognitive Stratum & Conformal CALM Routing

1. **Reflex Stratum (Early Depth, e.g., L1–L2)**:
   - Dense or shallow GLA + FFN blocks with a dedicated parameter-free `RMSNorm` and linear exit head.
   - Delivers **`4.31x` prefill speedup** over Cortex L8 (`741,070 tok/s` vs. `172,099 tok/s` at `B=4, T=512`).
   - Under held-out conformal CALM calibration (`benchmarks/calm_calibration.py`, $N=87,141$ calibration / $N=85,503$ evaluation tokens), achieves `AUROC = 0.8159` and **`95.73%` Cortex agreement** at softmax-margin threshold $\tau = 0.20$ (`31.34%` coverage, **`20.35%` transformer FLOPs saved**).
2. **Limbic Stratum (Mid Depth, e.g., L3–L5)**:
   - Sparse `MoE_FFN` blocks (`use_sparse_eval=True`, `1.648x` single-token zero-sync speedup and `7.518x` batch prefill speedup at `d=384, E=8, k=2, N=4096`) with an always-active shared expert.
   - Captures both contrastive input activation statistics (`last_centroids`, `[E, D]`) and true post-SwiGLU hidden activations (`last_hidden_centroids`, `[E, H]`) for `e[2]` down-projection steering.
   - Achieves `AUROC = 0.8319` for mid-depth CALM exit gating and serves as a high-acceptance draft source for self-speculative decoding.
3. **Cortex Stratum (Deep Terminal Depth, e.g., L6–L8)**:
   - Full-capacity recurrent `VectorisedGLA` / `Gated DeltaNet` + Sparse MoE reasoning layers.
   - Supports **100% token-for-token exact greedy self-speculative decoding** at `T=0.0` (`91.62 tok/s`, `51.28%` acceptance) and Pearson $\chi^2$-verified Leviathan et al. **stochastic rejection sampling** at `T=0.7` (`90.74 tok/s`, `63.64%` acceptance).

---

## 3. Multi-Directional CentroidSteer Architecture

`CentroidSteerOptimizer` (`lucid_machines/optim/centroid.py`) compresses Mixture-of-Experts optimizer states to `< 0.6 B/p` while outperforming scientific baselines (`No-Steer Low-Rank`, `Memory-Matched Random Subspace`, and `Full-Rank AdamW`):
- **Exact Orthogonal Direct-Sum ($G = G_{\parallel C_e} \oplus G_{\perp C_e}$) & Multi-Directional Subspaces ($k_c \in [4, 8]$)**: Block-Oja EMA activation subspaces and Grassmann parallel transport retain activation-aligned gradient energy (`87.36%` in Suite 1 diagnostics, `92.23%` in Suite 3 FineWeb-Edu LM, `70.54%` in Suite 4 75.8M continuation) while preserving the complementary Krylov residual subspace (`15.13x` activation-space step norm amplification).
- **True Post-SwiGLU Hidden Centroids (`last_hidden_centroids`)**: `e[2]` down-projections (`[H, D]`) steer along true post-SwiGLU hidden activations (`[E, H]`).
- **Centroid-Deflated Krylov, Batched `sCholQR2` & Sliced Householder Fallback**: Deflates $G$ along the activation subspace before Krylov iteration and maintains strict Stiefel manifold orthonormality ($\|Q^T Q - I\|_F < 10^{-4}$).
- **Independent `steer_step` Reset & BF16 Stochastic Rounding**: Resets `steer_step` whenever steering axes refresh, zeroes rotated moment coordinates (`r+1`), and preserves steering scale via Fisher alignment weighting.
- **Sub-Byte Triangular State & 5-Seed Scientific Baseline Validation**: Compresses Tier-2 expert optimizer states to `0.201–0.550 B/p` across validation suites (vs. `8.000 B/p` FP32 AdamW; `0.1118 B/p` / `3.78 MB` on 90.79M vs. `135.21 MB` FP32 AdamW — **98.4% reduction**; `0.1992 B/p` on `Junkie-Mini` / `Junkie-1B` / `Junkie-3B`; `0.3678 B/p` on `Junkie-7B`), achieving statistically significant FineWeb-Edu validation loss improvements (`Suite 3` cold-start LM at `0.264 B/p`: `2.2012` vs. `2.2472` `nosteer`, `p = 0.0112`, and `2.2585` `random_same_mem`, `p = 0.0006`; `Suite 4` 75.8M continuation at `0.201 B/p`: `4.9934` vs. `4.9945` `nosteer`, `p = 0.0004`, `4.9945` `random_same_mem`, `p = 0.0005`, and `4.9954` `adamw`, `p = 0.0038`) across 5 seeds (`[42, 1337, 2026, 777, 1024]`).

---

## 4. R4 Security Boundaries

- **Safe Deserialization**: `torch.load(..., weights_only=True)` enforced across `finetune.py`, `rescue_checkpoint.py`, `verify_triangular_qr_residuals.py`, and API checkpoint loaders.
- **Path Confinement**: Every `/v1/models/*`, `/v1/checkpoints/*`, `/v1/datasets/*`, `/v1/finetune/*`, and DAG file operation resolves symlinks (`Path.resolve(strict=False)`) and asserts `is_relative_to(...)` against workspace roots.
- **Network, Proxy Auth, CSRF & SSRF**: Non-loopback requests and any request carrying `Forwarded`, `X-Forwarded-For`, or `X-Real-IP` headers require a valid `LUCID_MACHINES_API_TOKEN` / `TRIUNE_API_TOKEN` bearer token; state-changing routes and `/ws/telemetry` validate `Origin`/`Host`; URL loaders enforce `validate_http_url`.
- **Cryptographic Plugins & Sandbox**: External plugin discovery defaults to `trusted=False` (Ed25519 signature required); vLLM defaults to `trust_remote_code=False`; Gemini keys transmit via `x-goog-api-key` header only; `PythonSandbox` enforces strict subprocess memory and time limits.
