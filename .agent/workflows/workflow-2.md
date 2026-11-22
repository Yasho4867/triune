---
description: Workflow 2 — High-Efficiency Token Maximization & Strict Project Rules Compliance Workflow for Lucid Machines v1.0.0
---

# Workflow 2: High-Efficiency Token & Rules Compliance Workflow (`Lucid Machines v1.0.0`)

This workflow governs all agentic engineering, multi-agent swarm execution, benchmark reporting, UI/Studio updates, and release verification in the **Lucid Machines** (`TriuneTransformer` / `Junkie` / `Polysom Studio`) repository. Every agent and contributor MUST follow both pillars below: **(A) Token Maximization & Anti-Wastage** and **(B) Strict Enforcement of Project Rules & Conventions**.

---

## Pillar A: Token Maximization & Anti-Wastage Protocol

### A1. Targeted Line-Range Reads Instead of Full-File Dumps
- **Never dump large files into context.** Files such as `studio/src/app.js` (~800 KB), `studio/src/style.css` (~220 KB), `lucid_machines/api/routes.py` (~180 KB), `lucid_machines/optim/centroid.py` (~110 KB), and `lucid_machines/model/moe.py` (~45 KB) consume massive context budgets if read end-to-end.
- **Locate first, slice second:**
  1. Use `git grep -n "<symbol_or_selector>" <path>` or a fast search query to identify exact line numbers.
  2. Read only a tight window (`StartLine` / `EndLine` of 40–120 lines) around the target region.
  3. Apply surgical `edit_file` replacements rather than rewriting entire large files.

### A2. Disjoint Parallel File Ownership in Multi-Agent Swarms
- **Zero overlapping write targets:** When decomposing work across parallel swarm agents, assign explicit, mutually exclusive file ownership lists in every agent brief (e.g., Agent A owns `lucid_machines/optim/centroid.py` and `benchmarks/validate_centroid_steer_upgrade.py`; Agent B owns `models/configs/junkie-*.json`, `cloud/*`, and `.agent/workflows/workflow-2.md`).
- **Staged barrier synchronization:** Downstream integration agents (e.g., `release_builder` or `verifier`) must sleep via `swarm_wait` until upstream parallel workers post `[ready]` on the board, eliminating merge collisions and wasted duplicate turns.

### A3. Sparse, Event-Driven Monitoring (Zero Busy-Loop Polling)
- **Never poll background tasks or message boards in a tight loop.** Each polling turn burns full context tokens.
- For swarm coordination, always call `swarm_wait(since=<last_seen_seq>)`, which consumes **zero tokens** while parked until a relevant `@mention` or `@all` arrives.
- For long-running commands (multi-seed benchmarks, `.NET 10` builds, wheel builds), either continue independent local work in the meantime or wait for the system task-completion wakeup notification.
- Keep board messages concise, factual, and refer to prior messages by sequence number (`#131`) rather than re-quoting long text blocks.

### A4. Avoiding Redundant Rebuilds, Reruns & Full-Tree Scans
- **Exclude heavy directories from recursive scans:** Never run `Get-ChildItem -Recurse` or unguarded `grep -R` across `studio/studio_env/`, `studio/publish/`, `website/node_modules/`, `.git/`, or `checkpoints/`. Always scope searches to source directories (`lucid_machines/`, `triune/`, `studio/src/`, `studio/windows/`, `tests/`, `docs/`, `cloud/`, `launch_kit/`, `models/`).
- **Run targeted tests during iteration, full suite at the gate:** While iterating on a specific subsystem, run only the corresponding test module (e.g., `pytest tests/test_cloud_junkie_kit.py -q`), and reserve the full suite and binary packaging (`csrc/build_native.py`, `dotnet publish`, `python -m build`) for the final pre-gate pass.
- **Use `torch.device("meta")` for parameter & memory audits:** Verify billion-scale architecture parameter counts (`Junkie-1B`, `Junkie-3B`, `Junkie-7B`) on the `meta` device in milliseconds rather than allocating multi-gigabyte CPU/GPU tensors.

---

## Pillar B: Strict Enforcement of All Project Rules & Conventions

### B1. 4-Layer Naming Boundary & Inaugural `v1.0.0` Framing
- **Strict 4-Layer Brand Split (`docs/RELEASE_NAME_REPORT.md`):**
  1. **Company / Python Distribution / Package / CLI:** **Lucid Machines** (`lucid-machines`, `lucid_machines`, `lucid-machines` CLI, `LUCID_MACHINES_*` env prefix).
  2. **Core Neural Architecture:** **Triune** (`TriuneTransformer`, `TriuneModel`, `TriuneConfig`, Hugging Face `"model_type": "triune"`, templates `models/configs/triune-*.json`).
  3. **Inaugural Flagship Models To Train:** **Junkie** (`models/configs/junkie.json`, `junkie-7b.json`, `junkie-3b.json`, `junkie-1b.json`, `junkie-mini.json`). Use hyphenated size suffixes (`Junkie-Mini`, `Junkie-1B`, `Junkie-3B`, `Junkie-7B`), and keep `TriuneTransformer` / `TriuneModel` / `TriuneConfig` as the sole neural architecture classes.
  4. **Desktop Workbench:** **Polysom Studio** (`polysom` CLI, `PolysomStudio.exe`, `PolysomStudioSetup.exe`, 6 themes: `Hypnagogic`, `Phosphene`, `Pareidolia`, `Afterimage`, `Mirage`, `Euphoria`).
- **Inaugural `v1.0.0` Release Framing:**
  - Branch: `release/lucid-machines-v1.0.0` | Package & Studio Version: `1.0.0`.
  - Frame **Lucid Machines `v1.0.0`** as the **first public release** of Lucid Machines. Do not use internal "upgraded from legacy v2 to v3" or "migration from previous public releases" framing in public documentation.
  - Compare **CentroidSteer** strictly against scientific baselines: **`No-Steer Low-Rank` (`nosteer`)**, **`Memory-Matched Random Subspace` (`random_same_mem`)**, and **`Full-Rank AdamW` (`adamw`)**.

### B2. 5-Tier Empirical Benchmark Taxonomy (`CONTRIBUTING.md §1`)
Never conflate isolated microkernel speedups with end-to-end model throughput. Always classify metrics into their explicit tier:
1. **Tier 1 — Isolated Kernel Microbenchmarks:** `VectorisedGLA` (`48.0 KiB` state, `10.07x` prefill / `7.31x` decode vs. SDPA at `H=6, L=16384`), `Gated DeltaNet` (`100.0%` standard & overwrite MQAR recall), sparse `MoE_FFN(use_sparse_eval=True)` (`1.648x` zero-sync at `N=1`, `0.0` max diff; `7.518x` at `d=384, E=8, k=2, N=4096`), `chunked_cross_entropy` (`98.5%` transient VRAM saved vs. PyTorch / `94.1%` vs. Unsloth at `V=32000, N=8192`), `FastSwiGLU` (`1.95x` vs. PyTorch).
2. **Tier 2 — End-to-End 90.79M Baseline Model (`64.21M` Active) & Layer Streaming:** Rescued `step_40555` checkpoint (`383,135 tok/s` peak prefill at `B=32, T=1024`; `200,463 tok/s` at `B=1, T=2048`; `157.0 tok/s` tensor decode; `40.8 tok/s` warm streaming decode / `71.3 ms` TTFT on RTX 5070; `25,973 tok/s` LoRA fine-tuning, `+6.7%` vs. Unsloth), `LayerStreamingEngine` (`1.52B + CentroidSteer` trained in `3,132.6 MB` VRAM), `NVFP4` (`3.20x` compression, `46.65 MB` across 96 blocks; Hadamard W4A16 4-bit PPL `119.106`).
3. **Tier 3 — Multi-Layer Wrapper vs. Legacy Diagnostics:** Multi-layer long-context sweep (`64.0 KiB` constant cache across `L=512..16384`, `99.8%` reduction vs. SDPA, `4.02x` prefill / `4.13x` decode), explicitly separating production `MoE_FFN` from legacy diagnostic `MoERoutingBlock` (`0.47x`).
4. **Tier 4 — Tokenizer Compression vs. Model Computation:** Held-out lossless text compression (`2.65x` aggregate, `100%` round-trip reconstruction) reported separately from compute-equivalent `compress_mode="expand"` (`0.0` max logit diff vs. base token IDs) and `order_pool` orthogonal RoPE phase rotation.
5. **Tier 5 — Speculative & Conformal CALM Early Exits:** Monotonic exit PPL (`111.80 -> 111.18 -> 110.92`), `4.31x` Reflex prefill speedup, `T=0.0` exact-match greedy self-speculative decode (`91.62 tok/s`, `51.28%` acceptance, 100% Cortex token parity), `T=0.7` Leviathan rejection sampling (`90.74 tok/s`, `63.64%` acceptance, Pearson $\chi^2$ verified), and held-out conformal CALM (`95.73%` Reflex agreement at $\tau=0.20$, `20.35%` FLOPs saved).

### B3. UI & Visual Design Compliance (`studio/src/` & `website/`)
- **Zero Emojis:** No emoji or decorative Unicode pictographs anywhere in `studio/src/` (`index.html`, `style.css`, `app.js`, SVG assets).
- **Zero CSS/SVG Gradients:** No `linear-gradient(`, `radial-gradient(`, `conic-gradient(`, `<linearGradient>`, or `<radialGradient>` in `studio/src/`. Use solid layered surfaces, borders, and crisp elevation tokens across all 6 themes (`hypnagogic`, `phosphene`, `pareidolia`, `afterimage`, `mirage`, `euphoria`).
- **Zero Unsafe DOM Sinks:** Never use `innerHTML`, `dangerouslySetInnerHTML`, or `document.write(` in `studio/src/app.js`.
- **Preserve Custom Website & Lucid Eye Brand Symbol:** Preserve the user's custom React/Vite `website/` design and Lucid Eye icon (`website/public/lucid-symbol.svg`, `studio/src/polysom-logo.svg`, `studio/windows/polysom.ico`). Ensure `website/dist/index.html` contains no `noindex` meta tag and `website/node_modules/` stays gitignored.

### B4. Defense-in-Depth Security Invariants (`R4`)
- **Safe Checkpoint Deserialization:** Every `torch.load` call across runtime, trainer, scripts, and benchmarks must pass `weights_only=True` (with CPU-first `map_location="cpu"` where appropriate).
- **Subprocess Safety:** Every `subprocess.run` / `subprocess.Popen` call in workspace, agent, and API code must use `shell=False` with an explicit argument list and command allowlist validation. Never use dynamic code evaluation or execution builtins.
- **Symlink-Resolved Path Confinement:** All file, model, checkpoint, config, dataset, and workspace endpoints must resolve paths via `Path.resolve(strict=False).is_relative_to(...)` and reject `..`, UNC shares, symlinks escaping the root, and protected directories (`checkpoints`, `data`, `exports`, `papers`, `.git`).
- **Auth, CSRF, SSRF & Plugin Trust:** Enforce proxy-aware `LUCID_MACHINES_API_TOKEN` bearer authentication (`TRIUNE_API_TOKEN` fallback), `Origin`/`Host` CSRF validation, `validate_http_url` SSRF checks, header-only `x-goog-api-key` transmission, `trust_remote_code=False` by default, and `trusted=False` default for external Ed25519-verified plugins.

### B5. Strict Asset Preservation & Untracked `WORKSPACE_RULES.md`
- **Immutable Research & Training Assets:** Never delete, overwrite, or corrupt:
  - `checkpoints/*.pt` (including `triune_studio_step_36500_rescued.pt` and `triune_step_0.pt`)
  - `checkpoints/session_state.json` (must remain at `"step": 40555`; always restore if any test mutates it)
  - `data/*.jsonl` (`fineweb_edu_real.jsonl`, `fineweb_edu_large.jsonl`, `finetune.jsonl`)
  - `exports/*`
  - `papers/paper1_centroidsteer/*` and `papers/paper2_triune_systems/*` (0 git modifications permitted in `papers/paper1_centroidsteer/`)
  - `triune_tokenizer.json`
- **Untracked Private Rules File:** `WORKSPACE_RULES.md` is strictly read-only and must never be staged or tracked in git (`git ls-files WORKSPACE_RULES.md` must always return empty output).

---

## Pre-Completion Verification Checklist

Run these commands with `studio\studio_env\Scripts\python.exe` before declaring any milestone complete:

```powershell
# 1. Format & lint check
.\studio\studio_env\Scripts\ruff.exe format --check .
.\studio\studio_env\Scripts\ruff.exe check .

# 2. Bytecode compilation check
.\studio\studio_env\Scripts\python.exe -m compileall -q lucid_machines triune scripts tests benchmarks

# 3. Core & release regression suites
.\studio\studio_env\Scripts\python.exe -m pytest tests/test_cloud_junkie_kit.py tests/test_workspace_and_agent_loops.py tests/test_studio_features.py tests/test_backward_compat.py tests/test_security_regression.py tests/e2e/test_release_pipeline_e2e.py -q

# 4. Verify preservation invariants
.\studio\studio_env\Scripts\python.exe -c "import json; assert json.loads(open('checkpoints/session_state.json', encoding='utf-8').read())['step'] == 40555"
git ls-files WORKSPACE_RULES.md
```
