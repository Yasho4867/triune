# Triune AI — Agentic Workspace Rules & Development Directives

> [!IMPORTANT]
> **CORE DIRECTIVE**:  
> **Review your work. Make sure everything is concrete before presenting. Check for redundancies and garbage, enforce maximum efficiency coding, no useless comments, and ALWAYS push changes to GitHub upon completion.**

---

## 1. Concrete Verification & Empirical Rigor Before Presenting
- **Evidence Before Claims**: Never assert that an optimization, kernel, or architectural change is complete or faster without running it on actual physical hardware first and confirming output.
- **Zero Hallucinated Numbers**: Record exact latencies (ms), throughputs (tok/s), memory allocations (MB), and numerical deviations ($\Delta_{\max}$).
- **Deterministic Parity**: For any optimized kernel or fast path, assert numerical parity against canonical references ($\Delta_{\max} < 2 \times 10^{-4}$ in BF16/FP32).

## 1.1 Agent Continuity & Handoff
- All AI agents are interchangeable workers, not sole owners of task context.
- Inspect the worktree and existing notes before starting. Continue and resume unfinished work from its current state; never discard or restart it merely because the agent/session changed.
- If work remains, leave a concise handoff with changes, relevant files, commands/tests and outcomes, and blockers.
- Treat existing dirty and untracked files as user/prior-worker work. Preserve them; ask before destructive cleanup or replacement.

---

## 2. Maximum Efficiency Coding & Clean Code Standards
- **No Useless Comments**: Write self-documenting code. Do not add trivial or performative comments (e.g. `# import torch`, `# initialize model`, `# return output`). Only explain non-obvious mathematical invariants, hardware memory bounds, or numerical stability guards.
- **Zero Dead Boilerplate**: Do not leave unused imports, uncalled functions, duplicate tensor allocations, or redundant tensor copies.
- **Direct Memory Management**: Always avoid redundant `.contiguous()` copies or full sequence tensor expansions when only slice views or inplace operations are required.

---

## 3. Benchmark Taxonomy & Defensible Claims
Always report performance within the explicit 4-tier hierarchy to maintain absolute scientific integrity:
1. **Tier 1 (Kernel Saturation)**: Isolated batched GLA CUDA stream execution (e.g. 5.21M positions/s at $B=32, T=4096, D=256$).
2. **Tier 2 (Full-Model Physical Forward)**: Complete 76M parameter model execution with active weights, MoE, and vocabulary head (e.g. 226k Cortex, 1.61M Reflex at $B=32, T=2048$).
3. **Tier 3 (Effective User-Space Throughput)**: Ingested user tokens accounting for Dynamic Tokenizer lossless sequence compression (e.g. 2.62x multiplier: 592k Cortex, 4.22M Reflex).
4. **Tier 4 (Autoregressive Decode)**: Sequential token-by-token generation rate (e.g. 48 tok/s single-stream) with strictly constant $O(1)$ recurrent memory ($48\text{ KiB}$/layer).

---

## 4. Test Invariants & Continuous Health
Before committing or presenting any task:
```bash
pytest tests/test_audit_validation.py tests/test_million_toks_throughput.py tests/test_studio_usability_and_sync.py
```
All tests must be 100% green. Never bypass or disable failing tests.

---

## 5. UI & Studio Conventions
- **Obsidian 3D Brain**: Dual-hemisphere biological topology with deep limbic core, brainstem base, corpus callosum axonal tracts, and obsidian neon node styling.
- **Ergonomics**: Vertical resize handle (`.canvas-resize-handle`) persisted via `localStorage` with bounded constraints ($160\text{px} \le h \le 560\text{px}$).
- **Realtime Sync**: The Studio UI must always reflect the active model's real weights and architecture without hardcoded simulations.

---

## 6. Always Push to GitHub
- Make pushing the normal habit after the requested changes are complete and verified: inspect branch/status/upstream, stage only intended files, commit conventionally, and push to the correct upstream.
- Never force-push or silently change branches. If there is no upstream, divergence, or a risk of overwriting remote history, stop and ask.
- Report commit, destination branch, and verification. If a check or push is blocked, preserve changes and clearly hand off what remains.

---

## 7. Elimination of Redundancies & Zero Garbage Policy
- **No Leftover Scratch / Garbage Files**: Never commit temporary test dumps, benchmark JSON scraps, `.tmp`, `.bak`, `.swp`, or unmaintained exploratory scripts into the repository root. Clean workspace before presentation.
- **Single Canonical Implementation**: Eliminate duplicate algorithmic logic across the codebase. Routing, GLA recurrence, RMSNorm, and attention must dispatch to a single canonical high-performance kernel rather than diverging implementations.
- **Zero Redundant Memory Materialization**: Audit every forward pass and kernel for unnecessary memory allocations (e.g. projecting full vocabularies during inference, expanding contiguous tensors when broadcast views suffice, or retaining dead intermediate activation graphs).
- **Dead Code Pruning**: Continuously audit imports, remove unused variables, and prune legacy fallback branches that degrade performance or clutter architecture flow.
