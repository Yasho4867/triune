# Hero Messaging & Brand Positioning Guide — Lucid Machines `v1.0.0`

## 1. Brand Identity & Taglines

- **Company / Lab & Package**: **Lucid Machines** (`lucid-machines` distribution / `lucid_machines` Python SDK & CLI, inaugural release `v1.0.0`)
- **Core Neural Architecture**: **Triune** (`TriuneTransformer` / `TriuneModel` / `TriuneConfig` — *the Triune architecture by Lucid Machines*, templates `models/configs/triune-*.json`)
- **Inaugural Flagship Models To Train**: **Junkie** (`models/configs/junkie-mini.json` `535.10M`, `junkie-1b.json` `1.16B`, `junkie-3b.json` `2.99B` 16-expert Colab L4 sweet spot, and `junkie-7b.json` / `junkie.json` `7.40B` flagship)
- **Desktop Workbench**: **Polysom Studio v1.0.0** (by Lucid Machines — `polysom` CLI, `PolysomStudio.exe`, `PolysomStudioSetup.exe`, with 6 themes: `Hypnagogic`, `Phosphene`, `Pareidolia`, `Afterimage`, `Mirage`, `Euphoria`)
- **Primary Tagline**: *3-Tier Multi-Exit MoE Language Models & Riemannian Centroid Steering on Consumer & Cloud GPUs.*
- **Developer Tagline**: *Train 1.52B models in 3.13 GB VRAM or 2.99B 16-expert Junkie-3B in 10.10 GiB FP8 on Colab L4. Compress MoE optimizer state by 98.4%. Serve O(1)-memory GLA & Conformal CALM models natively.*
- **Studio Tagline**: *Polysom Studio v1.0.0: The self-contained native Windows workbench, Project Workspace IDE, multi-mode Agent Loop harness, and 40-node visual DAG IDE for local model surgery and live Neuron Scope telemetry.*

---

## 2. Hero Copy Variants (Website & GitHub README)

### Variant A — Systems & Research Unified Hero (Recommended Default)
> **Headline:** 3-Tier Multi-Exit MoE Language Models & Riemannian Centroid Steering on Consumer GPUs  
> **Subheadline:** **Lucid Machines v1.0.0** (`lucid-machines`) unifies the open-source **Triune** architecture (`TriuneTransformer`) C/AVX2/CUDA/Triton engine with **Polysom Studio v1.0.0**, a self-contained `.NET 10` + `WebView2` desktop workbench, and ships the validated training substrate for **Junkie**—our inaugural flagship model (`Junkie-Mini` `535.1M`, `Junkie-1B` `1.16B`, `Junkie-3B` `2.99B` 16-expert Colab L4 FP8 sweet spot, and `Junkie-7B` `7.40B`). Train `1.52B` models inside `3,132.6 MB` VRAM with depth-invariant NVFP4 Layer Streaming, compress MoE expert optimizer states to `< 0.30 B/p` (`||Q^T Q - I||_F < 1e-4`, validated across 5 seeds against `No-Steer Low-Rank`, `Memory-Matched Random Subspace`, and `Full-Rank AdamW`) via multi-directional **CentroidSteer**, and run constant-$O(1)$ cache **VectorisedGLA** + **Reflex/Limbic/Cortex** conformal CALM inference (`40.8 tok/s` warm decode / `71.3 ms` TTFT on RTX 5070).  
> **Primary CTA:** `pip install lucid-machines` | **Secondary CTA:** Download `PolysomStudio-1.0.0-win-x64.zip`

### Variant B — 1-Line PyTorch / Hugging Face Booster Hero
> **Headline:** Upgrade Any PyTorch or Hugging Face Model with 1 Line of Code  
> **Subheadline:** Call `engine = lucid_machines.upgrade(model, lora_rank=16)` to inject fused `FastRMSNorm`, zero-allocation `addmm_` LoRA (`+6.7%` faster than Unsloth at `25,973 tok/s`), chunked Cross-Entropy (`98.5%` transient VRAM saved vs. PyTorch at `V=32,000`), and `CentroidSteerOptimizer`.

### Variant C — Desktop Workbench Hero (`Polysom Studio v1.0.0`)
> **Headline:** Polysom Studio v1.0.0: Visual Model Surgery, Live Stiefel Telemetry, Workspace IDE & Agentic Loops  
> **Subheadline:** Built on `.NET 10`, `WebView2`, and `lucid_machines_core.dll`, Polysom Studio v1.0.0 brings an Obsidian 3D brain topology inspector, 6 zero-gradient editorial themes (`Hypnagogic`, `Phosphene`, `Pareidolia`, `Afterimage`, `Mirage`, `Euphoria`), a confined Project Workspace IDE, multi-mode Agent Loops (`ReAct`, `PEV`, `Multi-Agent`, `Training Supervisor`), a 40-node typed DAG execution engine, and OpenAI-compatible local serving to Windows workstations.

---

## 3. Core Value Pillars (OSS Core vs. Polysom Studio v1.0.0)

| Pillar | Open-Source Core (`lucid-machines` SDK & CLI) | Desktop Workbench (`Polysom Studio v1.0.0`) |
| :--- | :--- | :--- |
| **Optimizer & VRAM** | 3-Tier `CentroidSteerOptimizer` (`Muon` 2.0 B/p + `CentroidSteer` `0.1118–0.3678 B/p` + `8-bit AdamW`), `LayerStreamingEngine` (`1.52B` in `3.13 GB` VRAM). | Live `sCholQR2` orthogonality monitor ($\|Q^T Q - I\|_F$), interactive VRAM planner, and real-time expert utilization heatmaps. |
| **Architecture & Kernels** | `TriuneTransformer`, `VectorisedGLA` ($O(1)$ `48.0 KiB` isolated kernel), `Gated DeltaNet` (100% MQAR recall), `MoE_FFN` (`7.518x` sparse eval), NVFP4 (`3.20x`), plus `junkie-*.json` configurations. | Obsidian 3D brain stratum visualizer (Reflex L2 / Limbic L5 / Cortex L8), 6 themes (`Hypnagogic`, `Phosphene`, `Pareidolia`, `Afterimage`, `Mirage`, `Euphoria`), and interactive CALM threshold ($\tau$) slider. |
| **Workflows & Tooling** | Unified `lucid-machines` & `polysom` CLIs (`train`, `finetune`, `serve`, `chat`, `doctor`, `benchmark`) + Colab Pro runner (`cloud/`) + `import triune` shim. | Project Workspace IDE, SSE Agent Loops with HITL diff approval, 40-node Visual DAG IDE, PC model scanner, and 1-click `PolysomStudioSetup.exe` wizard. |
| **Security & Governance** | `weights_only=True` checkpoint hydration, symlink-resolved path confinement, `shell=False`, SSRF validation, `PythonSandbox` resource caps. | Windows Credential Manager BYOK vault (`Polysom Studio`), Ed25519 plugin signature verification (`trusted=False` default), proxy-aware bearer auth. |
