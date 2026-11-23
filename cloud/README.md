# Lucid Machines `v1.0.0` — Cloud GPU Runner Guide (`Junkie-3B` Colab L4 Sweet Spot, `Junkie-1B`, `Junkie-Mini` & Flagship `Junkie-7B`)

This directory provides turnkey cloud-GPU training, curriculum orchestration, and **Polysom Studio v1.0.0** remote inspection workflows for **Lucid Machines** (`TriuneTransformer` architecture, `CentroidSteer` optimizer, and inaugural `Junkie` models; [https://lucidmachines.co](https://lucidmachines.co) · [https://github.com/lucid-machines/triune](https://github.com/lucid-machines/triune)).

---

## 1. Exact Measured Model & Optimizer Footprint (CPU `meta` + Live State)

All parameter counts below are measured directly from `TriuneTransformer` on `torch.device("meta")` (`vocab_size=32000`, `EXPERT_HIDDEN_MULTIPLIER=6`, `shared_expert=true`), and optimizer state bytes are measured from `CentroidSteerOptimizer` (`num_steer_dirs=4`):

| Model Config | Total Params | Active Params (`top_k=1`: Reflex / Limbic / Cortex / Expected 34-33-33 / Dynamic 50-30-20) | Expert Tier (`%`) | CentroidSteer Expert State (`B/p`) | Static VRAM: CentroidSteer + Muon + Base (FP8 / BF16) | Static VRAM: FP32 AdamW (`8 B/p` states) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| `models/configs/junkie-mini.json` (`12L / 1024D / 4E+1S`) | **535,101,975** (`535.10M`) | `161.30M` / `252.59M` / `280.65M` (**230.81M** expected / **212.56M** dynamic) | `251.66M` (`47.0%`) | **`0.1992 B/p`** (`rank=64`, `40.2x` vs AdamW) | **`2.62 GiB` FP8** / **`3.12 GiB` BF16** (`2.57 GiB` w/ 8-bit base) | `5.98 GiB` (`7.97 GiB` w/ FP32 master) |
| `models/configs/junkie-1b.json` (`22L / 1024D / 4E+1S`) | **1,162,535,483** (`1.16B`) | `174.94M` / `387.95M` / `568.15M` (**374.99M** expected / **317.48M** dynamic) | `704.64M` (`60.6%`) | **`0.1992 B/p`** (`rank=64`, `40.2x` vs AdamW) | **`4.79 GiB` FP8** / **`5.87 GiB` BF16** (`5.31 GiB` w/ 8-bit base) | `12.99 GiB` (`17.32 GiB` w/ FP32 master) |
| **`models/configs/junkie-3b.json` (`16L / 1024D / 16E+1S`)** | **2,990,355,155** (`2.99B`) | `106.75M` / `198.07M` / `469.69M` (**395.74M** MoE-trunk Cortex / **206.73M** dynamic / **256.66M** uniform) | `2,617.25M` (`87.5%`) | **`0.1992 B/p`** (`rank=64`, `40.2x` vs AdamW) | **`10.10 GiB` FP8** (`9.55 GiB` w/ 8-bit base) / **`12.89 GiB` BF16** (`< 20 GiB` resident on L4 24GB!) | **`33.42 GiB` (OOM on L4 24GB)** (`44.56 GiB` w/ FP32 master) |
| `models/configs/junkie-7b.json` (and `junkie.json`, `32L / 2048D / 8E+1S`) | **7,396,845,163** (`7.40B`) | `1.17B` / `2.02B` / `2.68B` (**1.95B** expected / **1.73B** dynamic) | `5,234.49M` (`70.8%`) | **`0.3678 B/p`** (`rank=256`, `21.7x` vs AdamW) | **`27.60 GiB` FP8** / **`34.49 GiB` BF16** (`33.38 GiB` w/ 8-bit base) | **`82.67 GiB` (OOM on 80GB A100)** (`110.22 GiB` w/ FP32 master) |

---

## 2. The #1 Colab Pro 200-Compute-Unit Sweet Spot: `Junkie-3B` on L4 (`24 GB`, Native FP8)

When training on a **200-compute-unit Google Colab Pro** budget, **`Junkie-3B` (`models/configs/junkie-3b.json`) on an NVIDIA L4 (`24 GB VRAM`, Ada Lovelace CC 8.9)** is the optimal Pareto point:

1. **~3x More Tokens per Compute Unit than A100**:
   - Colab Pro burns compute units roughly **4.5x–6x slower on L4** than on A100 while L4's native Ada FP8 Tensor Cores (`--use_fp8`) and `Junkie-3B`'s `206.73M` average dynamic-exit active parameters (`depth_mode="dynamic"`, `target_depth_dist=[0.50, 0.30, 0.20]`) sustain ~55%–65% of A100 per-second token throughput.
   - Across 200 compute units, L4 trains **`~21B–23B` real tokens** (equivalent to **`~35B+` AdamW-equivalent tokens** when combining `CentroidSteer` + `Muon` sample efficiency and `DynamicTokenizer` `2.65x` lossless span density).
2. **`< 20 GiB` Resident on L4 24 GB (Zero PCIe Offload Needed)**:
   - Because `87.5%` (`2.62B`) of `Junkie-3B`'s `2.99B` parameters reside in the 16-expert MoE tier compressed to **`0.1992 B/p`** by `CentroidSteer`, static memory in FP8 is only **`10.10 GiB`** (`9.55 GiB` with `--use_8bit_base`), leaving **~13.9 GiB** for activations and batch sequences (`< 20 GiB` total resident VRAM).
   - Standard FP32 AdamW requires **`33.42 GiB`** static (`44.56 GiB` with FP32 master weights) and immediately OOMs on L4 24 GB.

---

## 3. 3-Stage Decontaminated Training Curriculum (`Junkie-3B` & `Junkie-1B`)

To maximize reasoning, coding, and mathematical density per compute unit without benchmark contamination:

| Stage | Compute Budget | Dataset Mixture & Schedule | CLI Flags / Execution Mode |
| :--- | :---: | :--- | :--- |
| **Stage 1: Foundation Pretraining** | **80%** (`~17B–18.5B` tokens on L4) | **`50%` FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`) + **`30%` Permissive Code** (The Stack v2 / StarCoderData decontaminated) + **`20%` FineMath** (`HuggingFaceTB/finemath`). Warmup + stable peak LR (`3e-4` base / `0.02` Muon) with `CentroidSteer` (`--rank 64 --steer_dirs 4`). | `--model_config models/configs/junkie-3b.json --use_fp8 --depth_mode dynamic --target_depth_dist 0.50,0.30,0.20` |
| **Stage 2: High-Density WSD Annealing** | **15%** (`~3B–3.5B` tokens on L4) | **Warmup-Stable-Decay (WSD) Cooldown** on high-density synthetic textbooks (` Cosmopedia-v2` / `FineWeb-Edu` score $\ge 4$), verified algorithmic code, and formal math proofs, paired with joint multi-exit KL self-distillation (`depth_mode="joint"`) to calibrate Reflex (`L2`) and Limbic (`L5`) exits toward Cortex (`L16`). | `--model_config models/configs/junkie-3b.json --use_fp8 --depth_mode joint --auto_resume` |
| **Stage 3: Clean SFT + Verifiable GRPO RL** | **5%** (`~1B` tokens + RL rollouts) | **1.** Decontaminated multi-turn instruction SFT (n-gram filtered against GSM8K, MATH, HumanEval, MBPP, MMLU).<br>**2.** Verifiable **Group Relative Policy Optimization (GRPO)** RL using deterministic Python/unit-test execution and symbolic math verifiers inside `PythonSandbox`. | `kcx finetune` + `PythonSandbox` verifiable reward loop |

---

## 4. Primary Cloud Runner: Google Colab Pro (`lucid_machines_colab_runner.ipynb`)

Open **`cloud/lucid_machines_colab_runner.ipynb`** in Google Colab Pro (`L4 24GB` or `A100 80GB`):

1. **Cell 1 — Hardware & Google Drive Mount:** Verifies CUDA/BF16/FP8 capability and mounts Google Drive at `/content/drive/MyDrive/LucidMachines/checkpoints` for persistent checkpoints across sessions.
2. **Cell 2 — Install `lucid-machines` (`v1.0.0`):** Clones `release/lucid-machines-v1.0.0`, installs dependencies, and runs dry-run memory verification on `junkie-3b` and `junkie-1b`.
3. **Cell 3 — #1 Sweet Spot: Train `Junkie-3B` on Colab L4 (`--use_fp8 --depth_mode dynamic`):** Runs the 16-expert `2.99B` sparse MoE (`206.73M` dynamic active, `10.10 GiB` FP8 static) with automatic Drive checkpoint resume.
4. **Cell 4 — Train `Junkie-1B` (`1.16B` Total / `374.99M` Expected Active):** Runs `models/configs/junkie-1b.json` (`4.79 GiB` FP8 / `5.87 GiB` BF16 static).
5. **Cell 5 — Train `Junkie-Mini` (`535.10M` Total / `230.81M` Expected Active):** Fast single-GPU training on L4 / T4 / A100 (`3.12 GiB` BF16 static).
6. **Cell 6 — Train Flagship `Junkie-7B` (`7.40B` Total / `1.95B` Expected Active):** Runs `models/configs/junkie-7b.json` on A100-80GB (`34.49 GiB` BF16 static; automatically enables `--streaming` on `< 45 GiB` GPUs).
7. **Cell 7 — CentroidSteer 5-Seed Scaling Validation:** Runs `benchmarks/validate_centroid_steer_upgrade.py` across 5 seeds (`42 1337 2026 777 1024`) against `No-Steer Low-Rank`, `Memory-Matched Random Subspace`, and `Full-Rank AdamW`.
8. **Cell 8 — Live Polysom Studio v1.0.0 Inside Colab:** Starts `kcx serve --host 127.0.0.1 --port 8000` (or `lucid-machines serve`) and exposes Polysom Studio via `google.colab.output.serve_kernel_port_as_window(8000)`, `serve_kernel_port_as_iframe(8000)`, and `google.colab.kernel.proxyPort(8000)`.

---

## 5. Optional Reference: Google Cloud Compute Engine (GCE Linux CUDA)

If running on dedicated GCE GPU VMs with Spot pricing and GCS checkpoint sync (`gcloud storage rsync`):

### Provision a Spot GPU VM

- **L4 (`g2-standard-8`, 24GB VRAM — `Junkie-3B` FP8, `Junkie-1B`, `Junkie-Mini` & ablations):**
  ```bash
  gcloud compute instances create lucid-l4-3b \
    --zone=us-central1-a \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \
    --boot-disk-size=150GB --maintenance-policy=TERMINATE
  ```
- **A100 80GB (`a2-ultragpu-1g` — single-GPU `Junkie-7B` with CentroidSteer):**
  ```bash
  gcloud compute instances create lucid-a100-7b \
    --zone=us-central1-a \
    --machine-type=a2-ultragpu-1g \
    --accelerator=type=nvidia-a100-80gb,count=1 \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \
    --boot-disk-size=250GB --maintenance-policy=TERMINATE
  ```
- **H100 80GB (`a3-highgpu-1g` — FP8 / multi-GPU scaling):**
  ```bash
  gcloud compute instances create lucid-h100-7b \
    --zone=us-central1-a \
    --machine-type=a3-highgpu-1g \
    --accelerator=type=nvidia-h100-80gb,count=1 \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \
    --boot-disk-size=300GB --maintenance-policy=TERMINATE
  ```

### Bootstrap Training & GCS Auto-Resume

```bash
bash cloud/gcp_train_bootstrap.sh \
  --model junkie-3b \
  --mode train \
  --gpus 1 \
  --steps 10000 \
  --gcs-bucket gs://YOUR_BUCKET/lucid-machines-ckpts
```

### Connect Local Polysom Studio Over SSH Tunnel

```bash
gcloud compute ssh lucid-l4-3b --zone=us-central1-a -- -L 8000:127.0.0.1:8000
```
Then open `http://127.0.0.1:8000` in your local browser (or start `bash cloud/gcp_train_bootstrap.sh --mode studio` on the VM).
