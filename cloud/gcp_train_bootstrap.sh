#!/usr/bin/env bash
# Lucid Machines — Optional Linux CUDA VM Bootstrap (GCE / Vertex AI / Bare-Metal Linux)
# Supports: --model junkie-mini|junkie-1b|junkie-3b|junkie-7b, --mode ablation|train|studio,
#           single-GPU or multi-GPU torchrun, and optional GCS checkpoint auto-resume.
set -euo pipefail

MODEL="junkie-3b"
MODE="train"
GPUS=1
STEPS=5000
BATCH_SIZE=""
SEQ_LEN=1024
RANK=""
STEER_DIRS=4
STEER_MODE="centroid"
CHECKPOINT_DIR="${HOME}/lucid_checkpoints"
GCS_BUCKET=""
PORT=8000
REPO_DIR="${HOME}/TriuneTransformer"
REPO_URL="${LUCID_MACHINES_REPO_URL:-https://github.com/lucid-machines/triune.git}"
BRANCH="release/lucid-machines-v1.0.0"

usage() {
  cat <<'EOF'
Usage: bash cloud/gcp_train_bootstrap.sh [OPTIONS]

Options:
  --model <junkie-mini|junkie-1b|junkie-3b|junkie-7b>  Target Junkie configuration (default: junkie-3b)
  --mode <ablation|train|studio>    Execution mode (default: train)
  --gpus <N>                        Number of CUDA GPUs (1=single process, >1=torchrun DDP)
  --steps <N>                       Training steps (default: 5000)
  --batch-size <N>                  Micro-batch size per GPU (default: 4 for mini/1b/3b, 1 for 7b)
  --seq-len <N>                     Sequence length (default: 1024)
  --rank <N>                        CentroidSteer projection rank (default: 64 mini/1b/3b, 256 7b)
  --steer-dirs <N>                  CentroidSteer orthogonal subspace dirs (default: 4)
  --checkpoint-dir <PATH>           Local checkpoint directory
  --gcs-bucket <gs://...>           Optional GCS URI for auto-resume & periodic rsync
  --port <PORT>                     Polysom Studio port in studio mode (default: 8000)
  --repo-dir <PATH>                 Repository checkout path (default: ~/TriuneTransformer)
  -h, --help                        Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --rank) RANK="$2"; shift 2 ;;
    --steer-dirs) STEER_DIRS="$2"; shift 2 ;;
    --steer-mode) STEER_MODE="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --gcs-bucket) GCS_BUCKET="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

case "${MODEL}" in
  junkie-mini)
    MODEL_CFG="models/configs/junkie-mini.json"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    RANK="${RANK:-64}"
    GRAD_ACCUM=8
    ;;
  junkie-1b)
    MODEL_CFG="models/configs/junkie-1b.json"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    RANK="${RANK:-64}"
    GRAD_ACCUM=8
    ;;
  junkie-3b)
    MODEL_CFG="models/configs/junkie-3b.json"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    RANK="${RANK:-64}"
    GRAD_ACCUM=8
    ;;
  junkie-7b|junkie)
    MODEL_CFG="models/configs/junkie-7b.json"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    RANK="${RANK:-256}"
    GRAD_ACCUM=16
    ;;
  *)
    echo "Unsupported --model '${MODEL}'. Expected junkie-mini, junkie-1b, junkie-3b, or junkie-7b." >&2
    exit 1
    ;;
esac

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --branch "${BRANCH}" --depth 1 "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -e ".[dev]" || pip install -e .

RUN_CKPT_DIR="${CHECKPOINT_DIR}/${MODEL}"
mkdir -p "${RUN_CKPT_DIR}"

if [[ -n "${GCS_BUCKET}" ]] && command -v gcloud >/dev/null 2>&1; then
  echo "[GCS] Syncing existing checkpoints from ${GCS_BUCKET}/${MODEL} -> ${RUN_CKPT_DIR}..."
  gcloud storage rsync "${GCS_BUCKET%/}/${MODEL}" "${RUN_CKPT_DIR}" || true
fi

sync_checkpoints_gcs() {
  if [[ -n "${GCS_BUCKET}" ]] && command -v gcloud >/dev/null 2>&1; then
    echo "[GCS] Uploading checkpoints ${RUN_CKPT_DIR} -> ${GCS_BUCKET%/}/${MODEL}..."
    gcloud storage rsync "${RUN_CKPT_DIR}" "${GCS_BUCKET%/}/${MODEL}" || true
  fi
}
trap sync_checkpoints_gcs EXIT

case "${MODE}" in
  ablation)
    python3 benchmarks/validate_centroid_steer_upgrade.py \
      --seeds 42 1337 2026 777 1024 \
      --md-steps 120 \
      --fw-steps 250
    ;;
  train)
    TRAIN_CMD=(
      scripts/train.py
      --model_config "${MODEL_CFG}"
      --checkpoint_dir "${RUN_CKPT_DIR}"
      --auto_resume
      --steps "${STEPS}"
      --batch_size "${BATCH_SIZE}"
      --grad_accum_steps "${GRAD_ACCUM}"
      --seq_len "${SEQ_LEN}"
      --rank "${RANK}"
      --steer_dirs "${STEER_DIRS}"
      --steer_mode "${STEER_MODE}"
      --save_every 200
      --eval_every 200
      --log_every 10
      --no_wandb
      --no_hf_login
    )
    if [[ "${GPUS}" -gt 1 ]]; then
      torchrun --standalone --nproc_per_node="${GPUS}" "${TRAIN_CMD[@]}"
    else
      python3 "${TRAIN_CMD[@]}"
    fi
    ;;
  studio)
    echo "[Polysom Studio] Serving on 127.0.0.1:${PORT} (tunnel via: gcloud compute ssh <VM> -- -L ${PORT}:127.0.0.1:${PORT})"
    kcx serve --host 127.0.0.1 --port "${PORT}"
    ;;
  *)
    echo "Unsupported --mode '${MODE}'. Expected ablation, train, or studio." >&2
    exit 1
    ;;
esac
