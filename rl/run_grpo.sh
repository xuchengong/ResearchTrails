#!/usr/bin/env bash
# Dr. GRPO from the SFT epoch-3 checkpoint with the paper's settings.
#   bash rl/run_grpo.sh             # train on one GPU
#   GPUS=4 bash rl/run_grpo.sh      # train on four GPUs
#   bash rl/run_grpo.sh --dry-run   # validate and save the configuration only; no GPU or paid calls
# SFT_CHECKPOINT=sft/runs/qwen3-8b/seed-42/checkpoint-N, with a new RUN_DIR, starts from
# another SFT checkpoint (default: checkpoint-72, SFT epoch 3).
# CONTINUE_FROM=<run>/checkpoint-epoch-N, with a new RUN_DIR, keeps training that
# adapter and its optimizer state instead of starting a fresh one. Rerun the same
# command to resume. Other arguments go to rl.train_grpo, e.g. --epochs 5.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE"
GPUS="${GPUS:-1}"
[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "GPUS must be a positive integer" >&2; exit 2; }
PYTHON="${PYTHON:-sft/.venv/bin/python}"
# Paid calls are confirmed automatically; --dry-run prepares them, reports how many remain,
# and stops without submitting.
confirm=(--confirm-submit)
args=()
for arg in "$@"; do
  if [[ $arg == --dry-run ]]; then confirm=(); else args+=("$arg"); fi
done
options=(
  --sft-checkpoint "${SFT_CHECKPOINT:-sft/runs/qwen3-8b/seed-42/checkpoint-72}"
  --output-dir "${RUN_DIR:-rl/runs/grpo-lr1e-4}"
  --epochs 10 --learning-rate "${LEARNING_RATE:-1e-4}" --seed 42 --resume
)
if [[ -n "${CONTINUE_FROM:-}" ]]; then
  options+=(--continue-from "$CONTINUE_FROM")
fi

if (( GPUS > 1 )); then
  export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2048}"
  "$PYTHON" -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node "$GPUS" \
    -m rl.train_grpo "${options[@]}" "${args[@]}" "${confirm[@]}"
else
  # One process uses the first visible GPU.
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
  "$PYTHON" -m rl.train_grpo "${options[@]}" "${args[@]}" "${confirm[@]}"
fi
