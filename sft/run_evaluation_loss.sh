#!/usr/bin/env bash
# No prediction generation or judge calls. Run on a single GPU.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Set these before Python imports Hugging Face so downloads and Xet chunks use
# the project filesystem rather than the home-directory quota.
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE"

PYTHON="${PYTHON:-sft/.venv/bin/python}"
RUN_DIR="${RUN_DIR:-sft/runs/qwen3-8b/seed-42}"
# --dry-run checks the inputs without loading a model, like the other launchers' dry runs.
args=()
for arg in "$@"; do
  if [[ $arg == --dry-run ]]; then args+=(--prepare-only); else args+=("$arg"); fi
done
"$PYTHON" -m sft.evaluate_checkpoints \
  --run-dir "$RUN_DIR" \
  --batch-size "${EVAL_BATCH_SIZE:-1}" "${args[@]}"
