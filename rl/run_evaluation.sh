#!/usr/bin/env bash
# Per-epoch held-out evaluation of GRPO (the right panel of the training figure and the
# Sol column of the cross-judge table): the SFT initialization and each epoch's GRPO
# adapter predict the 402 first-1/2/3 cases greedily, and Sol medium judges both methods.
#   bash rl/run_evaluation.sh                  # predict and judge all ten epochs
#   EPOCHS="7 10" bash rl/run_evaluation.sh    # only epochs 7 and 10
#   bash rl/run_evaluation.sh all --dry-run    # predict, then prepare the judgments without submitting
# The first argument is the stage (prepare | predict | judge | all); the rest go to rl.evaluate.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a

export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
# One process uses the first visible GPU (default 0).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"

unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE"
# The optional first argument is the stage (default all), so options such as --dry-run may come first.
STAGE=all
if [[ $# -gt 0 && $1 != -* ]]; then STAGE=$1; shift; fi
# Paid calls are confirmed automatically; --dry-run prepares them, reports how many remain,
# and stops without submitting.
confirm=(--confirm-submit)
args=()
for arg in "$@"; do
  if [[ $arg == --dry-run ]]; then confirm=(); else args+=("$arg"); fi
done
RUN_DIR="${RUN_DIR:-rl/runs/grpo-lr1e-4}"
read -r -a epochs <<< "${EPOCHS:-1 2 3 4 5 6 7 8 9 10}"
for epoch in "${epochs[@]}"; do
  "${PYTHON:-sft/.venv/bin/python}" -m rl.evaluate "$STAGE" \
    --run-dir "$RUN_DIR" --adapter "$RUN_DIR/checkpoint-epoch-$epoch" \
    --output-dir "$RUN_DIR-eval-epoch$epoch" "${args[@]}" "${confirm[@]}"
done
