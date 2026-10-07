#!/usr/bin/env bash
# Held-out generation quality of every SFT checkpoint, under one decoding scheme.
#
# The loss curve from run_evaluation_loss.sh says what SFT optimizes; this says
# what it delivers. Both stages use the same cases, rubric and judge as
# the GRPO comparisons, and decode greedily now that sft/predict.py pins the
# sampling parameters, so the resulting curve sits on the same axis as
# rl/run_evaluation.sh's.
#
#   bash sft/run_checkpoint_generation.sh predict            # GPU, no paid calls
#   bash sft/run_checkpoint_generation.sh judge              # no GPU, ~2412 judgments
#   bash sft/run_checkpoint_generation.sh all
#   bash sft/run_checkpoint_generation.sh judge --dry-run    # prepare the judgments without submitting
#
# Trailing arguments go to the judge, which is what the routing flags belong to;
# sft.predict rejects them. Pass prediction-side options such as --max-new-tokens
# through PREDICT_ARGS instead.
#
# Rerunning resumes: saved predictions and judge responses are reused.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a

# Set these before Python imports Hugging Face so downloads and Xet chunks use
# the project filesystem rather than the home-directory quota.
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
case "$STAGE" in
  predict|judge|all) ;;
  *) echo "usage: $0 {predict|judge|all} [extra args]" >&2; exit 2 ;;
esac
# Paid calls are confirmed automatically; --dry-run prepares them, reports how many remain,
# and stops without submitting.
confirm=(--confirm-submit)
args=()
for arg in "$@"; do
  if [[ $arg == --dry-run ]]; then confirm=(); else args+=("$arg"); fi
done

PYTHON="${PYTHON:-sft/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-sft/data}"
RUN_DIR="${RUN_DIR:-sft/runs/qwen3-8b}"
SEED_DIR="${SEED_DIR:-$RUN_DIR/seed-42}"
PRECISION="${PRECISION:-bf16}"
JUDGE_DIR_DEFAULT="$RUN_DIR/judging"
read -r -a checkpoints <<< "${CHECKPOINTS:-24 48 72 96 120}"

if [[ "$STAGE" != judge ]]; then
  # sft.predict requires exactly one visible GPU; CUDA_VISIBLE_DEVICES is pinned
  # above so a multi-GPU allocation does not fail inside model_dtype.
  read -r -a predict_args <<< "${PREDICT_ARGS:-}"
  info="$SEED_DIR/model_info.json"
  model="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_model"])' "$info")"
  revision="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["resolved_revision"])' "$info")"

  # Epoch 0: the untrained base, the floor every checkpoint is measured against.
  "$PYTHON" -m sft.predict --data-dir "$DATA_DIR" \
    --model "$model" --revision "$revision" --precision "$PRECISION" \
    --output-dir "$RUN_DIR/predictions/greedy-baseline" "${predict_args[@]}"
  for step in "${checkpoints[@]}"; do
    "$PYTHON" -m sft.predict --data-dir "$DATA_DIR" \
      --adapter "$SEED_DIR/checkpoint-$step" --precision "$PRECISION" \
      --output-dir "$RUN_DIR/predictions/greedy-checkpoint-$step" "${predict_args[@]}"
  done
fi

if [[ "$STAGE" != predict ]]; then
  methods=("greedy-baseline")
  for step in "${checkpoints[@]}"; do methods+=("greedy-checkpoint-$step"); done
  "$PYTHON" -m sft.judge_predictions \
    --run-dir "$RUN_DIR" --data-dir "$DATA_DIR" \
    --output-dir "${JUDGE_DIR:-$JUDGE_DIR_DEFAULT/greedy}" --methods "${methods[@]}" \
    --model "${JUDGE_MODEL:-openai/gpt-5.6-sol:floor}" \
    --reasoning-effort "${JUDGE_REASONING_EFFORT:-medium}" \
    --judge-instructions harness/prompts/judge_rubrics_long.md \
    --concurrency "${JUDGE_CONCURRENCY:-16}" \
    --max-attempts "${JUDGE_MAX_ATTEMPTS:-5}" \
    --max-output-tokens "${JUDGE_MAX_OUTPUT_TOKENS:-3000}" \
    --openrouter-max-prompt-price "${JUDGE_MAX_PROMPT_PRICE:-4}" \
    --openrouter-max-completion-price "${JUDGE_MAX_COMPLETION_PRICE:-20}" "${args[@]}" "${confirm[@]}"
fi
