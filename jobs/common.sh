# Shared setup for the harness jobs. Each job runs from the repository root and
# starts with `source jobs/common.sh`.
set -euo pipefail

source .venv/bin/activate
set -a
source .env
set +a

# Models of the paper, with OpenRouter price caps in USD per million tokens.
PREDICTOR=google/gemini-3.1-flash-lite
PREDICTOR_REASONING=medium
PREDICTOR_PRICES=(--openrouter-max-prompt-price 0.25 --openrouter-max-completion-price 1.50)
SOL=openai/gpt-5.6-sol:floor
SOL_REASONING=medium
SOL_PRICES=(--openrouter-max-prompt-price 4 --openrouter-max-completion-price 20)
CONCURRENCY=${CONCURRENCY:-16}
REPEATS=(1 2 3)
RUNNER=harness/experiment.py

# predict SCRIPT ARGS...: Gemini predictions through SCRIPT's run-predictions.
predict() {
    local script=$1
    shift
    python "$script" run-predictions "$@" \
        --model "$PREDICTOR" --reasoning-effort "$PREDICTOR_REASONING" "${PREDICTOR_PRICES[@]}" \
        --api-provider openrouter --concurrency "$CONCURRENCY" --confirm-submit
}

# judge SCRIPT ARGS...: Sol judgments through SCRIPT's run-judging.
judge() {
    local script=$1
    shift
    python "$script" run-judging "$@" \
        --model "$SOL" --reasoning-effort "$SOL_REASONING" "${SOL_PRICES[@]}" \
        --api-provider openrouter --concurrency "$CONCURRENCY" --confirm-submit
}

# run_setting SETTING RUN_DIR: predict and Sol-judge every case and method of a setting.
run_setting() {
    predict "$RUNNER" --setting "$1" --run-dir "$2"
    judge "$RUNNER" --setting "$1" --run-dir "$2"
}

# copy_input SOURCE DESTINATION: reuse a saved input, refusing to overwrite a different one.
copy_input() {
    if [[ -e $2 ]]; then
        cmp -s "$1" "$2" || { echo "$2 differs from $1" >&2; exit 1; }
    else
        mkdir -p "$(dirname "$2")"
        cp "$1" "$2"
    fi
}
