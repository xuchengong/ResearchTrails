#!/usr/bin/env bash
# Judge variance and calibration (paper Appendix B, Tables "Standard deviation and
# Fleiss' Kappa" and "Judge's scores match the intended scores"): the Sol rows. The Opus 5
# rows of both tables come from harness/judge_grid.py, which reuses this job's saved Sol
# requests.
source jobs/common.sh

BASELINE=harness/experiments/judge-quality/setting.json
# The fixed 100 baseline predictions that the variance study rejudges. The run keeps
# its original repeat number, which the variance runs below are named after.
FIXED=harness/runs/judge-quality-r2
VARIANCE_ROOT=harness/runs/judge-variance-baseline-r2-sol-medium
CALIBRATION=harness/runs/judge-calibration-opus5-sol-medium

# 1. Baseline predictions by GPT-5.6 Luna on the judge-quality cases. The setting ships
#    prepared; its selection records how the 100 cases were chosen.
python "$RUNNER" run-predictions --setting "$BASELINE" --run-dir "$FIXED" \
    --model openai/gpt-5.6-luna:floor --reasoning-effort medium \
    --openrouter-max-prompt-price 0.2 --openrouter-max-completion-price 1.2 \
    --api-provider openrouter --concurrency "$CONCURRENCY" --confirm-submit

# 2. Judge variance: Sol judges the fixed predictions three times.
for REPEAT in "${REPEATS[@]}"; do
    copy_input "$FIXED/run.json" "$VARIANCE_ROOT-r$REPEAT/run.json"
    copy_input "$FIXED/prediction_output.jsonl" "$VARIANCE_ROOT-r$REPEAT/prediction_output.jsonl"
    judge "$RUNNER" --setting "$BASELINE" --run-dir "$VARIANCE_ROOT-r$REPEAT"
done
python harness/judge_quality.py variance --run-root "$VARIANCE_ROOT" \
    --repeats 3 --method baseline --output-dir "$VARIANCE_ROOT-analysis"

# 3. Calibration: Opus 5 writes exact, paraphrased, adjacent, and unrelated decisions for
#    the same cases (prompt in the appendix), and Sol judges them. judge_grid.py turns
#    these saved Sol requests into the Opus calibration requests.
python harness/judge_quality.py calibrate --setting "$BASELINE" --run-dir "$CALIBRATION" \
    --generator-model anthropic/claude-opus-5 --generator-reasoning-effort high \
    --generator-max-output-tokens 4000 --generator-concurrency 4 \
    --generator-max-prompt-price 5 --generator-max-completion-price 25 \
    --judge-model "$SOL" --judge-reasoning-effort "$SOL_REASONING" --judge-max-output-tokens 3000 \
    --judge-concurrency "$CONCURRENCY" --judge-max-prompt-price 4 --judge-max-completion-price 20 \
    --confirm-submit
