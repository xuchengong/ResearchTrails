#!/usr/bin/env bash
# Prefix-length plateau (paper Figure "Prefix information plateau"): Gemini predicts
# the next decision from no prefix, the last 1, 2, 3, 5, or 10 decisions, or the full
# prefix of 172 held-out cases, three repeats each, and Sol judges every point.
source jobs/common.sh

FULL=harness/experiments/prefix-length/setting.json
FULL_ROOT=harness/runs/prefix-length-gemini-3.1-flash-lite-full
NO_PREFIX_ROOT=harness/runs/prefix-length-gemini-3.1-flash-lite-no-prefix
LADDER_ROOT=harness/runs/prefix-length-gemini-3.1-flash-lite-last-k-sol-medium

# With REBUILD=1, first rebuild the setting from the annotations, in place; run it before any
# prefix-length runs exist, because existing runs keep their requests.
if [[ ${REBUILD:-0} == 1 ]]; then
    rm -rf harness/runs/prefix-length-base
    python "$RUNNER" prepare --output harness/runs/prefix-length-base/setting.json
    python "$RUNNER" prepare-full --base-setting harness/runs/prefix-length-base/setting.json \
        --output "$FULL" --methods baseline --replace
fi

# 1. Full prefix: every observed decision, the setting's only (baseline) method.
for REPEAT in "${REPEATS[@]}"; do
    predict "$RUNNER" --setting "$FULL" --run-dir "$FULL_ROOT-r$REPEAT"
done

# 2. No prefix: the prediction instructions alone.
for REPEAT in "${REPEATS[@]}"; do
    predict harness/prefix_length_experiment.py --no-prefix --setting "$FULL" --run-dir "$NO_PREFIX_ROOT-r$REPEAT"
done

# 3. The last k decisions. Sol judges these together with the no-prefix and full-prefix
#    predictions, so the whole curve shares one judge.
for REPEAT in "${REPEATS[@]}"; do
    predict harness/prefix_length_experiment.py --setting "$FULL" --run-dir "$LADDER_ROOT-r$REPEAT"
    judge harness/prefix_length_experiment.py --setting "$FULL" --run-dir "$LADDER_ROOT-r$REPEAT" \
        --no-prefix-run-dir "$NO_PREFIX_ROOT-r$REPEAT" --full-prefix-run-dir "$FULL_ROOT-r$REPEAT"
done
python harness/prefix_length_experiment.py summarize --setting "$FULL" --run-root "$LADDER_ROOT" --repeats 3
