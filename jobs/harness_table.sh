#!/usr/bin/env bash
# Harness table (paper Table "Decision similarity ... based on previous {1,2,3}
# decisions"): Gemini predicts the next decision of 134 held-out projects from their
# first 1, 2, or 3 decisions, three repeats per method, judged by Sol.
source jobs/common.sh

EXPERIMENTS=harness/experiments
DISTILL_DIR=harness/runs/skill-distillation-prefix-regimes-opus5
CASES=$EXPERIMENTS/cases/setting.json
SKILLS=$EXPERIMENTS/skills/setting.json
DEMOS=$EXPERIMENTS/demos/setting.json
RAG=$EXPERIMENTS/rag/setting.json
SKILLS_ROOT=harness/runs/skills-gemini-3.1-flash-lite-sol-medium
DEMOS_ROOT=harness/runs/demos-gemini-3.1-flash-lite-sol-medium
RAG_SELECTION_DIR=harness/runs/rag-selector-gemini-3.1-flash-lite-k2
RAG_ROOT=harness/runs/rag-gemini-3.1-flash-lite-sol-medium

# 1. Baseline, Skills, Shuffled skill, Final paper skill. The ordered and shuffled skills
#    are distilled from the training trajectories' first-1/2/3 prefixes in their true and
#    shuffled order; the final paper skill from the same projects' published papers (see
#    "Final paper skill" in harness/README.md). The setting ships with the paper's three
#    skills, which this step reuses by default. With DISTILL=1, Opus 5 first distills new
#    skills, which overwrite those in harness/skills/ and in the re-prepared setting; run it
#    before any skills runs exist, because existing runs keep their requests.
if [[ ${DISTILL:-0} == 1 ]]; then
    python harness/skills_experiment.py render --output-dir "$DISTILL_DIR/inputs"
    python harness/skills_experiment.py distill --run-dir "$DISTILL_DIR" \
        --model anthropic/claude-opus-5 --reasoning-effort high --max-output-tokens 12000 \
        --openrouter-max-prompt-price 5 --openrouter-max-completion-price 25 \
        --concurrency 2 --confirm-submit
    # Also re-prepares the setting, now with all three new skills.
    python harness/skills_experiment.py distill-final-paper --confirm-submit
else
    python harness/skills_experiment.py prepare
fi
for REPEAT in "${REPEATS[@]}"; do
    run_setting "$SKILLS" "$SKILLS_ROOT-r$REPEAT"
done
# The Baseline, Skills, Shuffled skill, and Final paper skill rows.
python harness/skills_experiment.py summarize --run-root "$SKILLS_ROOT" --repeats 3

# 2. All demos: every training trajectory's first 1-3 decisions as demonstrations. The
#    summary compares it with step 1's Baseline and Skills runs.
python harness/demos_experiment.py prepare --source-setting "$CASES" --output "$DEMOS"
for REPEAT in "${REPEATS[@]}"; do
    run_setting "$DEMOS" "$DEMOS_ROOT-r$REPEAT"
done
python harness/demos_experiment.py summarize \
    --setting "$DEMOS" --run-root "$DEMOS_ROOT" --skills-run-root "$SKILLS_ROOT" --repeats 3

# 3. Random two demos and Retrieve two demos. The setting ships with its selections;
#    without it, Gemini first selects two same-length demonstrations per case.
if [[ ! -f $RAG ]]; then
    python harness/rag_experiment.py select \
        --base-setting "$DEMOS" --selection-dir "$RAG_SELECTION_DIR" --max-output-tokens 2000 \
        --model "$PREDICTOR" --reasoning-effort "$PREDICTOR_REASONING" "${PREDICTOR_PRICES[@]}" \
        --api-provider openrouter --concurrency "$CONCURRENCY" --confirm-submit
    python harness/rag_experiment.py prepare \
        --base-setting "$DEMOS" --selections "$RAG_SELECTION_DIR/selections.json" --output "$RAG"
fi
for REPEAT in "${REPEATS[@]}"; do
    run_setting "$RAG" "$RAG_ROOT-r$REPEAT"
done
python harness/rag_experiment.py summarize \
    --setting "$RAG" --base-setting "$DEMOS" --run-root "$RAG_ROOT" \
    --skills-run-root "$SKILLS_ROOT" --demos-run-root "$DEMOS_ROOT" --repeats 3
