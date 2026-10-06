# Harness for Next Decision Prediction

This folder compares different test-time approaches for held-out next-decision prediction.

## Data

`annotate/harness_splits.json` lists the NeurIPS 2025 repositories every experiment uses: `train` (19 training repositories), `test` (134 held-out repositories of the method comparison), and `prefix_length` (20 of the test repositories, for the prefix-length figure).

Each training data pairs an observed prefix with the held-out next decision. `build_prompt.py` defines what the predictor sees: the instructions in `prompts/next_decision_system.md`, followed by the annotated prefix. The prefix lists every decision introduced before the target, in `first_date` order. A decision still running when the target begins is labeled `active at cutoff`. A completed one is `retained by cutoff`, `abandoned by cutoff`, or `superseded by observed Txxx`; the supersession link is shown only when its replacement is already in the prefix (otherwise it appears as `retained by cutoff`). Decisions and targets keep the scientific action and drop retrospective outcome clauses. The paper title, abstract, trajectory insight, and later decisions are withheld. 

### Settings

Each folder in `experiments/` holds one setting:

| Setting | Held-out cases | Methods | Run by |
|---|---|---|---|
| `skills` |  402 data points: the first decisions after T000 (T001–T003) of 134 repositories | `baseline`, `skill`, `shuffled_skills`, `final_paper_skill` | `jobs/harness_table.sh` |
| `demos` | as above | `demonstrations` | `jobs/harness_table.sh` |
| `rag` | as above | `retrieved_two_ordered`, `random_two_ordered` | `jobs/harness_table.sh` |
| `prefix-length` | 172 later decisions of 20 repositories, with their full history | `baseline` | `jobs/prefix_length.sh` |
| `judge-quality` | 100 decisions of 9 repositories | `baseline` | `jobs/judge_quality.sh` |

### Prefix-length figure

`experiments/prefix-length/` is for the score vs. prefix-length figure. Its 172 cases come from the 20 `prefix_length` repositories in `annotate/harness_splits.json`. `jobs/prefix_length.sh` predicts each target from no prefix, the last 1, 2, 3, 5, or 10 decisions, or the full history.


### Method comparisons

All methods are compared on the same 402 cases. The 134 held-out repositories are the `test` list: every annotated NeurIPS 2025 repository except the 19 training repositories and two license-sensitive ones. Each repository contributes its second, third, and fourth decisions (T001–T003) as targets. Each target is predicted from every decision before it, so the model sees 1, 2, or 3 decisions.

`experiments/cases/` holds these cases and judge prompts. `skills`, `demos`, and `rag` copy them. The ordered and shuffled skills and the demonstrations are built from the training trajectories' first 1–3 decisions to match these prefixes (`skills_experiment.py`, `demos_experiment.py`). The final paper skill is distilled from the same projects' published papers (`skills_experiment.py`; see [Skills, demos, and RAG](#skills-demos-and-rag)). `jobs/harness_table.sh` runs them for the paper's method-comparison table.


### Judge quality

`experiments/judge-quality/` holds the 100 cases of the judge variance and calibration studies (run by `jobs/judge_quality.sh` and the [Opus 5 judge grid](#opus-5-judge-grid)). Its `selection` records how they were chosen from an earlier 146-case set.


## Experiments

Run each job from the repository root. Jobs use the paper's models (`jobs/common.sh`), run three repeats, and make paid calls; rerun an interrupted job to resume it.

```bash
set -a; source .env; set +a        # load OPENROUTER_API_KEY
CONCURRENCY=8 bash jobs/<job>.sh   # optional: parallel requests per step (default 16)
```

### Prefix length

```bash
bash jobs/prefix_length.sh

# Optional: first rebuild experiments/prefix-length/ from the annotations (before any prefix-length runs exist)
REBUILD=1 bash jobs/prefix_length.sh

# Optional: regenerate the trajectory demos and skills the setting embeds (its baseline method does not use them)
python harness/render_demonstrations.py
python harness/render_demonstrations.py --shuffled   # keep the default seed: the shuffled prefix skill reuses its permutations
python harness/distill_skill.py --run-dir harness/runs/skill-distillation --model anthropic/claude-opus-5 \
  --reasoning-effort high --max-output-tokens 8000 \
  --openrouter-max-prompt-price 5 --openrouter-max-completion-price 25 --confirm-submit
```

### Skills, demos, and RAG

```bash
bash jobs/harness_table.sh
# Each summary's table.md holds its rows of the paper's table (mean ± SD over the 3 repeats):
# skills -> harness/runs/skills-summary/   Baseline, Skills, Shuffled skill, and Final paper skill
# demos  -> harness/runs/demos-summary/    All demos
# rag    -> harness/runs/rag-summary/      Random two and Retrieve two demos

# Optional: first distill new ordered, shuffled, and final paper skills with Opus 5,
# replacing those in harness/skills/ before any skills runs exist
DISTILL=1 bash jobs/harness_table.sh
```

### Judge quality

```bash
bash jobs/judge_quality.sh
# variance, Sol judging 3×   -> harness/runs/judge-variance-baseline-r2-sol-medium-analysis/
# calibration, Sol           -> harness/runs/judge-calibration-opus5-sol-medium/results/
```

### Opus 5 judge grid

`judge_grid.py` re-judges the reported predictions with Opus 5 (medium reasoning).

| Part | Predictions | Saved Sol judgments from |
| --- | --- | --- |
| `harness-r1`…`harness-r3` | the `skills`, `demos`, and `rag` settings, 7 methods × 402 per repeat | `jobs/harness_table.sh` |
| `rl` | Qwen3-8B base, SFT initialization, GRPO epochs 7 and 10 (402 each) | the Section 3 training runs |
| `calibration` | the 400 Opus-generated controls | `jobs/judge_quality.sh` |

```bash
# After the jobs above and the RL evaluation runs of Section 3
bash jobs/judge_grid.sh
# main grid      -> harness/rejudging/judge-grid/
# judge variance -> harness/rejudging/judge-variance/
```
