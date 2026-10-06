# RL for Next Decision Prediction

GRPO starts from the SFT run's epoch-3 checkpoint, `sft/runs/qwen3-8b/seed-42/checkpoint-72`. 

At startup the SFT adapter is merged into the original Qwen3-8B weights, the merged weights are
frozen. A LoRA adapter is then trained on top. Each collection round samples eight decisions per training prefix from the current policy, judges each against the training annotation, and updates the adapter with Dr. GRPO.

`run_grpo.sh` trains for ten epochs on one or multiple GPU(s); `run_evaluation.sh` evaluates every epoch on the held-out inputs.

## Run

On the `sft/.venv` environment:

```bash
# Dry run: validate and save the run configuration; no GPU or API calls
bash rl/run_grpo.sh --dry-run

# Train on one or several GPU(s)
bash rl/run_grpo.sh
GPUS=4 bash rl/run_grpo.sh

# Held-out evaluation of every epoch (--dry-run prepares the judgments without submitting)
bash rl/run_evaluation.sh
```

To start from another SFT checkpoint, pass it with a new run directory, and evaluate that directory:

```bash
SFT_CHECKPOINT=sft/runs/qwen3-8b/seed-42/checkpoint-48 RUN_DIR=rl/runs/grpo-from-epoch2 bash rl/run_grpo.sh
RUN_DIR=rl/runs/grpo-from-epoch2 bash rl/run_evaluation.sh
```

To keep training an existing adapter instead of a fresh one, pass its checkpoint and a
new run directory:

```bash
CONTINUE_FROM=rl/runs/grpo-lr1e-4/checkpoint-epoch-10 RUN_DIR=rl/runs/grpo-lr1e-4-more bash rl/run_grpo.sh
```

## Settings

| Setting | Paper runs |
| --- | --- |
| Initialization | Merge `seed-42/checkpoint-72` into Qwen3-8B (pinned revision, BF16), then attach a LoRA with the SFT rank, alpha and target modules (rank 8, alpha 16), dropout 0 |
| Training data | The SFT run's training split: 375 first-1/2/3 prefixes from 125 trajectories |
| Epochs | 10 |
| Candidates per prefix | 8 sampled completions |
| Batch | 8 informative prefix groups (64 completions) per optimizer update |
| Optimizer | AdamW, constant learning rate `1e-4`, gradient norm cap 1 |
| Sampling | Temperature 0.8, top-p 1, at most 512 new tokens, thinking disabled |
| Judge | Sol medium (`openai/gpt-5.6-sol:floor`), `harness/prompts/judge_rubrics_long.md`, $4/$20 per-million price ceilings, Azure excluded |
| Reward | `component_match + operation_match`, range 0-4; invalid JSON and truncated outputs get 0 |
| Advantage | Group reward minus the group mean, without standard-deviation scaling (Dr. GRPO) |
| Loss | Completion-token sum over a constant normalizer of 64; completions weigh equally within a group |
| KL | Beta 0.04 against the frozen merged SFT model, estimator `exp(r) - r - 1` with `r = logp_ref - logp` |
| Clipping | Epsilon 0.2 |
| Seed | 42 |

Groups whose rewards are all equal carry no signal and are skipped, and collection
continues until the batch has eight informative groups. 
