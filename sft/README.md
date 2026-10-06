# SFT for Next Decision Prediction

This folder fine-tunes `Qwen/Qwen3-8B` with LoRA to predict a project's next research
decision from its previous decisions, producing the SFT cold start for RL (`rl/`). It takes three commands:

```bash
bash sft/run_sft.sh                          # prepare data and train the adapter
bash sft/run_evaluation_loss.sh               # held-out loss of every epoch checkpoint
bash sft/run_checkpoint_generation.sh all   # greedy predictions + Sol judging
```

RL starts from the epoch-3 checkpoint, `sft/runs/qwen3-8b/seed-42/checkpoint-72`.

## Data

Each case is a project's first 1, 2, or 3 decisions followed by the next one: 375 training cases from 125 projects and 402 evaluation cases from 134 projects.

- `splits/train_projects.json` lists the training projects: the harness's 19 NeurIPS 2025 training projects (`train` in `annotate/harness_splits.json`) and the 106 NeurIPS 2024 projects in `annotate/trajectory_selections.json`.
- `splits/eval_projects.json` lists the held-out projects, the same 134 as the harness's method-comparison cases (`test` in `annotate/harness_splits.json`).

The model answers in this schema:

```json
{"category": "method", "decision": "The next research decision."}
```

## Setup

From the repository root, with Python 3.10–3.12:

```bash
python3 -m venv sft/.venv
sft/.venv/bin/python -m pip install --upgrade pip
# A PyTorch wheel for your GPU driver (CUDA 12.6 here; see https://pytorch.org/get-started/previous-versions/#v280)
sft/.venv/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
sft/.venv/bin/pip install -r sft/requirements.txt
sft/.venv/bin/hf auth login   # if downloading Qwen/Qwen3-8B needs authentication
```

Each run uses one GPU. Change `PYTHON=/path/to/python` to select another environment.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 bash sft/run_sft.sh
CUDA_VISIBLE_DEVICES=0 bash sft/run_sft.sh --precision fp16   # if bf16 is unsupported
```

## Evaluate loss

```bash
CUDA_VISIBLE_DEVICES=0 bash sft/run_evaluation_loss.sh
bash sft/run_evaluation_loss.sh --dry-run   # check the inputs without loading a model
```

## Evaluate generation

```bash
bash sft/run_checkpoint_generation.sh predict   # GPU: greedy predictions of the base and every epoch
bash sft/run_checkpoint_generation.sh judge   # no GPU, require API Sol judgments
bash sft/run_checkpoint_generation.sh judge --dry-run   # prepare the judgments without submitting
```

The judge uses GPT-5.6 Sol at medium reasoning with `harness/prompts/judge_rubrics_long.md`.
