# ResearchTrails: Learning Scientific Exploration from Human Research Decisions Trajectories

<div align="center">
  <a href="https://arxiv.org"><b>Paper</b></a> •
  <a href="https://huggingface.co/researchtrails"><b>Data & Models</b></a> •
  <a href="https://www.researchtrails.org"><b>Blog</b></a> •
  <a href="https://www.researchtrails.org/visualizer"><b>Data Visualizer</b></a>
</div>
<br>

Papers record the final proposed method and experiment results, not how the authors get there. We turn the **Git commit history** of a paper into a **human research trajectory** and present a dataset containing 599 research trajectories made of 13K evidence-grounded **research decisions**. We show that learning from these trajectories, as demonstrations, distilled skills or as training data, helps models propose the next research decision better than learning from final papers alone.

<div align="center">
<img src="assets/overview.svg" alt="ResearchTrails" width="800"/>
</div>

## Overview

| Folder | Paper | Contents |
|---|---|---|
| [`annotate/`](annotate/) | §3 | Build trajectories from repositories: paper/repository discovery, filtering, commit evidence, decision extraction and annotation. |
| [`harness/`](harness/) | §5.1 | Next-decision prediction with test-time harnesses (demonstrations, retrieved demonstrations, distilled skills), the decision-similarity judge, user simulations, and judge-quality studies. |
| [`sft/`](sft/) | §5.2 | Supervised fine-tuning of Qwen3-8B (LoRA) on decision sequences, prediction, and judging. |
| [`rl/`](rl/) | §5.2| RL from the SFT checkpoint with decision-similarity rewards (Dr. GRPO) and its per-epoch evaluation. |
| [`jobs/`](jobs/) | | Launch scripts for the annotation pipeline and the harness experiments reported in the paper. |

Setup and usage details are in each folder's README.

## Quick Start

1. **Install** (Python 3.12)
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. **Add API keys**
   ```bash
   cp .env.example .env   # fill in OPENROUTER_API_KEY; GITHUB_TOKEN is only needed to build trajectories
   ```

3. **Download the trajectories**
   ```bash
   .venv/bin/hf download researchtrails/annotations --repo-type dataset --local-dir annotations
   ```

4. **Run an experiment**, e.g. the harness table (paid API calls)
   ```bash
   bash jobs/harness_table.sh   # tables in harness/runs/{skills,demos,rag}-summary/table.md
   ```

To reproduce the paper, run the following commands from the repository root. Commands that call models through OpenRouter make paid API calls.

### Building trajectories (§3)

```bash
# Discover NeurIPS 2025 paper/repository pairs, filter them, collect commit evidence, and then annotate (needs GITHUB_TOKEN and git token)
VENUE=NeurIPS YEAR=2025 bash jobs/annotate.sh
```

See [`annotate/README.md`](annotate/README.md) for running single stages, the filtering rules, and the output format.

### Next-decision prediction with harnesses (§5.1, Appendix B)

```bash
# Prefix-length plateau: predict from no prefix, the last 1, 2, 3, 5, or 10 decisions, or the full history
bash jobs/prefix_length.sh     # -> harness/runs/prefix-length-gemini-3.1-flash-lite-last-k-sol-medium-summary/

# Harness table: Baseline, All demos, Random/Retrieve two demos, Skills, Shuffled skill, and
# Final paper skill on the 134 held-out projects, 3 repeats each
bash jobs/harness_table.sh     # -> harness/runs/{skills,demos,rag}-summary/table.md

# Judge variance and calibration: Sol judges 100 fixed predictions 3 times, and Opus-written controls
bash jobs/judge_quality.sh     # -> harness/runs/judge-variance-*-analysis/, harness/runs/judge-calibration-*/results/

# Opus 5 re-judges every reported prediction; run after the jobs above and the RL evaluation below
bash jobs/judge_grid.sh        # -> harness/rejudging/{judge-grid,judge-variance}/

# User simulation: GPT-5.6 Sol answers a research question with and without a research pattern
python3 harness/user_simulation/run.py bug_repair     # -> harness/user_simulation/runs/bug_repair/COMPARISON.md
python3 harness/user_simulation/run.py answer_judge   # -> harness/user_simulation/runs/answer_judge/COMPARISON.md
```

See [`harness/README.md`](harness/README.md) for details. 

### Training a next-decision policy (§5.2)

SFT and RL run on GPUs in their own environment, `sft/.venv` (the paper used one L40S for SFT and four for RL). Set it up first as in [`sft/README.md`](sft/README.md#setup).

```bash
# SFT: LoRA on Qwen3-8B over the 125 training projects -> sft/runs/qwen3-8b/seed-42/
bash sft/run_sft.sh
bash sft/run_evaluation_loss.sh              # held-out loss of every epoch checkpoint
bash sft/run_checkpoint_generation.sh all    # greedy predictions of the base model and every epoch, judged by Sol

# RL: Dr. GRPO from the SFT epoch-3 checkpoint, rewarded by Sol's judgment -> rl/runs/grpo-lr1e-4/
GPUS=4 bash rl/run_grpo.sh
bash rl/run_evaluation.sh                    # greedy predictions of the SFT start and every RL epoch, judged by Sol
```

See [`sft/README.md`](sft/README.md) and [`rl/README.md`](rl/README.md) for the data, settings, and other starting checkpoints. The paper's SFT epoch-3 and GRPO epoch-7 adapters are released as [researchtrails/qwen3-8b-sft](https://huggingface.co/researchtrails/qwen3-8b-sft) and [researchtrails/qwen3-8b-grpo](https://huggingface.co/researchtrails/qwen3-8b-grpo).

## Citation

```bibtex
@article{gong2026learning,
  title   = {Learning Scientific Exploration from Human Research Decisions Trajectories},
  author  = {Gong, Xuchen and Gu, Shane and Liu, Haokun and Yao, Dixi and Tan, Chenhao and Li, Tian},
  journal = {arXiv preprint arXiv:2610.07184},
  year    = {2026}
}
```
