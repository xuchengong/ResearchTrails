#!/usr/bin/env bash
# Prepare the SFT data and train Qwen3-8B with LoRA
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a

PYTHON="${PYTHON:-sft/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-sft/data}"
RUN_DIR="${RUN_DIR:-sft/runs/qwen3-8b/seed-42}"
"$PYTHON" -m sft.prepare_data --train-projects sft/splits/train_projects.json --output-dir "$DATA_DIR"
"$PYTHON" -m sft.train --data-dir "$DATA_DIR" --output-dir "$RUN_DIR" --resume "$@"
