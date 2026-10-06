"""Small file operations shared by dataset preparation, training and inference."""

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
# Every evaluation uses the held-out first-1/2/3 cases (eval_early from sft.prepare_data).
EVAL_SPLIT = "early"


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError(f"empty dataset or duplicate case IDs: {path}")
    return rows


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def load_split(data_dir: Path, name: str) -> list[dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    info = manifest["datasets"][name]
    path = data_dir / info["file"]
    rows = read_jsonl(path)
    if len(rows) != info["cases"]:
        raise ValueError(f"dataset case count differs: {path}")
    return rows


def write_once(path: Path, content: str) -> None:
    """An experiment may resume, but never silently change its inputs."""
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"saved contents differ: {path}; choose a new output directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(content)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json_text(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
