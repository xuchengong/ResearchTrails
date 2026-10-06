"""Prepare project-disjoint SFT data; no ML dependencies needed."""

import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from sft.io import HERE, ROOT, json_text
from sft.render import action_text, read_decisions, render_prefix


def load_projects(catalog: Path, root: Path) -> list[dict]:
    projects = []
    for entry in json.loads(catalog.read_text())["projects"]:
        path = (root / entry["annotation"]).resolve()
        annotation = json.loads(path.read_text())
        github = urlparse(annotation["source"]["github"])
        if github.hostname != "github.com" or len(github.path.strip("/").split("/")) != 2:
            raise ValueError(f"expected a GitHub repository URL: {path}")
        repository = github.path.strip("/").removesuffix(".git").lower()
        arxiv = urlparse(annotation["source"]["arxiv"])
        if arxiv.hostname not in {"arxiv.org", "www.arxiv.org"} or not arxiv.path.startswith("/abs/"):
            raise ValueError(f"expected an arXiv abstract URL: {path}")
        paper = re.sub(r"v\d+$", "", arxiv.path.removeprefix("/abs/").rstrip("/"))
        if not entry["project_id"] or not paper:
            raise ValueError(f"missing project/paper identity: {path}")
        projects.append({
            **entry, "repository": repository, "paper": paper,
            "decisions": read_decisions(path),
        })
    if not projects:
        raise ValueError(f"empty project catalog: {catalog}")
    return projects


def check_disjoint(train: list[dict], evaluation: list[dict]) -> None:
    # Numeric indices are venue-local: repository/paper checks also catch
    # cross-conference duplicates when additional training projects arrive.
    for key in ("project_id", "repository", "annotation"):
        values = [project[key] for project in train + evaluation]
        if len(set(values)) != len(values):
            duplicates = sorted(value for value in set(values) if values.count(value) > 1)
            raise ValueError(f"duplicate project or train/eval overlap by {key}: {duplicates}")
    overlap = {p["paper"] for p in train} & {p["paper"] for p in evaluation}
    if overlap:
        raise ValueError(f"train/eval overlap by paper: {sorted(overlap)}")


def make_rows(projects: list[dict], system_prompt: str) -> list[dict]:
    rows = []
    for project in projects:
        decisions = project["decisions"]
        if len(decisions) < 2:
            raise ValueError(f"no next-decision targets: {project['project_id']}")
        for step in range(1, len(decisions)):
            rows.append({
                "case_id": f"{project['project_id']}:T{step:03d}",
                "project_id": project["project_id"],
                "target_step": step,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": render_prefix(decisions, step)},
                ],
                "target": {
                    "category": decisions[step]["category"],
                    "decision": action_text(decisions[step]["decision"]),
                },
            })
    return rows


def prepare(args) -> dict:
    train = load_projects(args.train_projects, args.root)
    evaluation = load_projects(args.eval_projects, args.root)
    check_disjoint(train, evaluation)
    prompt = args.prompt.read_text().strip()
    if not prompt:
        raise ValueError("empty system prompt")
    train_rows = make_rows(train, prompt)
    eval_rows = make_rows(evaluation, prompt)
    # Exactly the same early cases for the base model and every checkpoint.
    datasets = {
        "train_early_only": [r for r in train_rows if r["target_step"] <= 3],
        "eval_early": [r for r in eval_rows if r["target_step"] <= 3],
    }
    for name, rows in datasets.items():
        if not rows:
            raise ValueError(f"empty dataset: {name}")
    # Write only after all annotations and both split boundaries pass validation.
    manifest = {
        "prefix": "v1, complete preceding decisions; no minimum-completed filter",
        "output_fields": ["category", "decision"],
        "projects": {
            split: [{k: v for k, v in p.items() if k != "decisions"} for p in projects]
            for split, projects in (("train", train), ("eval", evaluation))
        },
        "datasets": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in datasets.items():
        path = args.output_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest["datasets"][name] = {
            "file": path.name, "cases": len(rows),
            "projects": len({row["project_id"] for row in rows}),
        }
    (args.output_dir / "manifest.json").write_text(json_text(manifest))
    for name, info in manifest["datasets"].items():
        print(f"{name}: {info['cases']} cases, {info['projects']} projects")
    print(f"Dataset: {args.output_dir.resolve()}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--train-projects", type=Path, default=HERE / "splits/train_projects.json")
    parser.add_argument("--eval-projects", type=Path, default=HERE / "splits/eval_projects.json")
    parser.add_argument("--prompt", type=Path, default=HERE / "prompts/next_decision.md")
    parser.add_argument("--output-dir", type=Path, default=HERE / "data")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
