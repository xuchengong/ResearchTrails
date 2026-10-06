#!/usr/bin/env python3
"""Prepare and summarize the demos setting: insight-free first-1/2/3 demonstrations."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import build_prompt
import experiment
import render_demonstrations


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SOURCE_SETTING = (
    HERE / "experiments" / "cases" / "setting.json"
)
SKILLS_SETTING = (
    HERE / "experiments" / "skills" / "setting.json"
)
DEFAULT_SETTING = (
    HERE / "experiments" / "demos" / "setting.json"
)
DEFAULT_RUN_ROOT = (
    HERE
    / "runs"
    / "demos-gemini-3.1-flash-lite-sol-medium"
)
SKILLS_RUN_ROOT = (
    HERE
    / "runs"
    / "skills-gemini-3.1-flash-lite-sol-medium"
)
DEFAULT_SUMMARY_DIR = HERE / "runs" / "demos-summary"
PREFIX_LENGTHS = (1, 2, 3)
METHODS = ("demonstrations",)
# All demos is compared with the skills setting's Baseline and Skills runs.
CONTRASTS = (("demonstrations", "baseline"), ("demonstrations", "skill"))


def case_signature(case: dict) -> tuple:
    return (
        case["case_id"],
        case["evaluation_idx"],
        case["target_time_step_id"],
        case["observed_decision_count"],
        case["target"]["decision_id"],
        case["target"]["category"],
        case["target"]["decision"],
    )


def validate_existing_setting(output: Path, source_setting_path: Path) -> None:
    setting = experiment.load_setting(output)
    assets = experiment.load_setting_assets(output, setting)
    if setting.get("source_setting") != experiment.recorded_path(source_setting_path):
        raise ValueError(f"existing demo setting has a different source: {output}")
    if tuple(setting["methods"]) != METHODS:
        raise ValueError(f"existing demo setting has different methods: {output}")
    if "trajectory insight" in assets["demonstrations"].lower():
        raise ValueError("prepared demonstrations unexpectedly contain trajectory insight")
    print(f"already prepared: {output} ({setting['case_count']} cases)")


def render_example(
    trajectory_number: int,
    prefix_length: int,
    decisions: list[dict],
    order: list[int],
) -> str:
    if sorted(order) != list(range(prefix_length + 1)):
        raise ValueError("example order must permute exactly the first k+1 decisions")
    lines = [
        f"## Training trajectory {trajectory_number:03d}",
        f"### Prefix length {prefix_length}",
        "",
        "Observed decisions, from oldest to newest:",
        "",
    ]
    for displayed_position, original_position in enumerate(order[:-1]):
        decision = decisions[original_position]
        lines.extend(
            [
                f"- T{displayed_position:03d} | {decision['category']}",
                "  "
                + render_demonstrations.decision_action_text(
                    decision["decision"]
                ),
                "",
            ]
        )
    target = decisions[order[-1]]
    lines.extend(
        [
            "Actual immediate next decision:",
            f"Category: {target['category']}",
            "Decision: "
            + render_demonstrations.decision_action_text(target["decision"]),
        ]
    )
    return "\n".join(lines)


def render_corpus(training_indices: tuple[int, ...]) -> tuple[str, dict]:
    header = [
        "# Natural early-prefix next-decision demonstrations",
        "",
        "Each solved example shows the complete one-, two-, or three-decision "
        "history available near the beginning of a project, followed by the "
        "immediate next decision.",
    ]
    ordered_sections = list(header)
    examples = []
    sources = []
    for trajectory_number, index in enumerate(training_indices, 1):
        annotation_path = render_demonstrations.annotation_path(index)
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        decisions = build_prompt.validate_annotation(annotation_path, annotation)
        if len(decisions) <= max(PREFIX_LENGTHS):
            raise ValueError(f"idx={index} needs at least four decisions")
        sources.append(
            {
                "training_idx": index,
                "annotation": annotation_path.relative_to(REPO_ROOT).as_posix(),
            }
        )
        for prefix_length in PREFIX_LENGTHS:
            ordered = list(range(prefix_length + 1))
            ordered_sections.extend(
                [
                    "",
                    render_example(
                        trajectory_number,
                        prefix_length,
                        decisions,
                        ordered,
                    ),
                ]
            )
            examples.append(
                {
                    "training_idx": index,
                    "trajectory_number": trajectory_number,
                    "prefix_length": prefix_length,
                    "source_positions": ordered,
                    "ordered_presentation": ordered,
                }
            )

    ordered_text = "\n".join(ordered_sections).rstrip() + "\n"
    if "trajectory insight" in ordered_text.lower():
        raise ValueError("demonstrations contain a trajectory insight")
    metadata = {
        "format_version": 1,
        "outcome_rule": "outcomes and supersession links are omitted",
        "trajectory_insight_rule": "omitted",
        "training_indices": list(training_indices),
        "example_count": len(examples),
        "sources": sources,
        "examples": examples,
    }
    return ordered_text, metadata


def prepare_suite(source_setting_path: Path, output: Path) -> None:
    source_setting = experiment.load_setting(source_setting_path)
    if output.exists():
        validate_existing_setting(output, source_setting_path)
        return
    if {
        case.get("natural_prefix_length") for case in source_setting["cases"]
    } != set(PREFIX_LENGTHS):
        raise ValueError("source setting is not a natural first-1/2/3 suite")

    training_indices = render_demonstrations.TRAIN_INDICES
    if set(source_setting["evaluation_indices"]) != set(render_demonstrations.TEST_INDICES):
        raise ValueError(
            "source setting projects differ from the test list in annotate/harness_splits.json"
        )
    ordered, rendering = render_corpus(training_indices)
    if rendering["example_count"] != len(training_indices) * len(PREFIX_LENGTHS):
        raise ValueError("rendered demonstration count is incomplete")

    source_assets = experiment.load_setting_assets(
        source_setting_path, source_setting
    )
    assets = {
        "prediction_instructions": source_assets["prediction_instructions"],
        "demonstrations": ordered,
        "judge_instructions": source_assets["judge_instructions"],
    }
    filenames = {
        "prediction_instructions": "prediction_instructions.md",
        "demonstrations": "demonstrations.md",
        "judge_instructions": "judge_instructions.md",
    }
    cases = copy.deepcopy(source_setting["cases"])

    output.parent.mkdir(parents=True)
    asset_dir = output.parent / "assets"
    asset_dir.mkdir()
    for name, text in assets.items():
        (asset_dir / filenames[name]).write_text(text, encoding="utf-8")

    setting = {
        "format_version": 1,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_setting": experiment.recorded_path(source_setting_path),
        "evaluation_indices": copy.deepcopy(
            source_setting["evaluation_indices"]
        ),
        "methods": list(METHODS),
        "selection": {
            **copy.deepcopy(source_setting["selection"]),
            "experiment": (
                "natural first-1/2/3 with insight-free matched demos"
            ),
            "demonstration_training_regime": (
                "first-k observed decisions mapped to the immediate next decision"
            ),
        },
        "demonstration_rendering": rendering,
        "case_count": len(cases),
        "prediction_request_count": len(cases) * len(METHODS),
        "judge_request_count": len(cases) * len(METHODS),
        "assets": {name: {"path": f"assets/{filenames[name]}"} for name in assets},
        "sources": copy.deepcopy(source_setting["sources"]),
        "cases": cases,
    }
    experiment.write_json(output, setting)
    print(
        f"prepared {len(cases)} cases x {len(METHODS)} demo methods "
        f"at {output}"
    )


def summarize(args: argparse.Namespace) -> None:
    setting = experiment.load_setting(args.setting)
    skills_setting = experiment.load_setting(args.skills_setting)
    if setting.get("source_setting") != skills_setting.get("source_setting") or sorted(
        map(case_signature, setting["cases"])
    ) != sorted(map(case_signature, skills_setting["cases"])):
        raise ValueError("demos and skills settings contain different cases")
    experiment.summarize_comparison(
        args.setting,
        setting,
        METHODS,
        ((args.skills_run_root, ("baseline", "skill")), (args.run_root, METHODS)),
        CONTRASTS,
        args.repeats,
        args.output_dir,
        {"demonstration_design": setting["demonstration_rendering"]},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--source-setting", type=Path, default=SOURCE_SETTING
    )
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_SETTING)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--setting", type=Path, default=DEFAULT_SETTING)
    summarize_parser.add_argument("--skills-setting", type=Path, default=SKILLS_SETTING)
    summarize_parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    summarize_parser.add_argument("--skills-run-root", type=Path, default=SKILLS_RUN_ROOT)
    summarize_parser.add_argument("--output-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    summarize_parser.add_argument("--repeats", type=int, default=3)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_suite(args.source_setting, args.output)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
