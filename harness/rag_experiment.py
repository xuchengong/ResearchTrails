#!/usr/bin/env python3
"""Retrieve two same-length demos per case and compare them with two random demos.

Predictions and judgments of the prepared setting run through experiment.py.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import experiment
import demos_experiment


HERE = Path(__file__).resolve().parent
BASE_SETTING = (
    HERE / "experiments" / "demos" / "setting.json"
)
SKILLS_SETTING = (
    HERE / "experiments" / "skills" / "setting.json"
)
DEFAULT_SELECTION_DIR = (
    HERE
    / "runs"
    / "rag-selector-gemini-3.1-flash-lite-k2"
)
DEFAULT_SETTING = (
    HERE / "experiments" / "rag" / "setting.json"
)
DEFAULT_RUN_ROOT = (
    HERE
    / "runs"
    / "rag-gemini-3.1-flash-lite-sol-medium"
)
SKILLS_RUN_ROOT = (
    HERE
    / "runs"
    / "skills-gemini-3.1-flash-lite-sol-medium"
)
DEMOS_RUN_ROOT = (
    HERE
    / "runs"
    / "demos-gemini-3.1-flash-lite-sol-medium"
)
DEFAULT_SUMMARY_DIR = HERE / "runs" / "rag-summary"
PREFIX_LENGTHS = (1, 2, 3)
RETRIEVAL_K = 2
DEFAULT_RANDOM_SEED = 20260915
METHODS = ("retrieved_two_ordered", "random_two_ordered")
# Compared with each other, the skills setting's Baseline and Skills runs, and the All demos runs.
CONTRASTS = (
    ("retrieved_two_ordered", "random_two_ordered"),
    ("retrieved_two_ordered", "baseline"),
    ("random_two_ordered", "baseline"),
    ("retrieved_two_ordered", "demonstrations"),
    ("retrieved_two_ordered", "skill"),
)
SELECTOR_INSTRUCTIONS = """You are the retrieval stage for next-research-decision prediction near the natural beginning of a project.

The held-out project has exactly one, two, or three observed decisions. Select exactly two solved training examples with the same prefix length that are most instructive for predicting its next scientific decision. Prioritize similarity in the scientific object being developed, the operation just applied, the apparent early-project stage, and especially the demonstrated transition to the next operation. Topical similarity without a relevant transition is weak evidence.

Rank the more useful example first. The two example IDs must differ. Use only the supplied candidate examples and held-out observable prefix. Do not predict or state the held-out project's next decision. Return the required JSON only."""


def candidate_example_ids(blocks: dict[tuple[str, int], str]) -> list[str]:
    ids = sorted({example_id for example_id, _ in blocks})
    expected = {(example_id, length) for example_id in ids for length in PREFIX_LENGTHS}
    if set(blocks) != expected or len(ids) < RETRIEVAL_K:
        raise ValueError("demonstration datastore is not complete at every prefix length")
    return ids


def candidate_catalog(
    document: str, prefix: int
) -> tuple[str, list[str]]:
    header, blocks = experiment.example_demonstrations(document)
    ids = candidate_example_ids(blocks)
    return "\n\n".join([header, *(blocks[(example_id, prefix)] for example_id in ids)]), ids


def observable_prefix(evaluation_input: str) -> str:
    if not evaluation_input.endswith(experiment.PREDICTION_SUFFIX):
        raise ValueError("evaluation input has an unexpected prediction suffix")
    return evaluation_input[: -len(experiment.PREDICTION_SUFFIX)].rstrip()


def selector_schema(example_ids: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["primary_example_id", "secondary_example_id", "rationale"],
        "properties": {
            "primary_example_id": {"type": "string", "enum": example_ids},
            "secondary_example_id": {"type": "string", "enum": example_ids},
            "rationale": {"type": "string"},
        },
    }


def validate_selection_response(
    custom_id: str, value: dict, candidate_ids: set[str]
) -> tuple[list[str], str]:
    if not isinstance(value, dict) or set(value) != {
        "primary_example_id",
        "secondary_example_id",
        "rationale",
    }:
        raise ValueError(f"{custom_id} has unexpected selection fields")
    ids = [value["primary_example_id"], value["secondary_example_id"]]
    if any(not isinstance(item, str) or item not in candidate_ids for item in ids):
        raise ValueError(f"{custom_id} must select candidate example IDs")
    if len(set(ids)) != RETRIEVAL_K:
        raise ValueError(f"{custom_id} must select two distinct candidate examples")
    rationale = value["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"{custom_id} has an empty retrieval rationale")
    return ids, rationale.strip()


def random_example_ids(
    candidate_ids: list[str], case_id: str, random_seed: int
) -> list[str]:
    material = f"{random_seed}:{case_id}:matched-early-random-two".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(seed).sample(candidate_ids, RETRIEVAL_K)


def selector_request_input(catalog: str, case: dict) -> str:
    return "\n\n".join(
        [
            "# Candidate solved training examples",
            catalog,
            "# Held-out project's observable prefix",
            observable_prefix(case["evaluation_input"]),
            "Select and rank exactly two candidate example IDs.",
        ]
    )


def prepare_selection(args: argparse.Namespace) -> tuple[dict, list[str], Path, Path]:
    if not args.model.strip():
        raise ValueError("selector model must not be empty")
    if args.max_output_tokens <= 0:
        raise ValueError("selector max output tokens must be positive")
    setting = experiment.load_setting(args.base_setting)
    assets = experiment.load_setting_assets(args.base_setting, setting)
    if "trajectory insight" in assets["demonstrations"].lower():
        raise ValueError("demonstration datastore contains trajectory insight")
    _, blocks = experiment.example_demonstrations(assets["demonstrations"])
    ids = candidate_example_ids(blocks)
    training_indices = setting["demonstration_rendering"]["training_indices"]
    overlap = set(training_indices) & set(setting["evaluation_indices"])
    if overlap:
        raise ValueError(f"training and evaluation projects overlap: {sorted(overlap)}")
    if len(ids) != len(training_indices):
        raise ValueError("candidate examples and training projects differ")
    catalogs = {
        length: candidate_catalog(assets["demonstrations"], length)[0]
        for length in PREFIX_LENGTHS
    }
    routing = experiment.routing_from_args(args)
    schema = selector_schema(ids)
    requests = [
        experiment.api_request(
            f"select__{case['case_id']}",
            experiment.response_body(
                args.model,
                SELECTOR_INSTRUCTIONS,
                selector_request_input(catalogs[int(case["natural_prefix_length"])], case),
                "matched_early_demo_retrieval",
                schema,
                args.max_output_tokens,
                args.reasoning_effort,
                routing,
            ),
        )
        for case in setting["cases"]
    ]
    request_path = args.selection_dir / "selection_requests.jsonl"
    run_path = args.selection_dir / "selection_run.json"
    experiment.write_or_validate_plan(
        request_path,
        run_path,
        requests,
        {
            "format_version": 1,
            "base_setting": experiment.recorded_path(args.base_setting),
            "candidate_example_ids": ids,
            "candidate_prefix_lengths": list(PREFIX_LENGTHS),
            "retrieval_k": RETRIEVAL_K,
            "random_seed": args.random_seed,
            "selector_model": args.model,
            "selector_reasoning_effort": args.reasoning_effort,
            "selector_max_output_tokens": args.max_output_tokens,
            "selector_api_provider": args.api_provider,
            "selector_provider_preferences": routing,
            "stateless": True,
        },
    )
    return setting, ids, request_path, args.selection_dir / "selection_output.jsonl"


def validate_selections(path: Path, base_setting: dict) -> dict:
    selections = experiment.read_json(path)
    if selections.get("retrieval_k") != RETRIEVAL_K:
        raise ValueError(f"selection artifact must use retrieval_k={RETRIEVAL_K}")
    ids = selections.get("candidate_example_ids")
    if (
        not isinstance(ids, list)
        or len(ids) < RETRIEVAL_K
        or any(not isinstance(item, str) or not item for item in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("selection artifact has invalid candidate IDs")
    records = selections.get("selections")
    expected_cases = {case["case_id"] for case in base_setting["cases"]}
    if not isinstance(records, dict) or set(records) != expected_cases:
        raise ValueError("selection artifact does not cover exactly the evaluation cases")
    random_seed = selections.get("random_seed")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("selection artifact has an invalid random seed")
    candidate_set = set(ids)
    case_map = {case["case_id"]: case for case in base_setting["cases"]}
    for case_id, record in records.items():
        if not isinstance(record, dict):
            raise ValueError(f"{case_id} has a malformed selection record")
        if set(record) != {
            "prefix_length",
            "retrieved_example_ids",
            "random_example_ids",
            "rationale",
        }:
            raise ValueError(f"{case_id} has unexpected selection fields")
        if record["prefix_length"] != case_map[case_id]["natural_prefix_length"]:
            raise ValueError(f"{case_id} has the wrong prepared prefix length")
        retrieved_ids = record["retrieved_example_ids"]
        if not isinstance(retrieved_ids, list) or len(retrieved_ids) != RETRIEVAL_K:
            raise ValueError(f"{case_id} must have two retrieved example IDs")
        validate_selection_response(
            case_id,
            {
                "primary_example_id": retrieved_ids[0],
                "secondary_example_id": retrieved_ids[1],
                "rationale": record["rationale"],
            },
            candidate_set,
        )
        if record["random_example_ids"] != random_example_ids(
            ids, case_id, random_seed
        ):
            raise ValueError(f"{case_id} has an invalid random control")
    return selections


def run_selection(args: argparse.Namespace) -> None:
    setting, ids, request_path, output_path = prepare_selection(args)
    candidate_set = set(ids)

    def validate_response(custom_id: str, text: str) -> None:
        validate_selection_response(custom_id, json.loads(text), candidate_set)

    experiment.run_requests(
        args, request_path, output_path, "RAG selections", validate_response
    )
    artifact_path = args.selection_dir / "selections.json"
    if artifact_path.exists():
        selections = validate_selections(artifact_path, setting)
        print(f"already prepared {len(selections['selections'])} selections")
        return

    expected_ids = {f"select__{case['case_id']}" for case in setting["cases"]}
    responses, usage = experiment.read_response_output(output_path, expected_ids)
    records = {}
    for case in setting["cases"]:
        case_id = case["case_id"]
        selected, rationale = validate_selection_response(
            f"select__{case_id}", responses[f"select__{case_id}"], candidate_set
        )
        records[case_id] = {
            "prefix_length": case["natural_prefix_length"],
            "retrieved_example_ids": selected,
            "random_example_ids": random_example_ids(
                ids, case_id, args.random_seed
            ),
            "rationale": rationale,
        }
    selection_run = experiment.read_json(args.selection_dir / "selection_run.json")
    selections = {
        "format_version": 1,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "candidate_example_ids": ids,
        "retrieval_k": RETRIEVAL_K,
        "random_seed": args.random_seed,
        "selector": {
            "model": selection_run["selector_model"],
            "reasoning_effort": selection_run["selector_reasoning_effort"],
            "api_provider": selection_run["selector_api_provider"],
            "provider_preferences": selection_run[
                "selector_provider_preferences"
            ],
        },
        "selection_usage": usage,
        "selections": records,
    }
    experiment.write_json(artifact_path, selections)
    print(f"prepared {len(records)} selections at {artifact_path}")


def load_rag_setting(path: Path) -> dict:
    setting = experiment.load_setting(path)
    if tuple(setting["methods"]) != METHODS:
        raise ValueError(f"RAG setting must use methods {list(METHODS)}")
    return setting


def prepare_rag_setting(
    base_setting_path: Path, selections_path: Path, output: Path
) -> None:
    base_setting = experiment.load_setting(base_setting_path)
    selections = validate_selections(selections_path, base_setting)
    if output.exists():
        setting = load_rag_setting(output)
        experiment.load_setting_assets(output, setting)
        if setting.get("base_setting") != experiment.recorded_path(base_setting_path):
            raise ValueError("existing RAG setting has a different base")
        print(f"already prepared: {output} ({setting['case_count']} cases)")
        return

    base_assets = experiment.load_setting_assets(base_setting_path, base_setting)
    if "trajectory insight" in base_assets["demonstrations"].lower():
        raise ValueError("demonstration datastore contains trajectory insight")
    _, blocks = experiment.example_demonstrations(base_assets["demonstrations"])
    if selections["candidate_example_ids"] != candidate_example_ids(blocks):
        raise ValueError("selection candidates differ from the prepared datastore")
    if set(base_setting["demonstration_rendering"]["training_indices"]) & set(
        base_setting["evaluation_indices"]
    ):
        raise ValueError("training and evaluation projects overlap")
    assets = {
        "prediction_instructions": base_assets["prediction_instructions"],
        "demonstrations": base_assets["demonstrations"],
        "judge_instructions": base_assets["judge_instructions"],
    }
    filenames = {
        "prediction_instructions": "prediction_instructions.md",
        "demonstrations": "demonstrations.md",
        "judge_instructions": "judge_instructions.md",
    }
    cases = []
    for base_case in base_setting["cases"]:
        case = copy.deepcopy(base_case)
        record = selections["selections"][case["case_id"]]
        case["rag_selection"] = {
            "retrieved_example_ids": record["retrieved_example_ids"],
            "random_example_ids": record["random_example_ids"],
        }
        cases.append(case)

    output.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = output.parent / "assets"
    asset_dir.mkdir()
    for name, text in assets.items():
        (asset_dir / filenames[name]).write_text(text, encoding="utf-8")
    setting = {
        "format_version": 1,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "base_setting": experiment.recorded_path(base_setting_path),
        "evaluation_indices": copy.deepcopy(base_setting["evaluation_indices"]),
        "methods": list(METHODS),
        "selection": {
            **copy.deepcopy(base_setting["selection"]),
            "experiment": "matched early same-k retrieval of two ordered demos",
        },
        "retrieval": {
            "retrieval_k": RETRIEVAL_K,
            "same_prefix_length_only": True,
            "candidate_example_ids": selections["candidate_example_ids"],
            "random_seed": selections["random_seed"],
            "selector": selections["selector"],
        },
        "case_count": len(cases),
        "prediction_request_count": len(cases) * len(METHODS),
        "judge_request_count": len(cases) * len(METHODS),
        "assets": {name: {"path": f"assets/{filenames[name]}"} for name in assets},
        "sources": copy.deepcopy(base_setting["sources"]),
        "cases": cases,
    }
    experiment.write_json(output, setting)
    load_rag_setting(output)
    experiment.load_setting_assets(output, setting)
    print(f"prepared {len(cases)} cases x {len(METHODS)} RAG methods at {output}")


def summarize(args: argparse.Namespace) -> None:
    setting = load_rag_setting(args.setting)
    base_setting = experiment.load_setting(args.base_setting)
    skills_setting = experiment.load_setting(args.skills_setting)
    if setting["base_setting"] != experiment.recorded_path(args.base_setting):
        raise ValueError("the RAG setting was not prepared from this demos setting")
    if base_setting.get("source_setting") != skills_setting.get("source_setting"):
        raise ValueError("demos and skills settings have different sources")
    if sorted(map(demos_experiment.case_signature, setting["cases"])) != sorted(
        map(demos_experiment.case_signature, skills_setting["cases"])
    ):
        raise ValueError("RAG and skills settings contain different cases")
    experiment.summarize_comparison(
        args.setting,
        setting,
        METHODS,
        (
            (args.skills_run_root, ("baseline", "skill")),
            (args.demos_run_root, ("demonstrations",)),
            (args.run_root, METHODS),
        ),
        CONTRASTS,
        args.repeats,
        args.output_dir,
        {"retrieval": setting["retrieval"]},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--base-setting", type=Path, default=BASE_SETTING)
    select_parser.add_argument(
        "--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR
    )
    select_parser.add_argument(
        "--random-seed", type=int, default=DEFAULT_RANDOM_SEED
    )
    experiment.add_execution_options(select_parser, 2000)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--base-setting", type=Path, default=BASE_SETTING)
    prepare_parser.add_argument("--selections", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_SETTING)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--setting", type=Path, default=DEFAULT_SETTING)
    summarize_parser.add_argument("--base-setting", type=Path, default=BASE_SETTING)
    summarize_parser.add_argument("--skills-setting", type=Path, default=SKILLS_SETTING)
    summarize_parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    summarize_parser.add_argument("--skills-run-root", type=Path, default=SKILLS_RUN_ROOT)
    summarize_parser.add_argument("--demos-run-root", type=Path, default=DEMOS_RUN_ROOT)
    summarize_parser.add_argument("--output-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    summarize_parser.add_argument("--repeats", type=int, default=3)

    args = parser.parse_args()
    if args.command == "select":
        run_selection(args)
    elif args.command == "prepare":
        prepare_rag_setting(args.base_setting, args.selections, args.output)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
