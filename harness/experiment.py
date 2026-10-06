#!/usr/bin/env python3
"""Run and score trajectory predictions through concurrent, stateless API calls."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

import build_prompt
import render_demonstrations


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def recorded_path(path: Path) -> str:
    """Path to record in setting files and run files: relative to the repository root when inside it."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_recorded(value: str) -> Path:
    """Inverse of recorded_path: repository-relative records resolve against the repository root."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path
ANNOTATIONS = REPO_ROOT / "annotations" / "neurips_2025"
# `prepare` builds the intermediate base that `prepare-full` turns into prefix-length.
DEFAULT_SETTING = HERE / "runs" / "prefix-length-base" / "setting.json"
DEFAULT_FULL_SETTING = HERE / "experiments" / "prefix-length" / "setting.json"
DEFAULT_ORDERED_SKILL = HERE / "skills" / "trajectory" / "SKILL.md"
DEFAULT_ORDERED_DEMONSTRATIONS = HERE / "prompts" / "demonstrations.md"
DEFAULT_SHUFFLED_DEMONSTRATIONS = HERE / "prompts" / "shuffled_demonstrations.md"
DEFAULT_SHUFFLE_METADATA = HERE / "prompts" / "shuffled_demonstrations.metadata.json"
DEFAULT_SHUFFLED_SKILL = HERE / "skills" / "trajectory-shuffled" / "SKILL.md"
PREFIX_LENGTH_INDICES = render_demonstrations.PREFIX_LENGTH_INDICES
METHODS = ("baseline", "skill", "demonstrations")
FULL_METHODS = (
    "baseline",
    "skill",
    "demonstrations",
    "shuffled_skills",
    "shuffled_demonstrations",
)
# Retrieval methods show two demonstrations chosen per case (`rag_selection`):
# method -> which pair.
RAG_METHODS = {
    "random_two_ordered": "random",
    "retrieved_two_ordered": "retrieved",
}
# Skill methods add a distilled skill to the instructions: method -> asset.
SKILL_METHODS = {
    "skill": "skill_body",
    "shuffled_skills": "shuffled_skill_body",
    "final_paper_skill": "final_paper_skill_body",
}
SUPPORTED_METHODS = (*FULL_METHODS, *RAG_METHODS, "final_paper_skill")
TARGET_CATEGORIES = ("method", "experiment", "ablation")
PREDICTION_SUFFIX = "\nPredict the next research decision. Return the required JSON only."
# The predictor commits to the kind of decision, the component it acts on, and the
# operation applied before writing the sentence, so the sentence is conditioned on
# those choices rather than labelled after the fact. These verbs match the judge's
# operation_match anchor.
DECISION_OPERATIONS = (
    "introduce",
    "replace",
    "extend",
    "compare",
    "remove",
    "calibrate",
    "validate",
    "restrict",
)
MIN_COMPLETED_PRIOR_DECISIONS = 3
DEFAULT_OPENROUTER_MAX_PROMPT_PRICE = 4.0
DEFAULT_OPENROUTER_MAX_COMPLETION_PRICE = 15.0
DEFAULT_CONCURRENCY = 16
DEFAULT_MAX_ATTEMPTS = 5
RETRYABLE_HTTP_STATUSES = {408, 409, 429, 500, 502, 503, 504}
API_ROOTS = {
    "openrouter": "https://openrouter.ai/api",
    "openai": "https://api.openai.com",
}


PREDICTION_FIELDS = ("category", "object", "operation", "decision")
PREDICTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    # Property order is emission order under strict json_schema decoding.
    "required": list(PREDICTION_FIELDS),
    "properties": {
        "category": {"type": "string", "enum": list(TARGET_CATEGORIES)},
        "object": {"type": "string"},
        "operation": {"type": "string", "enum": list(DECISION_OPERATIONS)},
        "decision": {"type": "string"},
    },
}


JUDGE_METRICS = ("component_match", "operation_match", "specification_match")
# The trajectory determines which component is acted on and what operation is applied.
# It does not determine concrete settings, so specification_match is reported as a
# diagnostic and kept out of the headline score.
HEADLINE_METRIC = "trajectory_alignment"
HEADLINE_COMPONENTS = ("component_match", "operation_match")
REPORTED_METRICS = (HEADLINE_METRIC, *JUDGE_METRICS)
JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [*JUDGE_METRICS, "justification"],
    "properties": {
        **{
            metric: {"type": "integer", "minimum": 0, "maximum": 2}
            for metric in JUDGE_METRICS
        },
        "justification": {"type": "string"},
    },
}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_scores(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"required completed score table is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"score table is empty: {path}")
    return rows


def find_annotation(index: int) -> Path:
    matches = sorted(ANNOTATIONS.glob(f"{index}-*/annotation.json"))
    if len(matches) != 1:
        raise ValueError(f"idx={index} has {len(matches)} annotation.json files; expected exactly one")
    return matches[0]


def example_demonstrations(document: str) -> tuple[str, dict[tuple[str, int], str]]:
    """Header and (example ID, prefix length) blocks of a matched early-demonstration document."""
    matches = list(
        re.finditer(
            r"(?m)^## Training trajectory ([0-9]{3})\n### Prefix length ([123])\s*$",
            document,
        )
    )
    if not matches:
        raise ValueError("matched demonstration document contains no examples")
    blocks = {}
    for position, match in enumerate(matches):
        key = (match.group(1), int(match.group(2)))
        end = matches[position + 1].start() if position + 1 < len(matches) else len(document)
        if key in blocks:
            raise ValueError(f"demonstration document repeats example {key}")
        blocks[key] = document[match.start() : end].strip()
    return document[: matches[0].start()].strip(), blocks


def selected_demonstrations(document: str, ids: list, prefix_length: int) -> str:
    """The header plus the two selected early examples of one prefix length."""
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("a selection must contain two distinct demonstrations")
    header, blocks = example_demonstrations(document)
    keys = [(example_id, prefix_length) for example_id in ids]
    missing = [key for key in keys if key not in blocks]
    if missing:
        raise ValueError(f"selected demonstrations are missing: {missing}")
    return "\n\n".join([header, *(blocks[key] for key in keys)])


def compose_prepared_request(
    assets: dict[str, str], evaluation_input: str, method: str, case: dict | None = None
) -> dict:
    """Instructions and input of one prediction request. Retrieval methods read the
    two demonstrations selected for `case`."""
    instructions = assets["prediction_instructions"].strip()
    if method in SKILL_METHODS:
        instructions += (
            "\n\nUse the following distilled research-trajectory skill for this prediction:\n\n"
            + assets[SKILL_METHODS[method]].strip()
        )
    input_parts = []
    if method in {"demonstrations", "shuffled_demonstrations"}:
        input_parts.extend([assets[method].strip(), "# End reference trajectories"])
    elif method in RAG_METHODS:
        demonstrations = selected_demonstrations(
            assets["demonstrations"],
            case["rag_selection"][f"{RAG_METHODS[method]}_example_ids"],
            case["natural_prefix_length"],
        )
        input_parts.extend([demonstrations, "# End reference trajectories"])
    elif method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported method: {method}")
    input_parts.append(evaluation_input)
    return {"instructions": instructions, "input": "\n\n".join(input_parts)}


def load_setting(path: Path) -> dict:
    setting = json.loads(path.read_text(encoding="utf-8"))
    methods = tuple(setting.get("methods", []))
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("setting methods must be a nonempty list without duplicates")
    unsupported = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unsupported:
        raise ValueError(f"setting has unsupported methods: {unsupported}")
    cases = setting.get("cases")
    case_count = setting.get("case_count")
    if not isinstance(cases, list) or not isinstance(case_count, int) or case_count <= 0:
        raise ValueError("setting cases must be a nonempty list with a positive case_count")
    if len(cases) != case_count:
        raise ValueError(
            f"setting declares {case_count} cases but contains {len(cases)} case records"
        )
    expected_requests = case_count * len(methods)
    if setting.get("prediction_request_count") != expected_requests:
        raise ValueError(
            f"setting has {setting.get('prediction_request_count')} prediction requests, "
            f"expected {expected_requests}"
        )
    if setting.get("judge_request_count") != expected_requests:
        raise ValueError(
            f"setting has {setting.get('judge_request_count')} judge requests, "
            f"expected {expected_requests}"
        )
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("setting contains duplicate case IDs")
    return setting


def load_setting_assets(setting_path: Path, setting: dict) -> dict[str, str]:
    assets = {}
    for name, metadata in setting["assets"].items():
        path = setting_path.parent / metadata["path"]
        assets[name] = path.read_text(encoding="utf-8")
    return assets


def prepare_suite(output: Path, replace: bool = False) -> None:
    if output.exists() and not replace:
        setting = load_setting(output)
        load_setting_assets(output, setting)
        if tuple(setting["evaluation_indices"]) != PREFIX_LENGTH_INDICES:
            raise ValueError(
                f"{output} was prepared with evaluation indices "
                f"{setting['evaluation_indices']}, not {list(PREFIX_LENGTH_INDICES)}"
            )
        print(f"already prepared: {output} ({setting['case_count']} cases)")
        return
    if replace and not output.exists():
        raise FileNotFoundError(f"cannot replace missing prepared setting: {output}")

    asset_text = {
        "prediction_instructions": build_prompt.TASK_PROMPT.read_text(encoding="utf-8"),
        "demonstrations": build_prompt.DEMONSTRATIONS.read_text(encoding="utf-8"),
        "skill_body": build_prompt.skill_body() + "\n",
        "judge_instructions": (HERE / "prompts" / "judge_rubrics_long.md").read_text(encoding="utf-8"),
    }
    asset_filenames = {
        "prediction_instructions": "prediction_instructions.md",
        "demonstrations": "demonstrations.md",
        "skill_body": "skill_body.md",
        "judge_instructions": "judge_instructions.md",
    }
    sources = []
    cases = []
    for index in PREFIX_LENGTH_INDICES:
        annotation_path = find_annotation(index)
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        decisions = build_prompt.validate_annotation(annotation_path, annotation)
        source_rel = annotation_path.relative_to(REPO_ROOT).as_posix()
        source_case_count = 0
        for target_step in range(1, len(decisions)):
            completed = build_prompt.completed_prior_decisions(decisions, target_step)
            if len(completed) < MIN_COMPLETED_PRIOR_DECISIONS:
                continue
            observed = build_prompt.observed_prefix(decisions, target_step)
            evaluation_input = build_prompt.render_prefix(decisions, target_step)
            target = decisions[target_step]
            case_id = f"idx-{index}-t{target_step:03d}"
            cases.append(
                {
                    "case_id": case_id,
                    "evaluation_idx": index,
                    "target_time_step_id": target_step,
                    "observed_decision_count": len(observed),
                    "completed_prior_decision_count": len(completed),
                    "source_annotation": source_rel,
                    "evaluation_input": evaluation_input,
                    "target": {
                        "decision_id": target["decision_id"],
                        "category": target["category"],
                        "decision": render_demonstrations.decision_action_text(
                            target["decision"]
                        ),
                    },
                }
            )
            source_case_count += 1
        sources.append(
            {
                "evaluation_idx": index,
                "annotation": source_rel,
                "decision_count": len(decisions),
                "case_count": source_case_count,
            }
        )

    asset_dir = output.parent / "assets"
    asset_dir.mkdir(parents=True, exist_ok=replace)
    for name, text in asset_text.items():
        (asset_dir / asset_filenames[name]).write_text(text, encoding="utf-8")

    setting = {
        "format_version": 1,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_indices": list(PREFIX_LENGTH_INDICES),
        "methods": list(METHODS),
        "selection": {
            "indices_source": render_demonstrations.SPLITS.relative_to(
                REPO_ROOT
            ).as_posix(),
            "minimum_completed_prior_decisions": MIN_COMPLETED_PRIOR_DECISIONS,
            "target_rule": "at least three prior decisions have last_date strictly earlier than target first_date",
            "prefix_rule": "all prior decisions in timezone-aware first_date order",
            "outcome_rule": "overlapping decisions are active at cutoff; completed outcomes and observed supersession links are shown",
            "decision_text_rule": "scientific action only; retrospective outcome clauses removed",
        },
        "case_count": len(cases),
        "prediction_request_count": len(cases) * len(METHODS),
        "judge_request_count": len(cases) * len(METHODS),
        "assets": {
            name: {"path": f"assets/{asset_filenames[name]}"} for name in asset_text
        },
        "sources": sources,
        "cases": cases,
    }
    write_json(output, setting)
    action = "refroze" if replace else "froze"
    print(f"{action} {len(cases)} cases at {output}")


def prepare_full_suite(
    output: Path,
    base_setting_path: Path,
    ordered_skill_path: Path,
    ordered_demonstrations_path: Path,
    shuffled_demonstrations_path: Path,
    shuffle_metadata_path: Path,
    shuffled_skill_path: Path,
    replace: bool = False,
    methods: tuple[str, ...] = FULL_METHODS,
) -> None:
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be a nonempty sequence without duplicates")
    unsupported = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unsupported:
        raise ValueError(f"unsupported methods: {unsupported}")

    if output.exists() and not replace:
        setting = load_setting(output)
        load_setting_assets(output, setting)
        if tuple(setting["methods"]) != methods:
            raise ValueError(
                f"{output} uses methods {setting['methods']!r}, "
                f"not {list(methods)!r}"
            )
        print(f"already prepared: {output} ({setting['case_count']} cases)")
        return
    if replace and not output.exists():
        raise FileNotFoundError(f"cannot replace missing prepared setting: {output}")

    base_setting = load_setting(base_setting_path)
    if tuple(base_setting["evaluation_indices"]) != PREFIX_LENGTH_INDICES:
        raise ValueError(
            f"{base_setting_path} was prepared with evaluation indices "
            f"{base_setting['evaluation_indices']}, not {list(PREFIX_LENGTH_INDICES)}; "
            "prepare a new base setting from the prefix_length list in annotate/harness_splits.json"
        )
    if tuple(base_setting["methods"]) != METHODS:
        raise ValueError(
            f"base setting methods differ from the original methods: {METHODS}"
        )
    load_setting_assets(base_setting_path, base_setting)

    evaluation_decisions = {}
    for source in base_setting["sources"]:
        annotation_path = REPO_ROOT / source["annotation"]
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        evaluation_decisions[source["evaluation_idx"]] = (
            annotation_path,
            build_prompt.validate_annotation(annotation_path, annotation),
        )

    shuffle_metadata = json.loads(shuffle_metadata_path.read_text(encoding="utf-8"))
    if shuffle_metadata.get("format_version") != 2:
        raise ValueError(f"{shuffle_metadata_path} must use shuffle format version 2")
    seed = shuffle_metadata.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{shuffle_metadata_path} has no integer seed")
    trajectories = shuffle_metadata.get("trajectories")
    expected_indices = {str(index) for index in render_demonstrations.TRAIN_INDICES}
    if not isinstance(trajectories, dict) or set(trajectories) != expected_indices:
        raise ValueError(
            f"{shuffle_metadata_path} must contain permutations for {sorted(expected_indices)}"
        )
    for index, metadata in trajectories.items():
        source_annotation = REPO_ROOT / metadata.get("source_annotation", "")
        if not source_annotation.is_file():
            raise ValueError(f"idx={index} shuffle source does not exist: {source_annotation}")
        permutation = metadata.get("shuffled_position_to_original_time_step_id")
        decision_id_map = metadata.get("original_decision_id_to_shuffled_decision_id")
        randomized_outcomes = metadata.get("randomized_outcomes")
        randomized_links = metadata.get("randomized_superseded_by")
        if (
            not isinstance(permutation, list)
            or sorted(permutation) != list(range(len(permutation)))
        ):
            raise ValueError(f"idx={index} has an invalid stored permutation")
        expected_new_ids = {
            f"D{position + 1:03d}" for position in range(len(permutation))
        }
        if (
            not isinstance(decision_id_map, dict)
            or set(decision_id_map.values()) != expected_new_ids
        ):
            raise ValueError(f"idx={index} has an invalid shuffled decision-ID map")
        if (
            not isinstance(randomized_outcomes, dict)
            or set(randomized_outcomes) != expected_new_ids
            or any(
                outcome not in {"retained", "abandoned", "superseded"}
                for outcome in randomized_outcomes.values()
            )
        ):
            raise ValueError(f"idx={index} has invalid randomized outcomes")
        superseded_ids = {
            decision_id
            for decision_id, outcome in randomized_outcomes.items()
            if outcome == "superseded"
        }
        if (
            not isinstance(randomized_links, dict)
            or set(randomized_links) != superseded_ids
            or any(
                target not in expected_new_ids
                or int(target[1:]) <= int(source[1:])
                for source, target in randomized_links.items()
            )
        ):
            raise ValueError(f"idx={index} has invalid randomized supersession links")

    asset_text = {
        "prediction_instructions": build_prompt.TASK_PROMPT.read_text(encoding="utf-8"),
        "demonstrations": ordered_demonstrations_path.read_text(encoding="utf-8"),
        "skill_body": build_prompt.skill_body(ordered_skill_path) + "\n",
        "shuffled_demonstrations": shuffled_demonstrations_path.read_text(
            encoding="utf-8"
        ),
        "shuffled_skill_body": build_prompt.skill_body(shuffled_skill_path) + "\n",
        "judge_instructions": (HERE / "prompts" / "judge_rubrics_long.md").read_text(
            encoding="utf-8"
        ),
    }
    asset_filenames = {
        "prediction_instructions": "prediction_instructions.md",
        "demonstrations": "demonstrations.md",
        "skill_body": "skill_body.md",
        "shuffled_demonstrations": "shuffled_demonstrations.md",
        "shuffled_skill_body": "shuffled_skill_body.md",
        "judge_instructions": "judge_instructions.md",
    }
    cases = []
    for base_case in base_setting["cases"]:
        annotation_path, decisions = evaluation_decisions[base_case["evaluation_idx"]]
        target_step = base_case["target_time_step_id"]
        target = decisions[target_step]
        if target["decision_id"] != base_case["target"]["decision_id"]:
            raise ValueError(f"target changed for {base_case['case_id']}")
        observed = build_prompt.observed_prefix(decisions, target_step)
        completed = build_prompt.completed_prior_decisions(decisions, target_step)
        evaluation_input = build_prompt.render_prefix(decisions, target_step)
        case = {
            "case_id": base_case["case_id"],
            "evaluation_idx": base_case["evaluation_idx"],
            "target_time_step_id": target_step,
            "observed_decision_count": len(observed),
            "completed_prior_decision_count": len(completed),
            "source_annotation": annotation_path.relative_to(REPO_ROOT).as_posix(),
            "evaluation_input": evaluation_input,
            "target": {
                "decision_id": target["decision_id"],
                "category": target["category"],
                "decision": render_demonstrations.decision_action_text(
                    target["decision"]
                ),
            },
        }
        cases.append(case)

    category_counts = {
        category: sum(
            case["target"]["category"] == category for case in cases
        )
        for category in TARGET_CATEGORIES
        if any(
            case["target"]["category"] == category
            for case in cases
        )
    }
    selected_case_counts = {
        source["evaluation_idx"]: sum(
            case["evaluation_idx"] == source["evaluation_idx"] for case in cases
        )
        for source in base_setting["sources"]
    }
    sources = [
        {**source, "case_count": selected_case_counts[source["evaluation_idx"]]}
        for source in base_setting["sources"]
        if selected_case_counts[source["evaluation_idx"]] > 0
    ]
    evaluation_indices = [source["evaluation_idx"] for source in sources]
    excluded_evaluation_indices = sorted(
        set(base_setting["evaluation_indices"]) - set(evaluation_indices)
    )

    asset_dir = output.parent / "assets"
    asset_dir.mkdir(parents=True, exist_ok=replace)
    for name, text in asset_text.items():
        (asset_dir / asset_filenames[name]).write_text(text, encoding="utf-8")

    setting = {
        "format_version": 1,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "base_setting": recorded_path(base_setting_path),
        "evaluation_indices": evaluation_indices,
        "methods": list(methods),
        "selection": {
            "target_cases": "all eligible targets from the base setting",
            "base_target_selection": base_setting["selection"],
            "target_case_count": len(cases),
            "target_category_counts": category_counts,
            "excluded_evaluation_indices": excluded_evaluation_indices,
            "prefix_rule": "all prior decisions in timezone-aware first_date order",
            "outcome_rule": "overlapping decisions are active at cutoff; completed outcomes and observed supersession links are shown",
            "decision_text_rule": "scientific action only; retrospective outcome clauses removed",
        },
        "case_count": len(cases),
        "prediction_request_count": len(cases) * len(methods),
        "judge_request_count": len(cases) * len(methods),
        "shuffling": shuffle_metadata,
        "assets": {
            name: {"path": f"assets/{asset_filenames[name]}"} for name in asset_text
        },
        "sources": sources,
        "cases": cases,
    }
    write_json(output, setting)
    action = "refroze" if replace else "froze"
    print(
        f"{action} {len(cases)} cases and "
        f"{len(cases) * len(methods)} requests at {output}"
    )


def response_body(
    model: str,
    instructions: str,
    input_text: str,
    schema_name: str,
    schema: dict,
    max_output_tokens: int,
    reasoning_effort: str | None,
    provider_preferences: dict | None = None,
) -> dict:
    body = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "store": False,
        "max_output_tokens": max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    if provider_preferences:
        body["provider"] = provider_preferences
    return body


def provider_preferences(
    api_provider: str,
    max_prompt_price: float,
    max_completion_price: float,
    allow_azure: bool,
) -> dict | None:
    if api_provider == "openai":
        return None
    if api_provider != "openrouter":
        raise ValueError(f"unsupported API provider: {api_provider}")
    if max_prompt_price <= 0 or max_completion_price <= 0:
        raise ValueError("OpenRouter maximum prices must be positive")
    preferences = {
        "sort": "price",
        "max_price": {
            "prompt": max_prompt_price,
            "completion": max_completion_price,
        },
        "require_parameters": True,
    }
    if not allow_azure:
        preferences["ignore"] = ["azure"]
    return preferences


def prediction_custom_id(case_id: str, method: str) -> str:
    return f"pred__{case_id}__{method}"


def judge_custom_id(case_id: str, method: str) -> str:
    return f"judge__{case_id}__{method}"


def api_request(custom_id: str, body: dict) -> dict:
    """One line of a request file."""
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}


def prediction_request(
    custom_id: str,
    instructions: str,
    input_text: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    routing: dict | None,
) -> dict:
    return api_request(
        custom_id,
        response_body(
            model, instructions, input_text, "next_research_decision_prediction",
            PREDICTION_SCHEMA, max_output_tokens, reasoning_effort, routing,
        ),
    )


def judge_request(
    custom_id: str,
    instructions: str,
    case: dict,
    prediction: dict,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    routing: dict | None,
) -> dict:
    return api_request(
        custom_id,
        response_body(
            model, instructions, render_judge_input(case, prediction),
            "individual_prediction_judgment", JUDGE_SCHEMA, max_output_tokens,
            reasoning_effort, routing,
        ),
    )


def routing_from_args(args: argparse.Namespace) -> dict | None:
    return provider_preferences(
        args.api_provider,
        args.openrouter_max_prompt_price,
        args.openrouter_max_completion_price,
        args.allow_azure,
    )


def write_or_validate_plan(
    request_path: Path, run_path: Path, requests: list[dict], metadata: dict
) -> None:
    """Write a request file and its run record, or check that existing ones match."""
    request_text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests)
    metadata = {**metadata, "request_count": len(requests)}
    present = [path for path in (request_path, run_path) if path.exists()]
    if not present:
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(request_text, encoding="utf-8")
        write_json(run_path, metadata)
        print(f"wrote {len(requests)} stateless requests to {request_path}")
        return
    if len(present) != 2:
        raise ValueError(f"incomplete request plan; expected both {request_path} and {run_path}")
    existing = read_json(run_path)
    mismatches = {
        field: {"existing": existing.get(field), "requested": value}
        for field, value in metadata.items()
        if existing.get(field) != value
    }
    if mismatches:
        raise ValueError(f"existing request plan does not match: {mismatches}")
    load_request_file(request_path)


def run_requests(args: argparse.Namespace, request_path: Path, output_path: Path, label: str, validator) -> None:
    """Run a request file with the command's execution options; errors go next to the output."""
    run_concurrent_requests(
        request_path,
        output_path,
        output_path.with_name(output_path.name.replace("_output", "_errors")),
        label,
        args.api_provider,
        args.concurrency,
        args.max_attempts,
        args.confirm_submit,
        validator,
    )


def load_request_file(path: Path) -> list[dict]:
    requests = []
    custom_ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError(f"missing custom_id at {path}:{line_number}")
        if custom_id in custom_ids:
            raise ValueError(f"duplicate custom_id at {path}:{line_number}: {custom_id}")
        if item.get("method") != "POST" or item.get("url") != "/v1/responses":
            raise ValueError(f"unexpected request method or URL at {path}:{line_number}")
        body = item.get("body")
        if not isinstance(body, dict) or not isinstance(body.get("model"), str):
            raise ValueError(f"missing request body or model at {path}:{line_number}")
        if body.get("store") is not False or "previous_response_id" in body:
            raise ValueError(f"request is not stateless at {path}:{line_number}")
        custom_ids.add(custom_id)
        requests.append(item)
    if not requests:
        raise ValueError(f"{path} contains no requests")
    return requests


def api_key(api_provider: str) -> str:
    variable = "OPENROUTER_API_KEY" if api_provider == "openrouter" else "OPENAI_API_KEY"
    key = os.environ.get(variable)
    if not key:
        raise ValueError(f"{variable} is not set")
    return key


def execute_request(
    item: dict,
    api_provider: str,
    key: str,
    max_attempts: int,
    validator,
) -> dict:
    custom_id = item["custom_id"]
    url = API_ROOTS[api_provider] + item["url"]
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if api_provider == "openrouter":
        headers["X-OpenRouter-Title"] = "trajectory-ideation-harness"
        if os.environ.get("OPENROUTER_REFERER"):
            headers["HTTP-Referer"] = os.environ["OPENROUTER_REFERER"]
    data = json.dumps(item["body"], ensure_ascii=False).encode("utf-8")

    for attempt in range(1, max_attempts + 1):
        try:
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=600) as response:
                body = json.loads(response.read())
                result = {
                    "custom_id": custom_id,
                    "response": {
                        "status_code": response.status,
                        "request_id": response.headers.get("request-id"),
                        "body": body,
                    },
                    "error": None,
                }
            text, _ = extract_response_text(result)
            validator(custom_id, text)
            return result
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in RETRYABLE_HTTP_STATUSES or attempt == max_attempts:
                raise RuntimeError(
                    f"{custom_id} failed with HTTP {exc.code}: {detail}"
                ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"{custom_id} failed after {max_attempts} attempts: {exc}"
                ) from exc
        time.sleep(min(2 ** (attempt - 1), 30))
    raise AssertionError("unreachable")


def completed_response_ids(
    output_path: Path,
    expected_ids: set[str],
    validator,
) -> set[str]:
    if not output_path.exists():
        return set()
    completed = set()
    for line_number, line in enumerate(
        output_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item.get("custom_id")
        if custom_id not in expected_ids:
            raise ValueError(
                f"unexpected custom_id at {output_path}:{line_number}: {custom_id}"
            )
        if custom_id in completed:
            raise ValueError(
                f"duplicate custom_id at {output_path}:{line_number}: {custom_id}"
            )
        text, _ = extract_response_text(item)
        validator(custom_id, text)
        completed.add(custom_id)
    return completed


def show_progress(
    label: str,
    completed: int,
    failed: int,
    total: int,
    started_at: float,
    final: bool = False,
) -> None:
    finished = completed + failed
    ratio = finished / total
    width = 30
    filled = round(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = int(time.monotonic() - started_at)
    line = (
        f"{label} [{bar}] {finished}/{total} ({ratio:6.1%}, "
        f"{failed} failed, {elapsed}s)"
    )
    if sys.stderr.isatty():
        print(f"\r\033[K{line}", end="\n" if final else "", file=sys.stderr, flush=True)
    elif final or (finished and finished % 10 == 0 and finished != total):
        print(line, file=sys.stderr, flush=True)


def run_concurrent_requests(
    request_path: Path,
    output_path: Path,
    error_path: Path,
    label: str,
    api_provider: str,
    concurrency: int,
    max_attempts: int,
    confirm_submit: bool,
    validator,
) -> None:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if max_attempts <= 0:
        raise ValueError("max attempts must be positive")
    requests = load_request_file(request_path)
    expected_ids = {item["custom_id"] for item in requests}
    completed_ids = completed_response_ids(output_path, expected_ids, validator)
    remaining = [item for item in requests if item["custom_id"] not in completed_ids]
    if not remaining:
        print(f"{label}: all {len(requests)} responses already completed")
        return
    if not confirm_submit:
        raise ValueError(
            f"{len(remaining)} paid API requests remain; pass --confirm-submit to proceed"
        )

    key = api_key(api_provider)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    completed = len(completed_ids)
    failures = []
    show_progress(label, completed, 0, len(requests), started_at)
    with output_path.open("a", encoding="utf-8") as output_handle, error_path.open(
        "a", encoding="utf-8"
    ) as error_handle, ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                execute_request,
                item,
                api_provider,
                key,
                max_attempts,
                validator,
            ): item["custom_id"]
            for item in remaining
        }
        for future in as_completed(futures):
            custom_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append(custom_id)
                error_handle.write(
                    json.dumps(
                        {
                            "custom_id": custom_id,
                            "error": str(exc),
                            "recorded_at": datetime.now(timezone.utc).isoformat(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                error_handle.flush()
            else:
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_handle.flush()
                completed += 1
            show_progress(label, completed, len(failures), len(requests), started_at)
    show_progress(
        label, completed, len(failures), len(requests), started_at, final=True
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} requests failed after retries; rerun the same command "
            f"to retry them. Details: {error_path}"
        )


def prepare_predictions(
    setting_path: Path,
    run_dir: Path,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    api_provider: str = "openrouter",
    max_prompt_price: float = DEFAULT_OPENROUTER_MAX_PROMPT_PRICE,
    max_completion_price: float = DEFAULT_OPENROUTER_MAX_COMPLETION_PRICE,
    allow_azure: bool = False,
) -> None:
    if not model.strip():
        raise ValueError("prediction model must not be empty")
    if max_output_tokens <= 0:
        raise ValueError("prediction max_output_tokens must be positive")
    routing = provider_preferences(
        api_provider, max_prompt_price, max_completion_price, allow_azure
    )
    setting = load_setting(setting_path)
    assets = load_setting_assets(setting_path, setting)
    methods = tuple(setting["methods"])
    expected_predictions = setting["prediction_request_count"]
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "prediction_requests.jsonl"
    run_path = run_dir / "run.json"
    if request_path.exists() or run_path.exists():
        raise FileExistsError(f"prediction run already exists in {run_dir}")

    lines = []
    for case in setting["cases"]:
        for method in methods:
            request = compose_prepared_request(assets, case["evaluation_input"], method, case)
            lines.append(
                prediction_request(
                    prediction_custom_id(case["case_id"], method),
                    request["instructions"],
                    request["input"],
                    model,
                    max_output_tokens,
                    reasoning_effort,
                    routing,
                )
            )
    if len(lines) != expected_predictions:
        raise ValueError(
            f"prepared {len(lines)} predictions, expected {expected_predictions}"
        )

    request_path.write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
        encoding="utf-8",
    )
    run = {
        "format_version": 1,
        "setting": recorded_path(setting_path),
        "prediction_model": model,
        "prediction_reasoning_effort": reasoning_effort,
        "prediction_max_output_tokens": max_output_tokens,
        "prediction_request_count": len(lines),
        "prediction_requests": request_path.name,
        "prediction_api_provider": api_provider,
        "prediction_provider_preferences": routing,
        "execution_mode": "concurrent",
        "stateless": True,
    }
    write_json(run_path, run)
    print(f"wrote {len(lines)} stateless prediction requests to {request_path}")


def require_matching_prediction_run(
    setting_path: Path,
    run_dir: Path,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    api_provider: str = "openrouter",
    max_prompt_price: float = DEFAULT_OPENROUTER_MAX_PROMPT_PRICE,
    max_completion_price: float = DEFAULT_OPENROUTER_MAX_COMPLETION_PRICE,
    allow_azure: bool = False,
) -> None:
    routing = provider_preferences(
        api_provider, max_prompt_price, max_completion_price, allow_azure
    )
    setting = load_setting(setting_path)
    expected_predictions = setting["prediction_request_count"]
    request_path = run_dir / "prediction_requests.jsonl"
    run_path = run_dir / "run.json"
    present = [path for path in (request_path, run_path) if path.exists()]
    if not present:
        prepare_predictions(
            setting_path,
            run_dir,
            model,
            max_output_tokens,
            reasoning_effort,
            api_provider,
            max_prompt_price,
            max_completion_price,
            allow_azure,
        )
        return
    if len(present) != 2:
        raise ValueError(
            f"incomplete prediction preparation in {run_dir}; expected both "
            "run.json and prediction_requests.jsonl"
        )

    run = json.loads(run_path.read_text(encoding="utf-8"))
    expected = {
        "prediction_model": model,
        "prediction_reasoning_effort": reasoning_effort,
        "prediction_max_output_tokens": max_output_tokens,
        "prediction_request_count": expected_predictions,
        "prediction_api_provider": api_provider,
        "prediction_provider_preferences": routing,
        "execution_mode": "concurrent",
    }
    mismatches = {
        field: {"existing": run.get(field), "requested": value}
        for field, value in expected.items()
        if run.get(field) != value
    }
    if mismatches:
        raise ValueError(f"existing prediction run does not match this command: {mismatches}")
    requests = load_request_file(request_path)
    if len(requests) != expected_predictions or {item["body"]["model"] for item in requests} != {model}:
        raise ValueError(f"unexpected prediction request contents: {request_path}")


def load_prediction_results(
    setting: dict, prediction_output: Path
) -> tuple[dict[str, dict], dict[str, int]]:
    methods = tuple(setting["methods"])
    expected = {
        prediction_custom_id(case["case_id"], method)
        for case in setting["cases"]
        for method in methods
    }
    return read_predictions(prediction_output, expected)


def read_predictions(path: Path, expected_ids: set[str]) -> tuple[dict[str, dict], dict[str, int]]:
    predictions, usage = read_response_output(path, expected_ids)
    for custom_id, prediction in predictions.items():
        validate_prediction(custom_id, prediction)
    return predictions, usage


def extract_response_text(response_item: dict) -> tuple[str, dict]:
    response = response_item.get("response")
    if not isinstance(response, dict) or response.get("status_code") != 200:
        raise ValueError(f"API request failed: {response_item.get('custom_id')}: {response_item.get('error') or response}")
    body = response.get("body")
    status = body.get("status")
    if status is not None and status != "completed":
        raise ValueError(
            f"incomplete API response for {response_item.get('custom_id')}: "
            f"status={status!r}, details={body.get('incomplete_details')!r}"
        )
    if not isinstance(body, dict):
        raise ValueError(f"missing response body: {response_item.get('custom_id')}")
    texts = []
    for output in body.get("output", []):
        if output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content["text"])
    if not texts:
        raise ValueError(f"response has no output_text: {response_item.get('custom_id')}")
    return "".join(texts), body.get("usage", {})


def read_response_output(path: Path, expected_ids: set[str]) -> tuple[dict[str, dict], dict[str, int]]:
    parsed = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError(f"missing custom_id at {path}:{line_number}")
        if custom_id in parsed:
            raise ValueError(f"duplicate custom_id at {path}:{line_number}: {custom_id}")
        text, item_usage = extract_response_text(item)
        parsed[custom_id] = json.loads(text)
        for key in usage:
            usage[key] += int(item_usage.get(key, 0))
    actual_ids = set(parsed)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"response output ID mismatch; missing={missing[:5]}, extra={extra[:5]}")
    return parsed, usage


def validate_prediction(custom_id: str, prediction: dict) -> None:
    if prediction.get("category") not in set(TARGET_CATEGORIES):
        raise ValueError(f"{custom_id} has an invalid prediction category")
    if prediction.get("operation") not in set(DECISION_OPERATIONS):
        raise ValueError(f"{custom_id} has an invalid prediction operation")
    for field in ("object", "decision"):
        value = prediction.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{custom_id} has an empty prediction {field}")


def validate_prediction_response(custom_id: str, text: str) -> None:
    validate_prediction(custom_id, json.loads(text))


def render_judge_input(case: dict, prediction: dict) -> str:
    evaluation_input = case["evaluation_input"]
    if not evaluation_input.endswith(PREDICTION_SUFFIX):
        raise ValueError(f"{case['case_id']} has an unexpected evaluation input ending")
    observable_prefix = evaluation_input[: -len(PREDICTION_SUFFIX)]
    return "\n\n".join(
        [
            observable_prefix,
            "# Actual next annotated decision",
            f"Category: {case['target']['category']}",
            f"Decision: {case['target']['decision']}",
            "# Prediction to evaluate",
            json.dumps(prediction, indent=2, ensure_ascii=False),
            "Evaluate this prediction using the supplied rubric. Return the required JSON only.",
        ]
    )


def judge_rubric_source(judge_instructions_path: Path | None) -> str:
    if judge_instructions_path is None:
        return "setting"
    return recorded_path(judge_instructions_path)


def judge_rubric(assets: dict[str, str], judge_instructions_path: Path | None) -> str:
    """Return the judge rubric, optionally overridden by a live prompt file, so existing
    predictions can be re-scored under a revised rubric without re-preparing the setting."""
    if judge_instructions_path is None:
        return assets["judge_instructions"]
    text = judge_instructions_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"judge instructions are empty: {judge_instructions_path}")
    return text


def prepare_judging(
    setting_path: Path,
    run_dir: Path,
    prediction_output: Path,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    api_provider: str = "openrouter",
    max_prompt_price: float = DEFAULT_OPENROUTER_MAX_PROMPT_PRICE,
    max_completion_price: float = DEFAULT_OPENROUTER_MAX_COMPLETION_PRICE,
    allow_azure: bool = False,
    judge_instructions_path: Path | None = None,
) -> None:
    if not model.strip():
        raise ValueError("judge model must not be empty")
    if max_output_tokens <= 0:
        raise ValueError("judge max_output_tokens must be positive")
    routing = provider_preferences(
        api_provider, max_prompt_price, max_completion_price, allow_azure
    )
    setting = load_setting(setting_path)
    assets = load_setting_assets(setting_path, setting)
    methods = tuple(setting["methods"])
    expected_judgments = setting["judge_request_count"]
    predictions, prediction_usage = load_prediction_results(setting, prediction_output)
    judge_instructions = judge_rubric(assets, judge_instructions_path)

    request_path = run_dir / "judge_requests.jsonl"
    judge_run_path = run_dir / "judge_run.json"
    if request_path.exists() or judge_run_path.exists():
        raise FileExistsError(f"judge run already exists in {run_dir}")

    lines = []
    for case in setting["cases"]:
        for method in methods:
            lines.append(
                judge_request(
                    judge_custom_id(case["case_id"], method),
                    judge_instructions.strip(),
                    case,
                    predictions[prediction_custom_id(case["case_id"], method)],
                    model,
                    max_output_tokens,
                    reasoning_effort,
                    routing,
                )
            )
    if len(lines) != expected_judgments:
        raise ValueError(
            f"prepared {len(lines)} judges, expected {expected_judgments}"
        )

    request_path.write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
        encoding="utf-8",
    )
    write_json(
        judge_run_path,
        {
            "prediction_output": recorded_path(prediction_output),
            "prediction_usage": prediction_usage,
            "judge_model": model,
            "judge_instructions_source": judge_rubric_source(judge_instructions_path),
            "judge_metrics": list(JUDGE_METRICS),
            "judge_reasoning_effort": reasoning_effort,
            "judge_max_output_tokens": max_output_tokens,
            "judge_request_count": len(lines),
            "judge_api_provider": api_provider,
            "judge_provider_preferences": routing,
            "execution_mode": "concurrent",
        },
    )
    print(f"wrote {len(lines)} independent judge requests to {request_path}")


def require_matching_judge_run(
    setting_path: Path,
    run_dir: Path,
    prediction_output: Path,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    api_provider: str = "openrouter",
    max_prompt_price: float = DEFAULT_OPENROUTER_MAX_PROMPT_PRICE,
    max_completion_price: float = DEFAULT_OPENROUTER_MAX_COMPLETION_PRICE,
    allow_azure: bool = False,
    judge_instructions_path: Path | None = None,
) -> None:
    routing = provider_preferences(
        api_provider, max_prompt_price, max_completion_price, allow_azure
    )
    setting = load_setting(setting_path)
    expected_judgments = setting["judge_request_count"]
    load_prediction_results(setting, prediction_output)
    request_path = run_dir / "judge_requests.jsonl"
    judge_run_path = run_dir / "judge_run.json"
    present = [path for path in (request_path, judge_run_path) if path.exists()]
    if not present:
        prepare_judging(
            setting_path,
            run_dir,
            prediction_output,
            model,
            max_output_tokens,
            reasoning_effort,
            api_provider,
            max_prompt_price,
            max_completion_price,
            allow_azure,
            judge_instructions_path,
        )
        return
    if len(present) != 2:
        raise ValueError(
            f"incomplete judge preparation in {run_dir}; expected both "
            "judge_run.json and judge_requests.jsonl"
        )

    judge_run = json.loads(judge_run_path.read_text(encoding="utf-8"))
    expected = {
        "prediction_output": recorded_path(prediction_output),
        "judge_model": model,
        "judge_instructions_source": judge_rubric_source(judge_instructions_path),
        "judge_reasoning_effort": reasoning_effort,
        "judge_max_output_tokens": max_output_tokens,
        "judge_request_count": expected_judgments,
        "judge_api_provider": api_provider,
        "judge_provider_preferences": routing,
        "execution_mode": "concurrent",
    }
    mismatches = {
        field: {"existing": judge_run.get(field), "requested": value}
        for field, value in expected.items()
        if judge_run.get(field) != value
    }
    if mismatches:
        raise ValueError(f"existing judge run does not match this command: {mismatches}")
    requests = load_request_file(request_path)
    if len(requests) != expected_judgments or {item["body"]["model"] for item in requests} != {model}:
        raise ValueError(f"unexpected judge request contents: {request_path}")


def run_predictions(
    setting_path: Path,
    run_dir: Path,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    concurrency: int,
    max_attempts: int,
    confirm_submit: bool,
    api_provider: str = "openrouter",
    max_prompt_price: float = DEFAULT_OPENROUTER_MAX_PROMPT_PRICE,
    max_completion_price: float = DEFAULT_OPENROUTER_MAX_COMPLETION_PRICE,
    allow_azure: bool = False,
) -> None:
    require_matching_prediction_run(
        setting_path,
        run_dir,
        model,
        max_output_tokens,
        reasoning_effort,
        api_provider,
        max_prompt_price,
        max_completion_price,
        allow_azure,
    )
    request_path = run_dir / "prediction_requests.jsonl"
    output = run_dir / "prediction_output.jsonl"
    error_output = run_dir / "prediction_errors.jsonl"
    run_concurrent_requests(
        request_path,
        output,
        error_output,
        "Predictions",
        api_provider,
        concurrency,
        max_attempts,
        confirm_submit,
        validate_prediction_response,
    )
    setting = load_setting(setting_path)
    predictions, _ = load_prediction_results(setting, output)
    print(f"validated {len(predictions)} predictions; ready to run judging")


def run_judging(
    setting_path: Path,
    run_dir: Path,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    concurrency: int,
    max_attempts: int,
    confirm_submit: bool,
    api_provider: str = "openrouter",
    max_prompt_price: float = DEFAULT_OPENROUTER_MAX_PROMPT_PRICE,
    max_completion_price: float = DEFAULT_OPENROUTER_MAX_COMPLETION_PRICE,
    allow_azure: bool = False,
    judge_instructions_path: Path | None = None,
) -> None:
    prediction_output = run_dir / "prediction_output.jsonl"
    if not prediction_output.exists():
        raise FileNotFoundError(
            f"prediction output is missing: {prediction_output}; run run-predictions first"
        )
    require_matching_judge_run(
        setting_path,
        run_dir,
        prediction_output,
        model,
        max_output_tokens,
        reasoning_effort,
        api_provider,
        max_prompt_price,
        max_completion_price,
        allow_azure,
        judge_instructions_path,
    )
    request_path = run_dir / "judge_requests.jsonl"
    output = run_dir / "judge_output.jsonl"
    error_output = run_dir / "judge_errors.jsonl"
    run_concurrent_requests(
        request_path,
        output,
        error_output,
        "Judgments",
        api_provider,
        concurrency,
        max_attempts,
        confirm_submit,
        validate_judgment_response,
    )
    aggregate(setting_path, run_dir, output)


def validate_judgment(custom_id: str, judgment: dict) -> None:
    for metric in JUDGE_METRICS:
        value = judgment.get(metric)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 2:
            raise ValueError(f"{custom_id} has invalid {metric}")
    if not isinstance(judgment.get("justification"), str) or not judgment["justification"].strip():
        raise ValueError(f"{custom_id} has no justification")


def validate_judgment_response(custom_id: str, text: str) -> None:
    validate_judgment(custom_id, json.loads(text))


def average(rows: list[dict], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def headline_score(judgment: dict) -> int:
    return sum(judgment[metric] for metric in HEADLINE_COMPONENTS)


def metric_summary(rows: list[dict]) -> dict[str, float]:
    return {metric: average(rows, metric) for metric in REPORTED_METRICS}


def score_row(case: dict, method: str, judgment: dict, **extra) -> dict:
    """One scores.csv row; `extra` columns go between the method and the scores."""
    return {
        "evaluation_idx": case["evaluation_idx"],
        "case_id": case["case_id"],
        "target_time_step_id": case["target_time_step_id"],
        "observed_decision_count": case["observed_decision_count"],
        "target_decision_id": case["target"]["decision_id"],
        "target_category": case["target"]["category"],
        "target_decision": case["target"]["decision"],
        "method": method,
        **extra,
        HEADLINE_METRIC: headline_score(judgment),
        **{metric: judgment[metric] for metric in JUDGE_METRICS},
        "justification": judgment["justification"],
    }


def aggregate_cell(rows: list[dict], aggregation: str) -> dict[str, float]:
    """Micro (over cases) or macro (over repository means) averages of every metric."""
    if aggregation == "micro":
        return {metric: statistics.fmean(row[metric] for row in rows) for metric in REPORTED_METRICS}
    indices = sorted({row["evaluation_idx"] for row in rows})
    return {
        metric: statistics.fmean(
            statistics.fmean(row[metric] for row in rows if row["evaluation_idx"] == index)
            for index in indices
        )
        for metric in REPORTED_METRICS
    }


def run_models(prediction_run: dict, judge_run: dict) -> dict:
    """The predictor and judge recorded in a run's prediction and judge records."""
    return {
        **{
            field: prediction_run[field]
            for field in (
                "prediction_model",
                "prediction_api_provider",
                "prediction_provider_preferences",
                "prediction_reasoning_effort",
            )
        },
        **{
            field: judge_run[field]
            for field in (
                "judge_model",
                "judge_api_provider",
                "judge_provider_preferences",
                "judge_reasoning_effort",
            )
        },
    }


def load_run_settings(run_dir: Path) -> dict:
    """Models recorded in a completed run."""
    return run_models(read_json(run_dir / "run.json"), read_json(run_dir / "judge_run.json"))


# Summaries of the method-comparison settings (skills, demos, rag): each setting's methods at
# prefix lengths 1-3 over its repeats, contrasts with other methods, and the paper's table rows.
SUMMARY_PREFIX_LENGTHS = (1, 2, 3)
METHOD_LABELS = {
    "baseline": "Baseline",
    "demonstrations": "All demos",
    "random_two_ordered": "Random two demos",
    "retrieved_two_ordered": "Retrieve two demos",
    "skill": "Skills",
    "shuffled_skills": "Shuffled skill",
    "final_paper_skill": "Final paper skill",
}


def summary_row(repeat: int, row: dict, prefix_length: int) -> dict:
    return {
        "repeat": repeat,
        "prefix_length": prefix_length,
        "method": row["method"],
        "evaluation_idx": int(row["evaluation_idx"]),
        "case_id": row["case_id"],
        "target_category": row["target_category"],
        **{metric: float(row[metric]) for metric in REPORTED_METRICS},
    }


def load_summary_scores(
    sources: tuple[tuple[Path, tuple[str, ...]], ...],
    case_prefix: dict[str, int],
    repeats: int,
) -> tuple[list[dict], dict]:
    """Score rows of each (run root, methods) source in runs <run root>-r1, -r2, ...,
    checked to cover every case once and to share the same models."""
    rows = []
    expected_settings = None
    for repeat in range(1, repeats + 1):
        for run_root, methods in sources:
            run_dir = Path(f"{run_root}-r{repeat}")
            settings = load_run_settings(run_dir)
            if expected_settings is None:
                expected_settings = settings
            elif canonical_json(settings) != canonical_json(expected_settings):
                raise ValueError(f"model settings of {run_dir} differ from the other runs")
            scores = [
                row
                for row in read_scores(run_dir / "results" / "scores.csv")
                if row["method"] in methods
            ]
            for method in methods:
                case_ids = [row["case_id"] for row in scores if row["method"] == method]
                if sorted(case_ids) != sorted(case_prefix):
                    raise ValueError(f"{run_dir} does not score every case once with {method}")
            rows.extend(
                summary_row(repeat, row, case_prefix[row["case_id"]]) for row in scores
            )
    return rows, expected_settings


def summarize_methods(
    rows: list[dict],
    repositories: set[int],
    methods: tuple[str, ...],
    contrasts: tuple[tuple[str, str, str], ...],
    repeats: int,
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Per-repeat means, contrasts, and paired case deltas over the held-out repositories.
    Writes the CSV tables of `methods` and the contrasts, whose other operands may be methods
    of other settings, and returns the across-repeat summaries."""
    scored = (*methods, *sorted({row["method"] for row in rows} - set(methods)))
    method_rows = []
    for repeat in range(1, repeats + 1):
        for aggregation in ("micro", "macro"):
            for prefix in SUMMARY_PREFIX_LENGTHS:
                for method in scored:
                    cell = [
                        row
                        for row in rows
                        if row["repeat"] == repeat
                        and row["evaluation_idx"] in repositories
                        and row["prefix_length"] == prefix
                        and row["method"] == method
                    ]
                    if len(cell) != len(repositories):
                        raise ValueError(
                            f"repeat={repeat} prefix={prefix} "
                            f"method={method} has {len(cell)} rows"
                        )
                    method_rows.append(
                        {
                            "repeat": repeat,
                            "aggregation": aggregation,
                            "prefix_length": prefix,
                            "method": method,
                            "repository_count": len(repositories),
                            "case_count": len(cell),
                            **aggregate_cell(cell, aggregation),
                        }
                    )

    summary_index = {
        (
            row["repeat"],
            row["aggregation"],
            row["prefix_length"],
            row["method"],
        ): row
        for row in method_rows
    }
    contrast_rows = []
    paired_rows = []
    for repeat in range(1, repeats + 1):
        for aggregation in ("micro", "macro"):
            for prefix in SUMMARY_PREFIX_LENGTHS:
                for name, minuend, subtrahend in contrasts:
                    left = summary_index[(repeat, aggregation, prefix, minuend)]
                    right = summary_index[(repeat, aggregation, prefix, subtrahend)]
                    contrast_rows.append(
                        {
                            "repeat": repeat,
                            "aggregation": aggregation,
                            "prefix_length": prefix,
                            "contrast": name,
                            "repository_count": len(repositories),
                            **{
                                metric: left[metric] - right[metric]
                                for metric in REPORTED_METRICS
                            },
                        }
                    )
        indexed = {
            (row["case_id"], row["method"]): row
            for row in rows
            if row["repeat"] == repeat
            and row["evaluation_idx"] in repositories
        }
        for case_id in sorted({key[0] for key in indexed}):
            for name, minuend, subtrahend in contrasts:
                left = indexed[(case_id, minuend)]
                right = indexed[(case_id, subtrahend)]
                paired_rows.append(
                    {
                        "repeat": repeat,
                        "evaluation_idx": left["evaluation_idx"],
                        "case_id": case_id,
                        "target_category": left["target_category"],
                        "prefix_length": left["prefix_length"],
                        "contrast": name,
                        **{
                            metric: left[metric] - right[metric]
                            for metric in REPORTED_METRICS
                        },
                    }
                )

    across_methods = []
    for aggregation in ("micro", "macro"):
        for prefix in SUMMARY_PREFIX_LENGTHS:
            for method in methods:
                cell = [
                    row
                    for row in method_rows
                    if row["aggregation"] == aggregation
                    and row["prefix_length"] == prefix
                    and row["method"] == method
                ]
                trajectory = [row[HEADLINE_METRIC] for row in cell]
                across_methods.append(
                    {
                        "aggregation": aggregation,
                        "prefix_length": prefix,
                        "method": method,
                        "mean_trajectory": statistics.fmean(trajectory),
                        "sample_sd": (
                            statistics.stdev(trajectory)
                            if len(trajectory) > 1
                            else 0.0
                        ),
                        **{
                            f"mean_{metric}": statistics.fmean(
                                row[metric] for row in cell
                            )
                            for metric in JUDGE_METRICS
                        },
                    }
                )

    across_contrasts = []
    for aggregation in ("micro", "macro"):
        for prefix in SUMMARY_PREFIX_LENGTHS:
            for name, _, _ in contrasts:
                cell = [
                    row
                    for row in contrast_rows
                    if row["aggregation"] == aggregation
                    and row["prefix_length"] == prefix
                    and row["contrast"] == name
                ]
                trajectory = [row[HEADLINE_METRIC] for row in cell]
                across_contrasts.append(
                    {
                        "aggregation": aggregation,
                        "prefix_length": prefix,
                        "contrast": name,
                        "mean_delta": statistics.fmean(trajectory),
                        "sample_sd": (
                            statistics.stdev(trajectory)
                            if len(trajectory) > 1
                            else 0.0
                        ),
                        "repeat_deltas": " ".join(
                            f"{value:.3f}" for value in trajectory
                        ),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "scores.csv", [row for row in rows if row["method"] in methods])
    write_csv(
        output_dir / "repeat_summary.csv",
        [row for row in method_rows if row["method"] in methods],
    )
    write_csv(output_dir / "repeat_contrasts.csv", contrast_rows)
    write_csv(output_dir / "paired_case_deltas.csv", paired_rows)
    write_csv(output_dir / "across_repeat_summary.csv", across_methods)
    write_csv(output_dir / "across_repeat_contrasts.csv", across_contrasts)
    return across_methods, across_contrasts


def table_rows(across_methods: list[dict], methods: tuple[str, ...]) -> str:
    """The paper's table rows: micro mean ± sample SD across repeats at each prefix length."""
    cells = {
        (row["method"], row["prefix_length"]): row
        for row in across_methods
        if row["aggregation"] == "micro"
    }
    lines = [
        "| Method | " + " | ".join(f"Prefix-{k}" for k in SUMMARY_PREFIX_LENGTHS) + " |",
        "| --- |" + " ---: |" * len(SUMMARY_PREFIX_LENGTHS),
    ]
    for method in methods:
        values = [
            f"{cells[method, k]['mean_trajectory']:.3f} ± "
            + f"{cells[method, k]['sample_sd']:.3f}".lstrip("0")
            for k in SUMMARY_PREFIX_LENGTHS
        ]
        lines.append(f"| {METHOD_LABELS[method]} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def summarize_comparison(
    setting_path: Path,
    setting: dict,
    methods: tuple[str, ...],
    sources: tuple[tuple[Path, tuple[str, ...]], ...],
    contrasts: tuple[tuple[str, str], ...],
    repeats: int,
    output_dir: Path,
    details: dict | None = None,
) -> None:
    """Summarize a setting's methods over its repeats into output_dir. `sources` lists each
    (run root, methods) to read: the setting's own runs, and other settings' runs for the
    methods its contrasts compare against. table.md holds the setting's rows of the paper's table."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    case_prefix = {
        case["case_id"]: int(case["natural_prefix_length"]) for case in setting["cases"]
    }
    rows, model_settings = load_summary_scores(sources, case_prefix, repeats)
    repositories = set(setting["evaluation_indices"])
    named_contrasts = tuple((f"{left}_minus_{right}", left, right) for left, right in contrasts)
    across_methods, across_contrasts = summarize_methods(
        rows, repositories, methods, named_contrasts, repeats, output_dir
    )
    table = table_rows(across_methods, methods)
    (output_dir / "table.md").write_text(table, encoding="utf-8")
    write_json(
        output_dir / "summary.json",
        {
            "setting": recorded_path(setting_path),
            "runs": [
                {"run_root": recorded_path(run_root), "methods": list(source_methods)}
                for run_root, source_methods in sources
            ],
            "repeats": repeats,
            "prefix_lengths": list(SUMMARY_PREFIX_LENGTHS),
            "repository_count": len(repositories),
            "methods": list(methods),
            "model_settings": model_settings,
            **(details or {}),
            "across_repeat_summary": across_methods,
            "across_repeat_contrasts": across_contrasts,
        },
    )
    print("aggregation\tprefix\tmethod\tmean_trajectory\tsample_sd")
    for row in across_methods:
        print(
            f"{row['aggregation']}\t{row['prefix_length']}\t{row['method']}\t"
            f"{row['mean_trajectory']:.3f}\t{row['sample_sd']:.3f}"
        )
    print("\naggregation\tprefix\tcontrast\tmean_delta\tsample_sd\trepeat_deltas")
    for row in across_contrasts:
        print(
            f"{row['aggregation']}\t{row['prefix_length']}\t{row['contrast']}\t"
            f"{row['mean_delta']:+.3f}\t{row['sample_sd']:.3f}\t{row['repeat_deltas']}"
        )
    print(f"\n{table}\nwrote the summary to {output_dir}")


def score_rows(setting: dict, judge_output: Path) -> tuple[list[dict], dict[str, int]]:
    methods = tuple(setting["methods"])
    expected = {
        judge_custom_id(case["case_id"], method)
        for case in setting["cases"]
        for method in methods
    }
    judgments, judge_usage = read_response_output(judge_output, expected)

    rows = []
    for case in setting["cases"]:
        for method in methods:
            custom_id = judge_custom_id(case["case_id"], method)
            judgment = judgments[custom_id]
            validate_judgment(custom_id, judgment)
            rows.append(score_row(case, method, judgment))
    if len(rows) != setting["judge_request_count"]:
        raise ValueError(
            f"aggregated {len(rows)} scores, expected {setting['judge_request_count']}"
        )
    return rows, judge_usage


def write_score_tables(results_dir: Path, rows: list[dict], repository_rows: list[dict]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    score_fields = list(rows[0])
    with (results_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_fields)
        writer.writeheader()
        writer.writerows(rows)
    with (results_dir / "per_repository.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(repository_rows[0]))
        writer.writeheader()
        writer.writerows(repository_rows)


def summarize_rows(
    rows: list[dict],
    evaluation_indices: list[int],
    methods: tuple[str, ...],
    case_ids: list[str],
) -> tuple[list[dict], dict, dict, dict]:
    repository_rows = []
    for index in evaluation_indices:
        for method in methods:
            subset = [
                row
                for row in rows
                if row["evaluation_idx"] == index and row["method"] == method
            ]
            summary = metric_summary(subset)
            repository_rows.append(
                {"evaluation_idx": index, "method": method, "case_count": len(subset), **summary}
            )

    micro = {
        method: metric_summary([row for row in rows if row["method"] == method])
        for method in methods
    }
    macro = {
        method: {
            metric: statistics.fmean(
                row[metric]
                for row in repository_rows
                if row["method"] == method
            )
            for metric in REPORTED_METRICS
        }
        for method in methods
    }
    case_scores = {
        (row["case_id"], row["method"]): row[HEADLINE_METRIC] for row in rows
    }
    paired_deltas = {}
    if "baseline" in methods:
        paired_deltas = {
            f"{method}_minus_baseline": statistics.fmean(
                case_scores[(case_id, method)]
                - case_scores[(case_id, "baseline")]
                for case_id in case_ids
            )
            for method in methods
            if method != "baseline"
        }
    return repository_rows, micro, macro, paired_deltas


def aggregate(setting_path: Path, run_dir: Path, judge_output: Path) -> None:
    setting = load_setting(setting_path)
    methods = tuple(setting["methods"])
    run = read_json(run_dir / "run.json")
    judge_run = read_json(run_dir / "judge_run.json")
    rows, judge_usage = score_rows(setting, judge_output)
    repository_rows, micro, macro, paired_deltas = summarize_rows(
        rows,
        setting["evaluation_indices"],
        methods,
        [case["case_id"] for case in setting["cases"]],
    )
    results_dir = run_dir / "results"
    write_score_tables(results_dir, rows, repository_rows)
    summary = {
        "case_count": setting["case_count"],
        "scored_prediction_count": setting["prediction_request_count"],
        **run_models(run, judge_run),
        "micro_average_over_all_cases": micro,
        "macro_average_over_repository_means": macro,
        "paired_{}_deltas".format(HEADLINE_METRIC): paired_deltas,
        "prediction_usage": judge_run["prediction_usage"],
        "judge_usage": judge_usage,
    }
    write_json(results_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def add_execution_options(parser: argparse.ArgumentParser, max_output_tokens: int) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=max_output_tokens)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--confirm-submit", action="store_true")
    add_api_options(parser)


def add_api_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-provider",
        choices=("openrouter", "openai"),
        default="openrouter",
        help="regular API service to use (default: openrouter)",
    )
    parser.add_argument(
        "--openrouter-max-prompt-price",
        type=float,
        default=DEFAULT_OPENROUTER_MAX_PROMPT_PRICE,
        metavar="USD_PER_MILLION",
        help="hard OpenRouter provider prompt-price ceiling (default: 4)",
    )
    parser.add_argument(
        "--openrouter-max-completion-price",
        type=float,
        default=DEFAULT_OPENROUTER_MAX_COMPLETION_PRICE,
        metavar="USD_PER_MILLION",
        help="hard OpenRouter provider completion-price ceiling (default: 15)",
    )
    parser.add_argument(
        "--allow-azure",
        action="store_true",
        help="allow OpenRouter to route to Azure; Azure is excluded by default",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="prepare every eligible target from the configured evaluation indices"
    )
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_SETTING)
    prepare_parser.add_argument("--replace", action="store_true")

    full_prepare_parser = subparsers.add_parser(
        "prepare-full",
        help="prepare the selected methods over every eligible base-setting target",
    )
    full_prepare_parser.add_argument(
        "--output", type=Path, default=DEFAULT_FULL_SETTING
    )
    full_prepare_parser.add_argument(
        "--base-setting", type=Path, default=DEFAULT_SETTING
    )
    full_prepare_parser.add_argument(
        "--ordered-skill", type=Path, default=DEFAULT_ORDERED_SKILL
    )
    full_prepare_parser.add_argument(
        "--ordered-demonstrations",
        type=Path,
        default=DEFAULT_ORDERED_DEMONSTRATIONS,
    )
    full_prepare_parser.add_argument(
        "--shuffled-demonstrations",
        type=Path,
        default=DEFAULT_SHUFFLED_DEMONSTRATIONS,
    )
    full_prepare_parser.add_argument(
        "--shuffle-metadata", type=Path, default=DEFAULT_SHUFFLE_METADATA
    )
    full_prepare_parser.add_argument(
        "--shuffled-skill", type=Path, default=DEFAULT_SHUFFLED_SKILL
    )
    full_prepare_parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=FULL_METHODS,
        help="prepare only these methods (default: the five standard methods)",
    )
    full_prepare_parser.add_argument("--replace", action="store_true")

    prediction_parser = subparsers.add_parser(
        "prepare-predictions", help="write the setting-defined prediction requests"
    )
    prediction_parser.add_argument("--setting", type=Path, default=DEFAULT_FULL_SETTING)
    prediction_parser.add_argument("--run-dir", type=Path, required=True)
    prediction_parser.add_argument("--model", required=True)
    prediction_parser.add_argument("--max-output-tokens", type=int, default=4000)
    prediction_parser.add_argument("--reasoning-effort")
    add_api_options(prediction_parser)

    run_prediction_parser = subparsers.add_parser(
        "run-predictions",
        help="prepare and run concurrent stateless prediction requests",
    )
    run_prediction_parser.add_argument("--setting", type=Path, default=DEFAULT_FULL_SETTING)
    run_prediction_parser.add_argument("--run-dir", type=Path, required=True)
    add_execution_options(run_prediction_parser, 4000)

    judge_parser = subparsers.add_parser(
        "prepare-judging", help="write the setting-defined independent judge requests"
    )
    judge_parser.add_argument("--setting", type=Path, default=DEFAULT_FULL_SETTING)
    judge_parser.add_argument("--run-dir", type=Path, required=True)
    judge_parser.add_argument("--prediction-output", type=Path, required=True)
    judge_parser.add_argument("--model", required=True)
    judge_parser.add_argument("--max-output-tokens", type=int, default=3000)
    judge_parser.add_argument("--reasoning-effort")
    judge_parser.add_argument(
        "--judge-instructions",
        type=Path,
        help="score with this rubric file instead of the setting's prepared copy",
    )
    add_api_options(judge_parser)

    run_judge_parser = subparsers.add_parser(
        "run-judging",
        help="prepare and run concurrent stateless judgments, then aggregate",
    )
    run_judge_parser.add_argument("--setting", type=Path, default=DEFAULT_FULL_SETTING)
    run_judge_parser.add_argument("--run-dir", type=Path, required=True)
    run_judge_parser.add_argument(
        "--judge-instructions",
        type=Path,
        help="score with this rubric file instead of the setting's prepared copy",
    )
    add_execution_options(run_judge_parser, 3000)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="validate all judgments and compute target-alignment averages"
    )
    aggregate_parser.add_argument("--setting", type=Path, default=DEFAULT_FULL_SETTING)
    aggregate_parser.add_argument("--run-dir", type=Path, required=True)
    aggregate_parser.add_argument("--judge-output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_suite(args.output, args.replace)
    elif args.command == "prepare-full":
        prepare_full_suite(
            args.output,
            args.base_setting,
            args.ordered_skill,
            args.ordered_demonstrations,
            args.shuffled_demonstrations,
            args.shuffle_metadata,
            args.shuffled_skill,
            args.replace,
            tuple(args.methods),
        )
    elif args.command == "prepare-predictions":
        prepare_predictions(
            args.setting,
            args.run_dir,
            args.model,
            args.max_output_tokens,
            args.reasoning_effort,
            args.api_provider,
            args.openrouter_max_prompt_price,
            args.openrouter_max_completion_price,
            args.allow_azure,
        )
    elif args.command == "run-predictions":
        run_predictions(
            args.setting,
            args.run_dir,
            args.model,
            args.max_output_tokens,
            args.reasoning_effort,
            args.concurrency,
            args.max_attempts,
            args.confirm_submit,
            args.api_provider,
            args.openrouter_max_prompt_price,
            args.openrouter_max_completion_price,
            args.allow_azure,
        )
    elif args.command == "prepare-judging":
        prepare_judging(
            args.setting,
            args.run_dir,
            args.prediction_output,
            args.model,
            args.max_output_tokens,
            args.reasoning_effort,
            args.api_provider,
            args.openrouter_max_prompt_price,
            args.openrouter_max_completion_price,
            args.allow_azure,
            args.judge_instructions,
        )
    elif args.command == "run-judging":
        run_judging(
            args.setting,
            args.run_dir,
            args.model,
            args.max_output_tokens,
            args.reasoning_effort,
            args.concurrency,
            args.max_attempts,
            args.confirm_submit,
            args.api_provider,
            args.openrouter_max_prompt_price,
            args.openrouter_max_completion_price,
            args.allow_azure,
            args.judge_instructions,
        )
    else:
        aggregate(args.setting, args.run_dir, args.judge_output)


if __name__ == "__main__":
    main()
