#!/usr/bin/env python3
"""Distill ordered and shuffled trajectory skills with one shared configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import experiment


HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "prompts" / "distill_skill.md"
DEFAULT_RUN_DIR = (
    HERE
    / "runs"
    / "skill-distillation-normalized-actions-openrouter-concurrent-default-decoding"
)
DISTILLATION_SKILL_NAME = "research-decision-prediction"
MIN_SKILL_BODY_WORDS = 400
MAX_SKILL_BODY_WORDS = 1500
CONDITIONS = {
    "ordered": {
        "custom_id": "distill-ordered-research-trajectory-skill",
        "skill_name": "trajectory",
        "demonstrations": HERE / "prompts" / "demonstrations.md",
        "output": HERE / "skills" / "trajectory" / "SKILL.md",
    },
    "shuffled": {
        "custom_id": "distill-shuffled-research-trajectory-skill",
        "skill_name": "trajectory-shuffled",
        "demonstrations": HERE / "prompts" / "shuffled_demonstrations.md",
        "output": HERE / "skills" / "trajectory-shuffled" / "SKILL.md",
    },
}


def request_body(
    model: str,
    prompt: str,
    demonstrations: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    routing: dict | None,
) -> dict:
    body = {
        "model": model,
        "instructions": prompt.strip(),
        "input": (
            f"Required SKILL.md frontmatter name: {DISTILLATION_SKILL_NAME}\n\n"
            + demonstrations.strip()
        ),
        "store": False,
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    if routing:
        body["provider"] = routing
    return body


def validate_skill(text: str, expected_name: str) -> str:
    text = text.strip()
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("distillation output must begin with YAML frontmatter, not a code fence")
    try:
        frontmatter_end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("distillation output has unterminated YAML frontmatter") from exc
    frontmatter = lines[1:frontmatter_end]
    name_lines = [line for line in frontmatter if line.startswith("name:")]
    if not name_lines:
        raise ValueError("distillation output frontmatter has no name")
    name = name_lines[0].split(":", 1)[1].strip()
    if name != expected_name or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError(
            f"distillation output skill name must be {expected_name!r}, got {name!r}"
        )
    description_lines = [
        line for line in frontmatter if line.startswith("description:")
    ]
    if not description_lines or not description_lines[0].split(":", 1)[1].strip():
        raise ValueError("distillation output frontmatter has no description")
    if not any(line.strip() for line in lines[frontmatter_end + 1 :]):
        raise ValueError("distillation output has no skill body")
    body = "\n".join(lines[frontmatter_end + 1 :])
    word_count = len(re.findall(r"\b[\w'-]+\b", body))
    if not MIN_SKILL_BODY_WORDS <= word_count <= MAX_SKILL_BODY_WORDS:
        raise ValueError(
            "distillation output skill body must contain "
            f"{MIN_SKILL_BODY_WORDS}-{MAX_SKILL_BODY_WORDS} words; got {word_count}"
        )
    return text + "\n"


def materialize_skill_name(text: str, output_name: str) -> str:
    text = validate_skill(text, DISTILLATION_SKILL_NAME)
    lines = text.rstrip().splitlines()
    frontmatter_end = lines.index("---", 1)
    name_indices = [
        index
        for index, line in enumerate(lines[1:frontmatter_end], 1)
        if line.startswith("name:")
    ]
    if len(name_indices) != 1:
        raise ValueError("distillation output frontmatter must contain exactly one name")
    lines[name_indices[0]] = f"name: {output_name}"
    return validate_skill("\n".join(lines), output_name)


def validate_skill_response(custom_id: str, text: str) -> None:
    expected_ids = {config["custom_id"] for config in CONDITIONS.values()}
    if custom_id not in expected_ids:
        raise ValueError(f"unexpected skill distillation response ID: {custom_id}")
    validate_skill(text, DISTILLATION_SKILL_NAME)


def prepare_requests(
    prompt: Path,
    run_dir: Path,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    api_provider: str,
    max_prompt_price: float,
    max_completion_price: float,
    allow_azure: bool,
) -> tuple[Path, Path]:
    if not model.strip():
        raise ValueError("model must not be empty")
    if max_output_tokens <= 0:
        raise ValueError("max-output-tokens must be positive")
    routing = experiment.provider_preferences(
        api_provider, max_prompt_price, max_completion_price, allow_azure
    )
    prompt_text = prompt.read_text(encoding="utf-8")
    requests = []
    condition_metadata = {}
    for condition, config in CONDITIONS.items():
        demonstrations = config["demonstrations"]
        output = config["output"]
        body = request_body(
            model,
            prompt_text,
            demonstrations.read_text(encoding="utf-8"),
            max_output_tokens,
            reasoning_effort,
            routing,
        )
        requests.append(experiment.api_request(config["custom_id"], body))
        condition_metadata[condition] = {
            "custom_id": config["custom_id"],
            "skill_name": config["skill_name"],
            "demonstrations": experiment.recorded_path(demonstrations),
            "output": experiment.recorded_path(output),
        }

    request_text = "".join(
        json.dumps(request, ensure_ascii=False) + "\n" for request in requests
    )
    request_path = run_dir / "distillation_requests.jsonl"
    run_path = run_dir / "run.json"
    present = [path for path in (request_path, run_path) if path.exists()]
    if not present:
        run_dir.mkdir(parents=True, exist_ok=True)
        request_path.write_text(request_text, encoding="utf-8")
        experiment.write_json(
            run_path,
            {
                "format_version": 1,
                "prompt": experiment.recorded_path(prompt),
                "model": model,
                "reasoning_effort": reasoning_effort,
                "max_output_tokens": max_output_tokens,
                "api_provider": api_provider,
                "provider_preferences": routing,
                "execution_mode": "concurrent",
                "request_count": len(requests),
                "conditions": condition_metadata,
            },
        )
        print(f"wrote two-condition skill distillation request to {request_path}")
        return request_path, run_path
    if len(present) != 2:
        raise ValueError(
            f"incomplete distillation preparation in {run_dir}; expected request and run.json"
        )

    run = json.loads(run_path.read_text(encoding="utf-8"))
    expected = {
        "prompt": experiment.recorded_path(prompt),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "api_provider": api_provider,
        "provider_preferences": routing,
        "execution_mode": "concurrent",
        "request_count": len(CONDITIONS),
    }
    mismatches = {
        field: {"existing": run.get(field), "requested": value}
        for field, value in expected.items()
        if run.get(field) != value
    }
    for condition, metadata in condition_metadata.items():
        existing = run.get("conditions", {}).get(condition, {})
        for field in (
            "custom_id",
            "skill_name",
            "demonstrations",
            "output",
        ):
            if existing.get(field) != metadata[field]:
                mismatches[f"conditions.{condition}.{field}"] = {
                    "existing": existing.get(field),
                    "requested": metadata[field],
                }
    if mismatches:
        raise ValueError(f"existing distillation run does not match this command: {mismatches}")
    requests = experiment.load_request_file(request_path)
    if len(requests) != len(CONDITIONS) or {
        request["body"]["model"] for request in requests
    } != {model}:
        raise ValueError(f"unexpected distillation request contents: {request_path}")
    return request_path, run_path


def write_skills(response_output: Path, run_path: Path) -> None:
    expected_by_id = {
        config["custom_id"]: (condition, config)
        for condition, config in CONDITIONS.items()
    }
    items = {}
    for line_number, line in enumerate(
        response_output.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item.get("custom_id")
        if custom_id in items:
            raise ValueError(f"duplicate custom_id at {response_output}:{line_number}")
        items[custom_id] = item
    if set(items) != set(expected_by_id):
        raise ValueError(
            f"{response_output} result IDs differ from the two distillation conditions"
        )

    run = json.loads(run_path.read_text(encoding="utf-8"))
    generated = {}
    for custom_id, (condition, config) in expected_by_id.items():
        skill_text, usage = experiment.extract_response_text(items[custom_id])
        skill_text = materialize_skill_name(skill_text, config["skill_name"])
        generated[condition] = {"text": skill_text, "usage": usage}

    for condition, config in CONDITIONS.items():
        output = config["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(generated[condition]["text"], encoding="utf-8")
        run["conditions"][condition]["usage"] = generated[condition]["usage"]
        print(f"wrote {condition} skill to {output}")
    experiment.write_json(run_path, run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=3000)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--confirm-submit", action="store_true")
    experiment.add_api_options(parser)
    args = parser.parse_args()

    request_path, run_path = prepare_requests(
        args.prompt,
        args.run_dir,
        args.model,
        args.max_output_tokens,
        args.reasoning_effort,
        args.api_provider,
        args.openrouter_max_prompt_price,
        args.openrouter_max_completion_price,
        args.allow_azure,
    )
    response_output = args.run_dir / "distillation_output.jsonl"
    response_errors = args.run_dir / "distillation_errors.jsonl"
    experiment.run_concurrent_requests(
        request_path,
        response_output,
        response_errors,
        "Skill distillation",
        args.api_provider,
        args.concurrency,
        args.max_attempts,
        args.confirm_submit,
        validate_skill_response,
    )
    write_skills(response_output, run_path)

if __name__ == "__main__":
    main()
