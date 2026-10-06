#!/usr/bin/env python3
"""The skills setting: the ordered and shuffled prefix skills and the final paper skill.

`render` writes the training examples the prefix skills are distilled from, `distill`
distills the ordered and shuffled prefix skills into skills/, and `prepare` builds
experiments/skills/ from the method-comparison cases and the three skills.
`distill-final-paper` distills the final paper skill from the training projects' papers,
and `estimate-final-paper` estimates its cost. `summarize` writes the Baseline, Skills,
Shuffled skill, and Final paper skill rows from the skills runs.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import unicodedata
import urllib.error
import urllib.request

import build_prompt
import distill_skill
import experiment
import render_demonstrations


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SOURCE_SETTING = HERE / "experiments" / "cases" / "setting.json"
SETTING = HERE / "experiments" / "skills" / "setting.json"
ORDERED_SKILL = HERE / "skills" / "prefix" / "SKILL.md"
SHUFFLED_SKILL = HERE / "skills" / "prefix-shuffled" / "SKILL.md"
# Distilled from the training projects' published papers by `distill-final-paper`.
FINAL_PAPER_SKILL = HERE / "skills" / "final-paper" / "SKILL.md"
METHODS = ("baseline", "skill", "shuffled_skills", "final_paper_skill")
PREFIX_LENGTHS = (1, 2, 3)
DISTILLATION_DIR = HERE / "runs" / "skill-distillation-prefix-regimes-opus5"
SKILLS_RUN_ROOT = HERE / "runs" / "skills-gemini-3.1-flash-lite-sol-medium"
SUMMARY_DIR = HERE / "runs" / "skills-summary"
CONTRASTS = (
    ("skill", "baseline"),
    ("skill", "shuffled_skills"),
    ("final_paper_skill", "baseline"),
)
GENERATION_NAME = distill_skill.DISTILLATION_SKILL_NAME


# Distillation examples: each training project's first 1, 2, and 3 decisions with the next one.

def without_prediction_request(evaluation_input: str) -> str:
    if not evaluation_input.endswith(experiment.PREDICTION_SUFFIX):
        raise ValueError("rendered prefix has an unexpected prediction suffix")
    return evaluation_input[: -len(experiment.PREDICTION_SUFFIX)].rstrip()


def actual_next_decision(decision: dict) -> str:
    return "\n".join(
        [
            "Actual next annotated decision:",
            f"Category: {decision['category']}",
            f"Decision: {render_demonstrations.decision_action_text(decision['decision'])}",
        ]
    )


def render_example(label: str, evaluation_input: str, target: dict) -> str:
    return "\n\n".join(
        [
            f"### {label}",
            without_prediction_request(evaluation_input),
            actual_next_decision(target),
        ]
    )


def shuffled_decisions(index: int, shuffle_metadata: dict) -> list[dict]:
    path = render_demonstrations.annotation_path(index)
    annotation = json.loads(path.read_text(encoding="utf-8"))
    decisions = build_prompt.validate_annotation(path, annotation)
    metadata = shuffle_metadata["trajectories"].get(str(index))
    if not isinstance(metadata, dict):
        raise ValueError(f"shuffle metadata has no trajectory for idx={index}")

    permutation = metadata.get("shuffled_position_to_original_time_step_id")
    decision_id_map = metadata.get("original_decision_id_to_shuffled_decision_id")
    outcomes = metadata.get("randomized_outcomes")
    replacements = metadata.get("randomized_superseded_by")
    if not isinstance(permutation, list) or sorted(permutation) != list(
        range(len(decisions))
    ):
        raise ValueError(f"idx={index} has an invalid shuffled permutation")
    if not all(isinstance(value, dict) for value in (decision_id_map, outcomes, replacements)):
        raise ValueError(f"idx={index} has malformed shuffled decision metadata")

    time_step_id_map = {
        decisions[original_position]["time_step_id"]: f"T{position:03d}"
        for position, original_position in enumerate(permutation)
    }
    synthetic = []
    for position, original_position in enumerate(permutation):
        original = decisions[original_position]
        decision_id = decision_id_map[original["decision_id"]]
        synthetic.append(
            {
                "time_step_id": position,
                "decision_id": decision_id,
                "category": original["category"],
                "decision": render_demonstrations.remap_identifiers(
                    render_demonstrations.decision_action_text(original["decision"]),
                    decision_id_map,
                    time_step_id_map,
                ),
                "outcome": outcomes[decision_id],
                "superseded_by": replacements.get(decision_id),
            }
        )
    return synthetic


def render_shuffled_prefix(decisions: list[dict], target_step: int) -> str:
    if target_step <= 0 or target_step >= len(decisions):
        raise ValueError(f"target step must be between 1 and {len(decisions) - 1}")
    observed = decisions[:target_step]
    observed_by_id = {decision["decision_id"]: decision for decision in observed}
    lines = [
        "# Evaluation trajectory prefix",
        "",
        "All decisions introduced before the held-out decision are shown in synthetic presentation order. Outcomes and supersession links not established by the cutoff are withheld.",
        "",
        "Observed decisions:",
        "",
    ]
    for decision in observed:
        outcome = decision["outcome"]
        if outcome == "retained":
            state = "retained by cutoff"
        elif outcome == "abandoned":
            state = "abandoned by cutoff"
        elif outcome == "superseded":
            replacement = observed_by_id.get(decision["superseded_by"])
            state = (
                f"superseded by observed T{replacement['time_step_id']:03d}"
                if replacement
                else "retained by cutoff"
            )
        else:
            raise ValueError(
                f"shuffled decision {decision['decision_id']} has invalid outcome {outcome!r}"
            )
        lines.extend(
            [
                f"- T{decision['time_step_id']:03d} | {decision['category']} | {state}",
                f"  {decision['decision']}",
                "",
            ]
        )
    lines.append("Predict the next research decision. Return the required JSON only.")
    return "\n".join(lines)


def render_condition(
    training_indices: tuple[int, ...],
    shuffled: bool,
    shuffle_metadata: dict,
) -> tuple[str, int]:
    early_sections = [
        "# Natural early-prefix training examples",
        "",
        "Each example exposes the complete history available at a natural early cutoff and then gives the actual immediate next decision. No retrospective trajectory-insight summaries are included.",
    ]
    early_count = 0
    for index in training_indices:
        if shuffled:
            decisions = shuffled_decisions(index, shuffle_metadata)
            render_full_prefix = lambda step: render_shuffled_prefix(decisions, step)
        else:
            path = render_demonstrations.annotation_path(index)
            annotation = json.loads(path.read_text(encoding="utf-8"))
            decisions = build_prompt.validate_annotation(path, annotation)
            render_full_prefix = lambda step: build_prompt.render_prefix(decisions, step)
        if len(decisions) <= max(PREFIX_LENGTHS):
            raise ValueError(f"idx={index} needs at least four decisions")

        early_examples = []
        for target_step in PREFIX_LENGTHS:
            early_examples.append(
                render_example(
                    f"first_{target_step}",
                    render_full_prefix(target_step),
                    decisions[target_step],
                )
            )
            early_count += 1
        early_sections.extend(
            [f"\n## Training trajectory {index}", "\n\n".join(early_examples)]
        )

    early_text = "\n".join(early_sections).rstrip() + "\n"
    if "Trajectory insight:" in early_text:
        raise ValueError("rendered distillation data contains a trajectory insight")
    return early_text, early_count


def render_inputs(output_dir: Path, shuffle_metadata: Path) -> None:
    """Write the ordered and shuffled first-1/2/3 examples the prefix skills are distilled from."""
    training_indices = render_demonstrations.TRAIN_INDICES
    metadata = json.loads(shuffle_metadata.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 2:
        raise ValueError("shuffle metadata must use format version 2")
    if set(metadata.get("trajectories", {})) != {
        str(index) for index in training_indices
    }:
        raise ValueError("shuffle metadata indices differ from the current training split")

    ordered_early, early_count = render_condition(training_indices, False, metadata)
    shuffled_early, shuffled_early_count = render_condition(training_indices, True, metadata)
    if early_count != shuffled_early_count:
        raise ValueError("ordered and shuffled example counts differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "early_ordered": ordered_early,
        "early_shuffled": shuffled_early,
    }
    for name, text in outputs.items():
        (output_dir / f"{name}.md").write_text(text, encoding="utf-8")

    setting = {
        "format_version": 1,
        "training_indices": list(training_indices),
        "splits_source": experiment.recorded_path(render_demonstrations.SPLITS),
        "shuffle_metadata": experiment.recorded_path(shuffle_metadata),
        "source_annotations": [
            render_demonstrations.annotation_path(index).relative_to(REPO_ROOT).as_posix()
            for index in training_indices
        ],
        "early_examples_per_condition": early_count,
        "outputs": {
            f"{name}.md": {"bytes": len(text.encode("utf-8"))} for name, text in outputs.items()
        },
    }
    experiment.write_json(output_dir / "setting.json", setting)
    print(
        f"wrote ordered and shuffled insight-free corpora to {output_dir}: "
        f"{early_count} examples each"
    )


# Distillation of the ordered and shuffled prefix skills.

def conditions(run_dir: Path) -> dict[str, dict]:
    return {
        "early_ordered": {
            "custom_id": "distill-early-ordered-prefix-skill",
            "skill_name": "prefix",
            "prompt": HERE / "prompts" / "distill_skill_early_prefix.md",
            "examples": run_dir / "inputs" / "early_ordered.md",
            "output": HERE / "skills" / "prefix" / "SKILL.md",
        },
        "early_shuffled": {
            "custom_id": "distill-early-shuffled-prefix-skill",
            "skill_name": "prefix-shuffled",
            "prompt": HERE / "prompts" / "distill_skill_early_prefix.md",
            "examples": run_dir / "inputs" / "early_shuffled.md",
            "output": HERE / "skills" / "prefix-shuffled" / "SKILL.md",
        },
    }


def request_body(
    model: str,
    prompt: str,
    examples: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    routing: dict | None,
) -> dict:
    body = {
        "model": model,
        "instructions": prompt.strip(),
        "input": (
            f"Required SKILL.md frontmatter name: {GENERATION_NAME}\n\n"
            + examples.strip()
        ),
        "store": False,
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    if routing:
        body["provider"] = routing
    return body


def prepare_distillation(args: argparse.Namespace) -> tuple[Path, Path, dict[str, dict]]:
    if not args.model.strip():
        raise ValueError("model must not be empty")
    if args.max_output_tokens <= 0:
        raise ValueError("max-output-tokens must be positive")
    configs = conditions(args.run_dir)
    input_setting = args.run_dir / "inputs" / "setting.json"
    if not input_setting.is_file():
        raise FileNotFoundError(
            f"distillation inputs are missing: {input_setting}; render them first"
        )
    routing = experiment.provider_preferences(
        args.api_provider,
        args.openrouter_max_prompt_price,
        args.openrouter_max_completion_price,
        args.allow_azure,
    )

    requests = []
    condition_metadata = {}
    for condition, config in configs.items():
        prompt = config["prompt"]
        examples = config["examples"]
        output = config["output"]
        for required in (prompt, examples):
            if not required.is_file():
                raise FileNotFoundError(f"required distillation input is missing: {required}")
        body = request_body(
            args.model,
            prompt.read_text(encoding="utf-8"),
            examples.read_text(encoding="utf-8"),
            args.max_output_tokens,
            args.reasoning_effort,
            routing,
        )
        requests.append(experiment.api_request(config["custom_id"], body))
        condition_metadata[condition] = {
            "custom_id": config["custom_id"],
            "skill_name": config["skill_name"],
            "prompt": experiment.recorded_path(prompt),
            "examples": experiment.recorded_path(examples),
            "output": experiment.recorded_path(output),
        }

    request_text = "".join(
        json.dumps(request, ensure_ascii=False) + "\n" for request in requests
    )
    request_path = args.run_dir / "distillation_requests.jsonl"
    run_path = args.run_dir / "run.json"
    metadata = {
        "format_version": 1,
        "input_setting": experiment.recorded_path(input_setting),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "api_provider": args.api_provider,
        "provider_preferences": routing,
        "execution_mode": "concurrent",
        "request_count": len(requests),
        "conditions": condition_metadata,
    }
    present = [path for path in (request_path, run_path) if path.exists()]
    if not present:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        request_path.write_text(request_text, encoding="utf-8")
        experiment.write_json(run_path, metadata)
        print(f"wrote {len(requests)} distillation requests to {request_path}")
        return request_path, run_path, configs
    if len(present) != 2:
        raise ValueError(
            f"incomplete distillation plan in {args.run_dir}; expected request and run.json"
        )

    existing = json.loads(run_path.read_text(encoding="utf-8"))
    mismatches = {
        field: {"existing": existing.get(field), "requested": value}
        for field, value in metadata.items()
        if field != "conditions" and existing.get(field) != value
    }
    for condition, expected in condition_metadata.items():
        actual = existing.get("conditions", {}).get(condition, {})
        for field in (
            "custom_id",
            "skill_name",
            "prompt",
            "examples",
            "output",
        ):
            if actual.get(field) != expected[field]:
                mismatches[f"conditions.{condition}.{field}"] = {
                    "existing": actual.get(field),
                    "requested": expected[field],
                }
    if mismatches:
        raise ValueError(f"existing distillation plan does not match: {mismatches}")
    loaded = experiment.load_request_file(request_path)
    if len(loaded) != len(configs) or {
        item["body"]["model"] for item in loaded
    } != {args.model}:
        raise ValueError(f"unexpected request contents: {request_path}")
    return request_path, run_path, configs


def validate_response(expected_ids: set[str], custom_id: str, text: str) -> None:
    if custom_id not in expected_ids:
        raise ValueError(f"unexpected distillation response ID: {custom_id}")
    distill_skill.validate_skill(text, GENERATION_NAME)


def write_skills(
    response_output: Path, run_path: Path, configs: dict[str, dict]
) -> None:
    expected_by_id = {
        config["custom_id"]: (condition, config)
        for condition, config in configs.items()
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
            raise ValueError(
                f"duplicate custom_id at {response_output}:{line_number}: {custom_id}"
            )
        items[custom_id] = item
    if set(items) != set(expected_by_id):
        raise ValueError("distillation output IDs differ from the conditions")

    generated = {}
    for custom_id, (condition, config) in expected_by_id.items():
        text, usage = experiment.extract_response_text(items[custom_id])
        text = distill_skill.materialize_skill_name(text, config["skill_name"])
        generated[condition] = {"text": text, "usage": usage}

    run = json.loads(run_path.read_text(encoding="utf-8"))
    for condition, config in configs.items():
        output = config["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(generated[condition]["text"], encoding="utf-8")
        run["conditions"][condition]["usage"] = generated[condition]["usage"]
        print(f"wrote {condition} skill to {output}")
    experiment.write_json(run_path, run)


def distill(args: argparse.Namespace) -> None:
    """Distill the ordered and shuffled prefix skills and write them to skills/."""
    request_path, run_path, configs = prepare_distillation(args)
    response_output = args.run_dir / "distillation_output.jsonl"
    response_errors = args.run_dir / "distillation_errors.jsonl"
    expected_ids = {config["custom_id"] for config in configs.values()}
    experiment.run_concurrent_requests(
        request_path,
        response_output,
        response_errors,
        "Prefix skill distillation",
        args.api_provider,
        args.concurrency,
        args.max_attempts,
        args.confirm_submit,
        lambda custom_id, text: validate_response(expected_ids, custom_id, text),
    )
    write_skills(response_output, run_path, configs)


# The setting.

def prepared_skill_body(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"distilled skill is missing: {path}; run `skills_experiment.py distill` first"
        )
    return build_prompt.skill_body(path) + "\n"


def prepare_setting(
    output: Path,
    source_setting_path: Path,
    methods: tuple[str, ...],
    skill_bodies: dict[str, str],
    cases: list[dict],
    sources: list[dict],
    selection: dict,
    replace: bool = False,
) -> None:
    if output.exists() and not replace:
        setting = experiment.load_setting(output)
        experiment.load_setting_assets(output, setting)
        if setting.get("source_setting") != experiment.recorded_path(source_setting_path):
            raise ValueError(f"{output} was prepared from a different source setting")
        if setting["methods"] != list(methods):
            raise ValueError(
                f"{output} was prepared with methods {setting['methods']}, "
                f"not {list(methods)}"
            )
        print(f"already prepared: {output} ({setting['case_count']} cases)")
        return

    source_setting = experiment.load_setting(source_setting_path)
    source_assets = experiment.load_setting_assets(
        source_setting_path, source_setting
    )
    skill_methods = [t for t in methods if t in experiment.SKILL_METHODS]
    if sorted(skill_bodies) != sorted(skill_methods):
        raise ValueError(f"expected one skill body for each of {skill_methods}")
    asset_text = {
        "prediction_instructions": source_assets["prediction_instructions"],
        "judge_instructions": source_assets["judge_instructions"],
        **{experiment.SKILL_METHODS[t]: skill_bodies[t] for t in skill_methods},
    }
    asset_filenames = {name: f"{name}.md" for name in asset_text}

    prepared_cases = copy.deepcopy(cases)

    output.parent.mkdir(parents=True, exist_ok=replace)
    asset_dir = output.parent / "assets"
    asset_dir.mkdir(exist_ok=replace)
    for name, text in asset_text.items():
        (asset_dir / asset_filenames[name]).write_text(text, encoding="utf-8")

    setting = {
        "format_version": 1,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_setting": experiment.recorded_path(source_setting_path),
        "evaluation_indices": source_setting["evaluation_indices"],
        "methods": list(methods),
        "selection": selection,
        "case_count": len(prepared_cases),
        "prediction_request_count": len(prepared_cases) * len(methods),
        "judge_request_count": len(prepared_cases) * len(methods),
        "assets": {
            name: {"path": f"assets/{asset_filenames[name]}"} for name in asset_text
        },
        "sources": sources,
        "cases": prepared_cases,
    }
    experiment.write_json(output, setting)
    print(
        f"prepared {len(prepared_cases)} cases x {len(methods)} methods at {output}"
    )


def source_cases(source_setting: dict) -> list[dict]:
    cases = copy.deepcopy(source_setting["cases"])
    if {case.get("natural_prefix_length") for case in cases} != set(PREFIX_LENGTHS):
        raise ValueError("source setting is not a first-1/2/3 suite")
    return cases


def prepare(
    source_setting: Path = SOURCE_SETTING,
    output: Path = SETTING,
    ordered_skill: Path = ORDERED_SKILL,
    shuffled_skill: Path = SHUFFLED_SKILL,
    final_paper_skill: Path = FINAL_PAPER_SKILL,
    replace: bool = False,
) -> None:
    source = experiment.load_setting(source_setting)
    prepare_setting(
        output,
        source_setting,
        METHODS,
        {
            "skill": prepared_skill_body(ordered_skill),
            "shuffled_skills": prepared_skill_body(shuffled_skill),
            "final_paper_skill": prepared_skill_body(final_paper_skill),
        },
        source_cases(source),
        copy.deepcopy(source["sources"]),
        {
            **source["selection"],
            "experiment": "natural first-1/2/3 with matched short-prefix skills and the final paper skill",
            "skill_training_regime": "natural early cutoff-local prefix-to-target examples",
            "final_paper_skill_training_regime": (
                "final published paper text only from the same 19 training projects"
            ),
        },
        replace,
    )


def summarize(run_root: Path, repeats: int, output_dir: Path) -> None:
    """The four methods of the skills runs <run_root>-r1, -r2, ..."""
    setting = experiment.load_setting(SETTING)
    experiment.summarize_comparison(
        SETTING, setting, METHODS, ((run_root, METHODS),), CONTRASTS, repeats, output_dir
    )


# The final paper skill: distilled from the training projects' published papers.

FINAL_PAPER_DIR = HERE / "runs" / "final-paper-skill"
FINAL_PAPER_SOURCES = HERE / "skills" / "final-paper" / "papers.json"
FINAL_PAPER_PROMPT = HERE / "prompts" / "distill_skill_final_paper.md"
FINAL_PAPER_DISTILLATION_ID = "distill-final-artifact-skill"
FINAL_PAPER_SKILL_NAME = "final-paper"
# Distilled with the same Opus 5 settings as the prefix skills (DISTILL=1 in jobs/harness_table.sh).
FINAL_PAPER_DISTILLATION_SETTINGS = {
    "model": "anthropic/claude-opus-5",
    "store": False,
    "max_output_tokens": 12000,
    "reasoning": {"effort": "high"},
    "provider": experiment.provider_preferences("openrouter", 5.0, 25.0, allow_azure=False),
}
REPEATS = (1, 2, 3)
# Public OpenRouter list prices checked 2026-09-24, USD per million tokens.
# These are for estimation only; execution retains the recorded skills-run routing caps.
PRICE_SNAPSHOT = {
    "date": "2026-09-24",
    "prediction": {"input": 0.25, "output": 1.5, "source": "https://openrouter.ai/google/gemini-3.1-flash-lite"},
    "judge": {"input": 2.0, "output": 10.0, "source": "https://openrouter.ai/openai/gpt-5.6-sol"},
}


def write_once(path: Path, text: str) -> None:
    """Preserve immutable inputs and paid-call artifacts across resumptions."""
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"refusing to overwrite a different prepared artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def paper_sources(source_setting: dict) -> list[dict]:
    papers = json.loads(FINAL_PAPER_SOURCES.read_text(encoding="utf-8"))["papers"]
    indices = [p["training_idx"] for p in papers]
    if indices != source_setting["selection"]["training_indices_excluded"]:
        raise ValueError("paper list must contain exactly the prepared 19 training projects, in order")
    if len(set(indices)) != len(indices) or set(indices) & set(source_setting["evaluation_indices"]):
        raise ValueError("duplicate training papers or training/evaluation overlap")
    for paper in papers:
        if not paper["title"].strip() or not paper["pdf_url"].startswith("https://"):
            raise ValueError(f"malformed paper source: {paper['training_idx']}")
    return papers


def prepare_papers(work_dir: Path) -> dict:
    """Prepare complete published PDF text and record each paper's source."""
    source = experiment.load_setting(SETTING)
    papers = paper_sources(source)
    inputs = work_dir / "inputs"
    setting_path = inputs / "setting.json"
    if setting_path.exists():
        return json.loads(setting_path.read_text(encoding="utf-8"))

    inputs.mkdir(parents=True, exist_ok=True)
    records = []
    corpus = ["# Final published training papers"]
    for number, paper in enumerate(papers, 1):
        stem = str(paper["training_idx"])
        pdf = inputs / "papers" / (stem + ".pdf")
        pdf.parent.mkdir(parents=True, exist_ok=True)
        if not pdf.exists():
            request = urllib.request.Request(paper["pdf_url"], headers={"User-Agent": "research-final-paper-skill/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if not data.startswith(b"%PDF-"):
                raise ValueError(f"source is not a PDF: {paper['pdf_url']}")
            temporary = pdf.with_suffix(".pdf.tmp")
            temporary.write_bytes(data)
            temporary.replace(pdf)
        extraction = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(pdf), "-"],
            check=True, capture_output=True, text=True, encoding="utf-8",
        ).stdout
        normalize = lambda value: "".join(c for c in unicodedata.normalize("NFKD", value).casefold() if c.isalnum())
        if normalize(paper["title"]) not in normalize(extraction.split("\f")[0]):
            raise ValueError(f"PDF first page does not match the recorded title: {paper['title']}")
        text = extraction.replace("\f", "\n\n").strip() + "\n"
        if len(text) < 1000:
            raise ValueError(f"paper has insufficient extractable text: {pdf}")
        text_path = pdf.with_suffix(".txt")
        write_once(text_path, text)
        records.append({
            **paper, "pdf": pdf.relative_to(inputs).as_posix(),
            "text": text_path.relative_to(inputs).as_posix(),
            "pages": len(extraction.rstrip("\f\n").split("\f")), "characters": len(text),
        })
        corpus.append(f"## Training paper {number:03d}\n\n{text}")
        print(f"Prepared paper {number}/{len(papers)}: {paper['title']}", flush=True)
    corpus_path = inputs / "papers.md"
    write_once(corpus_path, "\n\n".join(corpus) + "\n")
    metadata = {
        "content": "Complete published PDF text, including embedded appendices; no truncation or trajectory annotations",
        "extraction": subprocess.run(["pdftotext", "-v"], capture_output=True, text=True, check=True).stderr.splitlines()[0],
        "papers": records,
        "corpus": {"path": "papers.md", "characters": len(corpus_path.read_text(encoding="utf-8"))},
    }
    write_once(setting_path, json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def reference_settings() -> dict:
    """Settings recorded by the skills-setting repeats, checked to agree."""
    settings = None
    for repeat in REPEATS:
        run_dir = Path(f"{SKILLS_RUN_ROOT}-r{repeat}")
        current = {}
        for phase, filename in (("prediction", "run.json"), ("judge", "judge_run.json")):
            run = json.loads((run_dir / filename).read_text(encoding="utf-8"))
            current[phase] = {key: run[f"{phase}_{key}"] for key in (
                "model", "reasoning_effort", "max_output_tokens", "api_provider", "provider_preferences",
            )}
        if settings is not None and current != settings:
            raise ValueError("skills-setting repeat settings differ")
        settings = current
    settings["distillation"] = dict(FINAL_PAPER_DISTILLATION_SETTINGS)
    return settings


def prepare_final_paper_distillation(work_dir: Path) -> Path:
    prepare_papers(work_dir)
    body = {
        **FINAL_PAPER_DISTILLATION_SETTINGS,
        "instructions": FINAL_PAPER_PROMPT.read_text(encoding="utf-8").strip(),
        "input": f"Required SKILL.md frontmatter name: {distill_skill.DISTILLATION_SKILL_NAME}\n\n"
        + (work_dir / "inputs/papers.md").read_text(encoding="utf-8").strip(),
    }
    request_path = work_dir / "distillation/requests.jsonl"
    experiment.write_or_validate_plan(request_path, request_path.parent / "run.json", [
        experiment.api_request(FINAL_PAPER_DISTILLATION_ID, body),
    ], {
        "settings": dict(FINAL_PAPER_DISTILLATION_SETTINGS),
    })
    return request_path


def normalize_distilled_skill(text: str) -> str:
    """Extract a labelled skill or remove an outer fence; keep its contents intact."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        if (
            lines[0] not in {"```", "```markdown", "```md", "```yaml"}
            or len(lines) < 3
            or lines[-1] != "```"
            or "```" in lines[1:-1]
        ):
            raise ValueError("expected one complete Markdown code fence around SKILL.md")
        text = "\n".join(lines[1:-1])
        lines = text.strip().splitlines()
    # The saved model response puts a paper title and '# SKILL.md' before the
    # complete artifact. Only an explicit, unique filename heading delimits it;
    # do not guess where a skill begins in arbitrary prose or add frontmatter.
    if lines and lines[0] != "---" and "# SKILL.md" in lines:
        if lines.count("# SKILL.md") != 1:
            raise ValueError("expected exactly one labelled SKILL.md artifact")
        text = "\n".join(lines[lines.index("# SKILL.md") + 1:])
    return distill_skill.validate_skill(text, distill_skill.DISTILLATION_SKILL_NAME)


def submit_final_paper_distillation(item: dict, output: Path) -> None:
    """Make one attempt and save the response before any content validation."""
    headers = {
        "Authorization": f"Bearer {experiment.api_key('openrouter')}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "trajectory-ideation-harness",
    }
    if os.environ.get("OPENROUTER_REFERER"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_REFERER"]
    request = urllib.request.Request(
        experiment.API_ROOTS["openrouter"] + item["url"],
        data=json.dumps(item["body"], ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST",
    )
    print("Final paper skill distillation: submitting one request (no automatic retries)", flush=True)
    try:
        response = urllib.request.urlopen(request, timeout=600)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        result = {
            "custom_id": item["custom_id"],
            "response": {
                "status_code": response.status,
                "request_id": response.headers.get("request-id"),
                "body": json.loads(response.read()),
            },
            "error": None,
        }
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"Saved distillation response and usage: {output}", flush=True)


def distill_final_paper(work_dir: Path, confirm_submit: bool) -> Path:
    requests = prepare_final_paper_distillation(work_dir)
    output = requests.parent / "output.jsonl"
    if not output.exists() or not output.read_text(encoding="utf-8").strip():
        if not confirm_submit:
            raise ValueError("1 paid API request remains; pass --confirm-submit to proceed")
        items = experiment.load_request_file(requests)
        if len(items) != 1 or items[0]["custom_id"] != FINAL_PAPER_DISTILLATION_ID:
            raise ValueError("expected exactly one final paper distillation request")
        submit_final_paper_distillation(items[0], output)
    responses = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(responses) != 1 or responses[0]["custom_id"] != FINAL_PAPER_DISTILLATION_ID:
        raise ValueError("expected exactly one final paper distillation response")
    try:
        text, usage = experiment.extract_response_text(responses[0])
        normalized = normalize_distilled_skill(text)
    except ValueError as exc:
        raise ValueError(
            f"Distillation response saved at {output}, but validation failed: {exc}. "
            "Inspect that saved response; rerunning reuses it without another paid request."
        ) from exc
    if normalized.strip() != text.strip():
        print(f"Extracted SKILL.md from response formatting; original retained at {output}", flush=True)
    skill_path = requests.parent / "SKILL.md"
    write_once(skill_path, distill_skill.materialize_skill_name(normalized, FINAL_PAPER_SKILL_NAME))
    write_once(requests.parent / "artifact.json", json.dumps({"usage": usage}, indent=2) + "\n")
    return skill_path


def install_final_paper_skill(skill_path: Path) -> None:
    """Make a newly distilled skill the final_paper_skill method of the skills setting."""
    FINAL_PAPER_SKILL.parent.mkdir(parents=True, exist_ok=True)
    FINAL_PAPER_SKILL.write_text(skill_path.read_text(encoding="utf-8"), encoding="utf-8")
    prepare(replace=True)


def estimate_final_paper(work_dir: Path) -> dict:
    """Estimate the new row using measured token usage from the matched-skill method."""
    config = reference_settings()
    phases = {}
    for phase in ("prediction", "judge"):
        usage = []
        for repeat in REPEATS:
            path = Path(f"{SKILLS_RUN_ROOT}-r{repeat}") / f"{phase}_output.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if item["custom_id"].endswith("__skill"):
                    _, record = experiment.extract_response_text(item)
                    usage.append(record)
        if len(usage) != 1206:
            raise ValueError(f"expected 1206 reference {phase} usages")
        caps = config[phase]["provider_preferences"]["max_price"]
        input_tokens = sum(u["input_tokens"] for u in usage)
        output_tokens = sum(u["output_tokens"] for u in usage)
        phases[phase] = {
            "calls": len(usage), "reference_input_tokens": input_tokens, "reference_output_tokens": output_tokens,
            "historical_successful_call_cost_usd": sum(u["cost"] for u in usage),
            "cost_at_price_snapshot_and_reference_tokens_usd": (input_tokens * PRICE_SNAPSHOT[phase]["input"] + output_tokens * PRICE_SNAPSHOT[phase]["output"]) / 1e6,
            "cost_at_routing_caps_and_reference_tokens_usd": (input_tokens * caps["prompt"] + output_tokens * caps["completion"]) / 1e6,
        }
    request_path = work_dir / "distillation/requests.jsonl"
    body = experiment.load_request_file(request_path)[0]["body"]
    characters = len(body["instructions"]) + len(body["input"])
    # Text extraction is not tokenization. Report a range rather than pretend it is exact.
    token_range = [round(characters / 4), round(characters / 2.5)]
    rates = config["distillation"]["provider"]["max_price"]
    distillation_cost = [(tokens * rates["prompt"] + body["max_output_tokens"] * rates["completion"]) / 1e6 for tokens in token_range]
    result = {
        "price_snapshot": PRICE_SNAPSHOT,
        "distillation": {"calls": 1, "estimated_input_tokens_range": token_range, "output_token_limit": body["max_output_tokens"], "cost_range_usd": distillation_cost},
        **phases,
        "total_using_historical_evaluation_billing_usd": [sum(p["historical_successful_call_cost_usd"] for p in phases.values()) + cost for cost in distillation_cost],
        "total_at_price_snapshot_and_reference_token_usage_usd": [sum(p["cost_at_price_snapshot_and_reference_tokens_usd"] for p in phases.values()) + cost for cost in distillation_cost],
        "total_at_routing_caps_and_reference_token_usage_usd": [sum(p["cost_at_routing_caps_and_reference_tokens_usd"] for p in phases.values()) + cost for cost in distillation_cost],
        "assumptions": "One distillation, 1206 predictions, 1206 judgments; no reruns of old methods. Evaluation token usage is estimated from the skills-setting runs, including reasoning. New skill length and output length can differ. Retries and credit-purchase fees are excluded; this is not a spending cap.",
    }
    experiment.write_json(work_dir / "cost_estimate.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="write the distillation examples")
    render_parser.add_argument("--output-dir", type=Path, default=DISTILLATION_DIR / "inputs")
    render_parser.add_argument(
        "--shuffle-metadata",
        type=Path,
        default=render_demonstrations.DEFAULT_SHUFFLE_METADATA,
    )

    distill_parser = subparsers.add_parser(
        "distill", help="distill the ordered and shuffled prefix skills (paid)"
    )
    distill_parser.add_argument("--run-dir", type=Path, default=DISTILLATION_DIR)
    distill_parser.add_argument("--model", required=True)
    distill_parser.add_argument("--max-output-tokens", type=int, default=8000)
    distill_parser.add_argument("--reasoning-effort")
    distill_parser.add_argument("--concurrency", type=int, default=2)
    distill_parser.add_argument("--max-attempts", type=int, default=5)
    distill_parser.add_argument("--confirm-submit", action="store_true")
    experiment.add_api_options(distill_parser)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--source-setting", type=Path, default=SOURCE_SETTING
    )
    prepare_parser.add_argument(
        "--output", type=Path, default=SETTING
    )
    prepare_parser.add_argument(
        "--ordered-skill", type=Path, default=ORDERED_SKILL
    )
    prepare_parser.add_argument(
        "--shuffled-skill", type=Path, default=SHUFFLED_SKILL
    )
    prepare_parser.add_argument(
        "--final-paper-skill", type=Path, default=FINAL_PAPER_SKILL
    )
    prepare_parser.add_argument(
        "--replace",
        action="store_true",
        help="re-prepare the existing setting from the current skill files",
    )
    summarize_parser = subparsers.add_parser(
        "summarize", help="summarize the four methods of the skills runs"
    )
    summarize_parser.add_argument("--run-root", type=Path, default=SKILLS_RUN_ROOT)
    summarize_parser.add_argument("--output-dir", type=Path, default=SUMMARY_DIR)
    summarize_parser.add_argument("--repeats", type=int, default=3)

    for command, help_text in (
        ("distill-final-paper", "distill the final paper skill from the papers and re-prepare the setting (paid)"),
        ("estimate-final-paper", "prepare the papers and the distillation request, and estimate costs"),
    ):
        final_parser = subparsers.add_parser(command, help=help_text)
        final_parser.add_argument("--work-dir", type=Path, default=FINAL_PAPER_DIR)
        if command == "distill-final-paper":
            final_parser.add_argument("--confirm-submit", action="store_true")

    args = parser.parse_args()
    if args.command == "distill-final-paper":
        install_final_paper_skill(distill_final_paper(args.work_dir, args.confirm_submit))
    elif args.command == "estimate-final-paper":
        prepare_final_paper_distillation(args.work_dir)
        estimate_final_paper(args.work_dir)
    elif args.command == "summarize":
        summarize(args.run_root, args.repeats, args.output_dir)
    elif args.command == "render":
        render_inputs(args.output_dir, args.shuffle_metadata)
    elif args.command == "distill":
        distill(args)
    else:
        prepare(
            args.source_setting,
            args.output,
            args.ordered_skill,
            args.shuffled_skill,
            args.final_paper_skill,
            args.replace,
        )


if __name__ == "__main__":
    main()
