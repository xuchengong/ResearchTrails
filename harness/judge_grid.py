#!/usr/bin/env python3
"""Judge the reported predictions with Opus 5 under judge_rubrics_long.md.

Every Opus request is the saved Sol-medium request that produced a reported score,
with the rubric (instructions and response schema) and the judge (model, reasoning
effort, provider routing) replaced. The Sol cell holds those saved Sol judgments, so
the reported Sol scores are summarized next to the Opus scores without new requests.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
from pathlib import Path
import statistics

import experiment
import judge_quality
import rag_experiment as rag


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT_ROOT = HERE / "rejudging"
DEFAULT_OUTPUT = OUTPUT_ROOT / "judge-grid"
LONG_RUBRIC = HERE / "prompts" / "judge_rubrics_long.md"

REPEATS = (1, 2, 3)
HARNESS = (
    # (run family, setting loader, methods)
    ("skills", experiment.load_setting,
     ("baseline", "skill", "shuffled_skills", "final_paper_skill")),
    ("demos", experiment.load_setting, ("demonstrations",)),
    ("rag", rag.load_rag_setting, ("random_two_ordered", "retrieved_two_ordered")),
)
RL_METHODS = (
    # (reported method, saved judging directory relative to the repository, saved method)
    ("base_qwen3_8b", "sft/runs/qwen3-8b/judging/greedy/greedy-baseline",
     "greedy-baseline"),
    ("sft_initialization", "rl/runs/grpo-lr1e-4-eval-epoch7", "sft_initialization"),
    ("grpo_epoch7", "rl/runs/grpo-lr1e-4-eval-epoch7", "grpo"),
    ("grpo_epoch10", "rl/runs/grpo-lr1e-4-eval-epoch10", "grpo"),
)
CALIBRATION_RUN = "judge-calibration-opus5-sol-medium"
# The 100 GPT-5.6 Luna baseline predictions that Sol judged three times.
JUDGE_VARIANCE_RUN = "runs/judge-variance-baseline-r2-sol-medium-r1"
JUDGE_VARIANCE_OUTPUT = OUTPUT_ROOT / "judge-variance"
JUDGE_REPEATS = (1, 2, 3)
JUDGE_LABELS = {"sol-medium": "GPT-5.6 Sol", "opus5-medium": "Opus 5"}
LABELS = {
    **experiment.METHOD_LABELS,
    "base_qwen3_8b": "Qwen3-8B base",
    "sft_initialization": "SFT initialization",
    "grpo_epoch7": "GRPO epoch 7",
    "grpo_epoch10": "GRPO epoch 10",
}
PAIRED_REFERENCE = {"harness": "baseline", "rl": "sft_initialization"}

SOL_SETTINGS = ("openai/gpt-5.6-sol:floor", {"effort": "medium"}, 3000, (4, 20))
# None keeps the saved Sol-medium model, effort, and routing of each request.
JUDGES = {
    "sol-medium": None,
    "opus5-medium": {
        "model": "anthropic/claude-opus-5",
        "reasoning": {"effort": "medium"},
        "provider": experiment.provider_preferences("openrouter", 5, 25, False),
    },
}
LONG_JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [*experiment.JUDGE_METRICS, "justification"],
    "properties": {
        **{key: {"type": "integer", "minimum": 0, "maximum": 2} for key in experiment.JUDGE_METRICS},
        "justification": {
            "type": "string",
            "description": "Briefly justify each of the three scores using the actual and predicted decisions.",
        },
    },
}


def validate_long_judgment(custom_id: str, text: str) -> None:
    judgment = json.loads(text)
    if not isinstance(judgment, dict) or set(judgment) != {*experiment.JUDGE_METRICS, "justification"}:
        raise ValueError(f"{custom_id}: expected the three new metrics and justification")
    for metric in experiment.JUDGE_METRICS:
        if type(judgment[metric]) is not int or not 0 <= judgment[metric] <= 2:
            raise ValueError(f"{custom_id}: {metric} must be an integer from 0 to 2")
    if not isinstance(judgment["justification"], str) or not judgment["justification"].strip():
        raise ValueError(f"{custom_id}: justification must be nonempty")


VALIDATORS = {
    "judge_prediction": experiment.validate_judgment_response,
    "judge_rubrics_long": validate_long_judgment,
}

# USD per million input/output tokens. Sol's OpenRouter flex tier bills half its
# default tier and the saved runs contain both; Opus is Anthropic's list price.
SOL_PRICES = {"flex": (1.0, 5.0), "default": (2.0, 10.0)}
OPUS_PRICES = (5.0, 25.0)
OPUS_OUTPUT_MARGIN = 1.5  # Opus output may run up to this multiple of Sol's measured output


def cell_name(judge: str, rubric: str) -> str:
    return f"{judge}__{rubric}"


# Opus judges every reported prediction under judge_rubrics_long.md; the Sol cell holds
# the saved Sol judgments that produced the reported Sol scores.
CELLS = (cell_name("sol-medium", "judge_prediction"), cell_name("opus5-medium", "judge_rubrics_long"))
SUITES = {
    # suite: (default output directory, prepared cells)
    "main": (DEFAULT_OUTPUT, CELLS),
    "judge-variance": (JUDGE_VARIANCE_OUTPUT, (cell_name("opus5-medium", "judge_rubrics_long"),)),
}
# Main-grid outputs whose measured cost per judgment prices each extra suite: the Luna
# cases are the calibration cases.
PRICED_LIKE = {"judge-variance": "calibration.output.jsonl"}


def saved_requests(path: Path) -> dict[str, dict]:
    return {item["custom_id"]: item["body"] for item in experiment.load_request_file(path)}


def load_saved(directory: Path, cache: dict) -> tuple[dict, dict]:
    """Requests and responses of an earlier run, keyed by custom_id."""
    if directory not in cache:
        outputs = {}
        path = directory / "judge_output.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item["custom_id"] in outputs:
                    raise ValueError(f"duplicate response in {path}: {item['custom_id']}")
                outputs[item["custom_id"]] = item
        cache[directory] = (saved_requests(directory / "judge_requests.jsonl"), outputs)
    return cache[directory]


def item_record(item_id: str, part: str, method: str | None, repeat: int | None,
                case_id: str, prefix_length: int | None, control: str | None,
                source: Path, source_custom_id: str, bodies: dict) -> dict:
    return {
        "item_id": item_id, "part": part, "method": method, "repeat": repeat,
        "case_id": case_id, "prefix_length": prefix_length, "control": control,
        "source": experiment.recorded_path(source), "source_custom_id": source_custom_id,
        "body": bodies[source_custom_id],
    }


def load_items(cache: dict) -> list[dict]:
    """Every reported prediction with the saved Sol-medium request that scored it."""
    items = []
    for family, load_setting, methods in HARNESS:
        setting = load_setting(HERE / "experiments" / family / "setting.json")
        prefix = {case["case_id"]: case["natural_prefix_length"] for case in setting["cases"]}
        for repeat in REPEATS:
            source = HERE / "runs" / f"{family}-gemini-3.1-flash-lite-sol-medium-r{repeat}"
            bodies, _ = load_saved(source, cache)
            expected = {experiment.judge_custom_id(c, t) for c in prefix for t in methods}
            if set(bodies) != expected:
                raise ValueError(f"unexpected saved judge requests in {source}")
            items.extend(
                item_record(f"harness__r{repeat}__{case_id}__{method}", f"harness-r{repeat}",
                            method, repeat, case_id, prefix[case_id], None, source,
                            experiment.judge_custom_id(case_id, method), bodies)
                for case_id in prefix for method in methods
            )
    for method, directory, saved_method in RL_METHODS:
        source = ROOT / directory
        bodies, _ = load_saved(source, cache)
        with (source / "scores.csv").open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["method"] == saved_method]
        items.extend(
            item_record(f"rl__{method}__{row['case_id']}", "rl", method, None, row["case_id"],
                        int(row["target_step"]), None, source, row["custom_id"], bodies)
            for row in rows
        )
    source = HERE / "runs" / CALIBRATION_RUN
    bodies, _ = load_saved(source, cache)
    for custom_id in bodies:
        _, case_id, control = custom_id.split("__")
        items.append(item_record(f"calibration__{case_id}__{control}", "calibration", None,
                                 None, case_id, None, control, source, custom_id, bodies))
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("duplicate grid items")
    return items


def load_judge_variance_items(cache: dict) -> list[dict]:
    """The Luna baseline predictions once per judge repeat, with their saved Sol-medium requests."""
    source = HERE / JUDGE_VARIANCE_RUN
    bodies, _ = load_saved(source, cache)
    return [
        item_record(f"variance__j{repeat}__{custom_id.split('__')[1]}", f"judge-r{repeat}", None,
                    None, custom_id.split("__")[1], None, None, source, custom_id, bodies)
        for repeat in JUDGE_REPEATS for custom_id in bodies
    ]


def check_saved_sol(items: list[dict]) -> None:
    rubrics = {}
    for item in items:
        rubrics.setdefault(item["part"], set()).add(item["body"]["instructions"])
    if any(len(texts) != 1 for texts in rubrics.values()):
        raise ValueError("saved requests of one part do not share one rubric")
    for item in items:
        body = item["body"]
        price = body["provider"]["max_price"]
        settings = (body["model"], body["reasoning"], body["max_output_tokens"],
                    (price["prompt"], price["completion"]))
        if settings != SOL_SETTINGS or body["text"]["format"]["schema"] != experiment.JUDGE_SCHEMA:
            raise ValueError(f"{item['item_id']} was not judged by Sol medium under the original schema")


def cell_body(body: dict, judge: str, rubric: str, long_rubric: str) -> dict:
    body = dict(body)
    if rubric == "judge_rubrics_long":
        body["instructions"] = long_rubric
        body["text"] = {"format": {"type": "json_schema", "name": "decision_similarity",
                                   "strict": True, "schema": LONG_JUDGE_SCHEMA}}
    body.update(JUDGES[judge] or {})
    return body


def prior_location(judge: str, rubric: str, item: dict) -> tuple[Path, str] | None:
    """The saved Sol run and custom_id that already answer the Sol cell's request."""
    if (judge, rubric) == ("sol-medium", "judge_prediction"):
        return experiment.resolve_recorded(item["source"]), item["source_custom_id"]
    return None


def build_grid(cache: dict, suite: str = "main") -> tuple[list[dict], dict[str, list[dict]]]:
    loaders = {"main": load_items, "judge-variance": load_judge_variance_items}
    items = loaders[suite](cache)
    check_saved_sol(items)
    long_rubric = LONG_RUBRIC.read_text(encoding="utf-8")
    cells = {}
    for name in SUITES[suite][1]:
        judge, rubric = name.split("__")
        entries = []
        for item in items:
            body = cell_body(item["body"], judge, rubric, long_rubric)
            prior = None
            location = prior_location(judge, rubric, item)
            if location:
                directory, custom_id = location
                requests, outputs = load_saved(directory, cache)
                if custom_id in requests:
                    if experiment.canonical_json(requests[custom_id]) != experiment.canonical_json(body):
                        raise ValueError(f"{directory} answered a different request for {item['item_id']}")
                    if custom_id in outputs:
                        text, _ = experiment.extract_response_text(outputs[custom_id])
                        VALIDATORS[rubric](custom_id, text)
                        prior = {"path": experiment.recorded_path(directory / "judge_output.jsonl"), "custom_id": custom_id}
            entries.append({"item": item, "body": body, "prior": prior})
        cells[name] = entries
    return items, cells


def parts_of(entries: list[dict]) -> list[str]:
    return list(dict.fromkeys(entry["item"]["part"] for entry in entries))


def estimate(cells: dict[str, list[dict]], cache: dict) -> dict[str, dict[str, tuple[float, float]]]:
    """Low/high USD of each cell's new requests, from token counts of the saved Sol judgments.

    A new request is assumed to read the saved request's tokens plus its longer rubric
    (about four characters per token) and, for Opus, to write up to OPUS_OUTPUT_MARGIN
    times Sol's output.
    """
    saved = {}
    for entry in cells[cell_name("sol-medium", "judge_prediction")]:
        if entry["prior"]:
            directory = experiment.resolve_recorded(entry["prior"]["path"]).parent
            saved[entry["item"]["item_id"]] = cache[directory][1][entry["prior"]["custom_id"]]["response"]["body"]["usage"]
    fallback = {key: statistics.fmean(usage[key] for usage in saved.values())
                for key in ("input_tokens", "output_tokens")}

    costs = {}
    for name, entries in cells.items():
        judge = name.split("__")[0]
        costs[name] = {}
        for part in parts_of(entries):
            new = [e for e in entries if e["item"]["part"] == part and not e["prior"]]
            input_tokens = sum(
                saved.get(e["item"]["item_id"], fallback)["input_tokens"]
                + (len(e["body"]["instructions"]) - len(e["item"]["body"]["instructions"])) / 4
                for e in new)
            output_tokens = sum(saved.get(e["item"]["item_id"], fallback)["output_tokens"] for e in new)
            if judge == "sol-medium":
                low, high = (
                    (input_tokens * prices[0] + output_tokens * prices[1]) / 1e6
                    for prices in (SOL_PRICES["flex"], SOL_PRICES["default"]))
            else:
                low, high = (
                    (input_tokens * OPUS_PRICES[0] + output_tokens * margin * OPUS_PRICES[1]) / 1e6
                    for margin in (1.0, OPUS_OUTPUT_MARGIN))
            costs[name][part] = (low, high)
    return costs


def measured_costs(cells: dict[str, list[dict]], pattern: str) -> dict[str, dict[str, tuple[float, float]]]:
    """USD at the mean billed cost per judgment of matching outputs of the same cell in the main grid."""
    costs = {}
    for name, entries in cells.items():
        paths = sorted((SUITES["main"][0] / name).glob(pattern))
        per_judgment = statistics.fmean(
            json.loads(line)["response"]["body"]["usage"]["cost"]
            for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        costs[name] = {}
        for part in parts_of(entries):
            count = sum(1 for e in entries if e["item"]["part"] == part and not e["prior"])
            costs[name][part] = (count * per_judgment, count * per_judgment)
    return costs


def plan(suite: str = "main") -> None:
    """Print what prepare would request and its estimated cost, without writing anything."""
    cache = {}
    _, cells = build_grid(cache, suite)
    costs = estimate(cells, cache) if suite == "main" else measured_costs(cells, PRICED_LIKE[suite])
    print(f"{'cell':34} {'part':12} {'skipped':>8} {'new':>6} {'estimated USD':>16}")
    totals = {"all": [0.0, 0.0], "without harness-r2/r3": [0.0, 0.0]}
    for name, entries in cells.items():
        for part in parts_of(entries):
            selected = [e for e in entries if e["item"]["part"] == part]
            new = sum(1 for e in selected if not e["prior"])
            low, high = costs[name][part]
            print(f"{name:34} {part:12} {len(selected) - new:8d} {new:6d} {low:8.2f}-{high:7.2f}")
            for scope, total in totals.items():
                if scope == "all" or part not in ("harness-r2", "harness-r3"):
                    total[0] += low
                    total[1] += high
    for scope, (low, high) in totals.items():
        if scope == "all" or [low, high] != totals["all"]:
            print(f"Total, {scope}: ${low:.2f}-{high:.2f}")


@contextmanager
def locked_output(path: Path):
    """Confine writes to the rejudging tree and serialize preparation/submission/scoring."""
    output = path.resolve()
    if output == OUTPUT_ROOT or not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError(f"output must be a child directory of {OUTPUT_ROOT}")
    output.mkdir(parents=True, exist_ok=True)
    # A linked output file could otherwise overwrite an original run on resume.
    for child in output.iterdir():
        if child.is_symlink() or (child.is_file() and child.stat().st_nlink != 1):
            raise ValueError(f"linked files are not allowed in the output directory: {child}")
    with (output / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"another command is using {output}") from exc
        try:
            yield output
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare(output: Path, suite: str = "main") -> None:
    cache = {}
    _, cells = build_grid(cache, suite)
    for name, entries in cells.items():
        judge, rubric = name.split("__")
        with locked_output(output / name) as cell_dir:
            files = {}
            requests = {}
            for part in parts_of(entries):
                selected = [e for e in entries if e["item"]["part"] == part and not e["prior"]]
                if selected:
                    requests[part] = len(selected)
                    files[f"{part}.requests.jsonl"] = "".join(experiment.canonical_json(
                        experiment.api_request(e["item"]["item_id"], e["body"])
                    ) + "\n" for e in selected)
            records = []
            for entry in entries:
                item = {key: value for key, value in entry["item"].items() if key != "body"}
                result = entry["prior"] or {
                    "path": experiment.recorded_path(cell_dir / f"{item['part']}.output.jsonl"),
                    "custom_id": item["item_id"],
                }
                records.append({**item, "prior": entry["prior"] is not None, "result": result})
            files["items.json"] = json.dumps(records, indent=1, ensure_ascii=False) + "\n"
            metadata = {
                "judge": judge, "rubric": rubric,
                "judge_settings": JUDGES[judge] or "saved Sol-medium settings of each request",
                "item_count": len(entries), "requests": requests,
                "prior_count": sum(1 for e in entries if e["prior"]),
            }
            files["run.json"] = json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
            if not (cell_dir / "run.json").exists() and any(cell_dir.glob("*.output.jsonl")):
                raise ValueError(f"unidentified judge outputs exist in {cell_dir}")
            # Check every existing file before writing anything; preparation is immutable.
            for key, text in files.items():
                path = cell_dir / key
                if path.exists() and path.read_text(encoding="utf-8") != text:
                    raise ValueError(f"prepared inputs differ at {path}; choose a new output directory")
            for key, text in files.items():
                path = cell_dir / key
                if not path.exists():
                    with path.open("x", encoding="utf-8") as handle:
                        handle.write(text)
            print(f"{name}: {metadata['prior_count']} skipped, {requests or 'no'} new requests")


def load_prepared(cell_dir: Path) -> tuple[dict, list[dict]]:
    metadata = json.loads((cell_dir / "run.json").read_text(encoding="utf-8"))
    records = json.loads((cell_dir / "items.json").read_text(encoding="utf-8"))
    return metadata, records


def run(output: Path, cells: list[str], parts: list[str] | None, concurrency: int,
        max_attempts: int, confirm_submit: bool) -> None:
    for name in cells:
        with locked_output(output / name) as cell_dir:
            metadata, _ = load_prepared(cell_dir)
            for part in metadata["requests"]:
                if parts is None or part in parts:
                    experiment.run_concurrent_requests(
                        cell_dir / f"{part}.requests.jsonl", cell_dir / f"{part}.output.jsonl",
                        cell_dir / f"{part}.errors.jsonl", f"{name} {part}", "openrouter",
                        concurrency, max_attempts, confirm_submit, VALIDATORS[metadata["rubric"]],
                    )


def mean(rows: list[dict], metric: str) -> float:
    return statistics.fmean(row[metric] for row in rows)


def summarize(output: Path, parts: list[str] | None, suite: str = "main") -> None:
    cells = SUITES[suite][1]
    rows = []
    spent = {}
    for name in cells:
        cell_dir = output / name
        metadata, records = load_prepared(cell_dir)
        judge, rubric = metadata["judge"], metadata["rubric"]
        responses = {}
        for record in records:
            if parts is not None and record["part"] not in parts:
                continue
            path = experiment.resolve_recorded(record["result"]["path"])
            if path not in responses:
                responses[path] = {}
                if path.exists():
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            item = json.loads(line)
                            responses[path][item["custom_id"]] = item
            custom_id = record["result"]["custom_id"]
            if custom_id not in responses[path]:
                raise ValueError(f"{name} has no judgment for {record['item_id']}; run it or pass --parts")
            text, usage = experiment.extract_response_text(responses[path][custom_id])
            VALIDATORS[rubric](custom_id, text)
            judgment = json.loads(text)
            if not record["prior"]:
                spent[name] = spent.get(name, 0.0) + float(usage.get("cost", 0.0))
            component = judgment["component_match"]
            rows.append({
                "cell": name, "judge": judge, "rubric": rubric,
                **{key: record[key] for key in (
                    "item_id", "part", "method", "repeat", "case_id", "prefix_length", "control")},
                "prior": record["prior"], "component_match": component,
                "operation_match": judgment["operation_match"],
                "specification_match": judgment["specification_match"],
                "trajectory_alignment": component + judgment["operation_match"],
                "justification": judgment["justification"],
            })

    if suite == "judge-variance":
        summarize_reliability(output, rows, cells, parts, spent)
        return
    predictions = [row for row in rows if row["method"]]
    summary_rows = []
    for name in cells:
        for method in LABELS:
            for prefix in (1, 2, 3, "overall"):
                selected = [row for row in predictions if row["cell"] == name and row["method"] == method
                            and (prefix == "overall" or row["prefix_length"] == prefix)]
                if selected:
                    summary_rows.append({
                        "cell": name, "method": method, "prefix_length": prefix, "n": len(selected),
                        **{metric: mean(selected, metric) for metric in (
                            "trajectory_alignment", "component_match", "operation_match",
                            "specification_match")},
                    })

    means, deltas, calibration_rows = {}, {}, []
    for name in cells:
        cell_rows = [row for row in predictions if row["cell"] == name]
        by_key = {(row["method"], row["repeat"], row["case_id"]): row["trajectory_alignment"]
                  for row in cell_rows}
        for method in LABELS:
            selected = [row for row in cell_rows if row["method"] == method]
            if not selected:
                continue
            means[name, method] = mean(selected, "trajectory_alignment")
            reference = PAIRED_REFERENCE["rl" if selected[0]["part"] == "rl" else "harness"]
            if method != reference and any(row["method"] == reference for row in cell_rows):
                deltas[name, method] = statistics.fmean(
                    by_key[method, row["repeat"], row["case_id"]]
                    - by_key[reference, row["repeat"], row["case_id"]] for row in selected)
        controls = [row for row in rows if row["cell"] == name and row["control"]]
        if controls:
            by_case = {}
            for row in controls:
                by_case.setdefault(row["case_id"], {})[row["control"]] = row["trajectory_alignment"]
            summary = {"cell": name, "cases": len(by_case)}
            for control, expected in judge_quality.EXPECTED_SCORES.items():
                selected = [row for row in controls if row["control"] == control]
                summary[f"{control}_mean"] = mean(selected, "trajectory_alignment")
                summary[f"{control}_exact_rate"] = statistics.fmean(
                    (row["component_match"], row["operation_match"])
                    == (expected["component_match"], expected["operation_match"]) for row in selected)
            summary["strict_order_rate"] = statistics.fmean(
                min(s["exact_actual"], s["faithful_paraphrase"]) > s["adjacent"] > s["unrelated"]
                for s in by_case.values())
            calibration_rows.append(summary)

    header = "| Method | " + " | ".join(cells) + " |\n|---|" + "---:|" * len(cells) + "\n"
    table = "Mean component + operation (0-4).\n\n"
    table += header + "".join(
        f"| {LABELS[method]} | " + " | ".join(
            f"{means[name, method]:.3f}" if (name, method) in means else "" for name in cells) + " |\n"
        for method in LABELS if any((name, method) in means for name in cells))
    if deltas:
        table += "\nPaired difference from Baseline (harness) or SFT initialization (SFT/GRPO).\n\n"
        table += header + "".join(
            f"| {LABELS[method]} | " + " | ".join(
                f"{deltas[name, method]:+.3f}" if (name, method) in deltas else "" for name in cells) + " |\n"
            for method in LABELS if any((name, method) in deltas for name in cells))

    # LaTeX rows per harness method: mean and sample SD of the per-repeat means at each prefix.
    tex = []
    for name in cells:
        harness_rows = [row for row in predictions if row["cell"] == name and row["repeat"]]
        if harness_rows:
            tex.append(f"% {name}: mean $\\pm$ sample SD across repeat means")
        for method in LABELS:
            selected = [row for row in harness_rows if row["method"] == method]
            if not selected:
                continue
            values = []
            for prefix in (1, 2, 3):
                by_repeat = {}
                for row in selected:
                    if row["prefix_length"] == prefix:
                        by_repeat.setdefault(row["repeat"], []).append(row["trajectory_alignment"])
                repeat_means = [statistics.fmean(scores) for scores in by_repeat.values()]
                text = f"{statistics.fmean(repeat_means):.3f}"
                if len(repeat_means) > 1:
                    text += " $\\pm$ " + f"{statistics.stdev(repeat_means):.3f}".lstrip("0")
                values.append(text)
            tex.append(f"{LABELS[method]} & " + " & ".join(values) + " \\\\")
    if calibration_rows:
        table += "\nCalibration controls: mean score, and rate of intended component/operation scores.\n\n"
        table += "| Cell | Exact | Paraphrase | Adjacent | Unrelated | Adjacent exact rate | Strict order |\n"
        table += "|---|---:|---:|---:|---:|---:|---:|\n" + "".join(
            f"| {s['cell']} | {s['exact_actual_mean']:.2f} | {s['faithful_paraphrase_mean']:.2f} "
            f"| {s['adjacent_mean']:.2f} | {s['unrelated_mean']:.2f} | {s['adjacent_exact_rate']:.2f} "
            f"| {s['strict_order_rate']:.2f} |\n" for s in calibration_rows)

    with locked_output(output) as directory:
        for filename, table_rows in (("scores.csv", rows), ("summary.csv", summary_rows),
                                     ("calibration.csv", calibration_rows)):
            if table_rows:
                with (directory / filename).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
                    writer.writeheader()
                    writer.writerows(table_rows)
        (directory / "table.md").write_text(table, encoding="utf-8")
        (directory / "prefix_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        experiment.write_json(directory / "summary.json", {
            "parts": parts or "all", "judgments": len(rows),
            "new_judgment_cost_usd": spent, "rows": summary_rows, "calibration": calibration_rows,
        })
    print(table)


def summarize_reliability(output: Path, rows: list[dict], cells: tuple[str, ...],
                          parts: list[str] | None, spent: dict) -> None:
    """Sample SD of the per-repeat means and Fleiss' kappa across repeated judgments."""
    reliability, tex = [], []
    for name in cells:
        repeats = sorted({row["part"] for row in rows if row["cell"] == name})
        if len(repeats) < 2:
            raise ValueError(f"{name}: reliability needs at least two judge repeats, got {repeats}")
        score_sets = [{row["case_id"]: row for row in rows if row["cell"] == name and row["part"] == part}
                      for part in repeats]
        keys = sorted(score_sets[0])
        if any(sorted(scores) != keys for scores in score_sets):
            raise ValueError(f"{name}: judge repeats cover different predictions")
        judge, rubric = name.split("__")
        tex.append(f"% {name} ({rubric}.md): {len(repeats)} judgments of {len(keys)} predictions")
        for metric, label in (("trajectory_alignment", "\\texttt{Score}"),
                              ("component_match", "\\textit{Component}"),
                              ("operation_match", "\\textit{Operation}")):
            sd = statistics.stdev(statistics.fmean(scores[key][metric] for key in keys)
                                  for scores in score_sets)
            kappa = judge_quality.fleiss_kappa(score_sets, keys, metric)
            reliability.append({"cell": name, "metric": metric, "repeats": len(repeats),
                                "predictions": len(keys), "sample_sd": sd, "fleiss_kappa": kappa})
            first = f"\\multirow{{3}}{{*}}{{{JUDGE_LABELS[judge]}}}" if metric == "trajectory_alignment" else ""
            tex.append(f"{first} & {label} & {sd:.3f} & {kappa:.3f} \\\\")
    with locked_output(output) as directory:
        for filename, table_rows in (("scores.csv", rows), ("reliability.csv", reliability)):
            with (directory / filename).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
                writer.writeheader()
                writer.writerows(table_rows)
        (directory / "reliability_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        experiment.write_json(directory / "summary.json", {
            "parts": parts or "all", "judgments": len(rows),
            "new_judgment_cost_usd": spent, "reliability": reliability,
        })
    print("\n".join(tex))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "prepare", "run", "summarize"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--suite", choices=SUITES, default="main",
                               help="main grid; judge-variance (Opus x long rubric only)")
        if command != "plan":
            subparser.add_argument("--output-dir", type=Path,
                                   help="default: the suite's directory under rejudging/")
        if command in ("run", "summarize"):
            subparser.add_argument("--parts", nargs="+",
                                   help="e.g. calibration rl harness-r1 (default: all parts)")
        if command == "run":
            subparser.add_argument("--cells", nargs="+", choices=CELLS,
                                   help="default: every cell of the suite")
            subparser.add_argument("--concurrency", type=int, default=16)
            subparser.add_argument("--max-attempts", type=int, default=5)
            subparser.add_argument("--confirm-submit", action="store_true")
    args = parser.parse_args()
    output, cells = SUITES[args.suite]
    if args.command == "plan":
        plan(args.suite)
    elif args.command == "prepare":
        prepare(args.output_dir or output, args.suite)
    elif args.command == "run":
        run(args.output_dir or output, args.cells or list(cells), args.parts, args.concurrency,
            args.max_attempts, args.confirm_submit)
    else:
        summarize(args.output_dir or output, args.parts, args.suite)


if __name__ == "__main__":
    main()
