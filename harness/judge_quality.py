#!/usr/bin/env python3
"""Judge quality: score variance across repeated judgments, and calibration against
controlled predictions.

`variance` summarizes repeated judgments of one fixed prediction set. `calibrate` has
Opus write exact, paraphrased, adjacent, and unrelated controls for each case and judges
them through OpenRouter.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

import experiment


# Variance across repeated judgments.

METRICS = experiment.REPORTED_METRICS


def load_scores(run_dir: Path) -> dict[tuple[str, str], dict]:
    path = run_dir / "results" / "scores.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = {}
    for raw in raw_rows:
        key = (raw["case_id"], raw["method"])
        if key in rows:
            raise ValueError(f"duplicate score {key} in {path}")
        rows[key] = {
            **raw,
            "evaluation_idx": int(raw["evaluation_idx"]),
            **{metric: int(raw[metric]) for metric in METRICS},
        }
    return rows


def load_summary(run_dir: Path) -> dict:
    return json.loads(
        (run_dir / "results" / "summary.json").read_text(encoding="utf-8")
    )


def billed_cost(output_path: Path) -> float | None:
    """OpenRouter's billed cost of a judge output; None when no response records a cost."""
    costs = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            response = item.get("response") or {}
            costs.append(((response.get("body") or {}).get("usage") or {}).get("cost"))
    if all(cost is None for cost in costs):
        return None
    if any(cost is None for cost in costs):
        raise ValueError(f"missing OpenRouter billed cost in {output_path}")
    total = 0.0
    for cost in costs:
        total += float(cost)
    return total


def repeat_dirs(run_root: Path, repeats: int) -> list[Path]:
    if repeats < 2:
        raise ValueError("judge reliability requires at least two repeats")
    return [Path(f"{run_root}-r{repeat}") for repeat in range(1, repeats + 1)]


def fleiss_kappa(score_sets: list[dict], keys: list[tuple[str, str]], metric: str) -> float:
    categories = range(5) if metric == experiment.HEADLINE_METRIC else range(3)
    counts_by_item = [
        Counter(scores[key][metric] for scores in score_sets) for key in keys
    ]
    ratings_per_item = len(score_sets)
    observed = statistics.fmean(
        (
            sum(counts[category] ** 2 for category in categories)
            - ratings_per_item
        )
        / (ratings_per_item * (ratings_per_item - 1))
        for counts in counts_by_item
    )
    category_frequencies = {
        category: sum(counts[category] for counts in counts_by_item)
        / (len(keys) * ratings_per_item)
        for category in categories
    }
    expected = sum(value**2 for value in category_frequencies.values())
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def reliability_rows(
    score_sets: list[dict], methods: list[str]
) -> list[dict]:
    rows = []
    for scope in ("all", *methods):
        keys = sorted(
            key
            for key in score_sets[0]
            if scope == "all" or key[1] == scope
        )
        for metric in METRICS:
            exact = statistics.fmean(
                len({scores[key][metric] for scores in score_sets}) == 1
                for key in keys
            )
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "item_count": len(keys),
                    "fleiss_kappa": fleiss_kappa(score_sets, keys, metric),
                    "all_judges_exact_agreement": exact,
                }
            )
    return rows


def analyze_variance(args: argparse.Namespace) -> None:
    directories = repeat_dirs(args.run_root, args.repeats)
    summaries = [load_summary(directory) for directory in directories]
    scores = [load_scores(directory) for directory in directories]
    expected_keys = set(scores[0])
    if any(set(run) != expected_keys for run in scores[1:]):
        raise ValueError("judge repetitions do not contain the same predictions")
    if {method for _, method in expected_keys} != {args.method}:
        raise ValueError(
            f"variance runs do not contain only method {args.method!r}"
        )
    for field in ("prediction_model", "judge_model", "judge_reasoning_effort"):
        if len({experiment.canonical_json(summary.get(field)) for summary in summaries}) != 1:
            raise ValueError(f"judge variance runs disagree on {field}")

    keys = sorted(expected_keys)
    repeat_means = []
    print("repeat\ttrajectory_alignment\tcomponent_match\toperation_match\tspecification_match")
    for repeat, run_scores in enumerate(scores, 1):
        means = {
            metric: statistics.fmean(run_scores[key][metric] for key in keys)
            for metric in METRICS
        }
        repeat_means.append(means)
        print(
            f"{repeat}\t"
            + "\t".join(f"{means[metric]:.3f}" for metric in METRICS)
        )

    reliability = reliability_rows(scores, [args.method])
    overall = {row["metric"]: row for row in reliability if row["scope"] == "all"}
    print("\nmetric\tmean\tsample_sd\tfleiss_kappa\tall_judges_exact_agreement")
    metric_summary = {}
    for metric in METRICS:
        values = [means[metric] for means in repeat_means]
        metric_summary[metric] = {
            "mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values),
            "fleiss_kappa": overall[metric]["fleiss_kappa"],
            "all_judges_exact_agreement": overall[metric][
                "all_judges_exact_agreement"
            ],
        }
        row = metric_summary[metric]
        print(
            f"{metric}\t{row['mean']:.3f}\t{row['sample_sd']:.3f}\t"
            f"{row['fleiss_kappa']:.3f}\t"
            f"{row['all_judges_exact_agreement']:.1%}"
        )
    costs = [billed_cost(directory / "judge_output.jsonl") for directory in directories]
    total_cost = None if None in costs else sum(costs)
    if total_cost is None:
        print("billed judge cost: not recorded")
    else:
        print(
            "billed judge cost by repeat: "
            + ", ".join(f"${cost:.4f}" for cost in costs)
            + f"; total=${total_cost:.4f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment.write_csv(args.output_dir / "reliability.csv", reliability)
    experiment.write_json(
        args.output_dir / "summary.json",
        {
            "run_root": experiment.recorded_path(args.run_root),
            "repeats": args.repeats,
            "item_count": len(keys),
            "method": args.method,
            "prediction_model": summaries[0]["prediction_model"],
            "judge_model": summaries[0]["judge_model"],
            "judge_reasoning_effort": summaries[0]["judge_reasoning_effort"],
            "billed_judge_cost_by_repeat_usd": costs,
            "total_billed_judge_cost_usd": total_cost,
            "metrics": metric_summary,
            "reliability_table": "reliability.csv",
        },
    )


# Calibration controls.

CONTROL_NAMES = (
    "exact_actual",
    "faithful_paraphrase",
    "adjacent",
    "unrelated",
)
EXPECTED_SCORES = {
    "exact_actual": {"component_match": 2, "operation_match": 2},
    "faithful_paraphrase": {"component_match": 2, "operation_match": 2},
    "adjacent": {"component_match": 1, "operation_match": 1},
    "unrelated": {"component_match": 0, "operation_match": 0},
}
CONTROL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(CONTROL_NAMES),
    "properties": {
        name: experiment.PREDICTION_SCHEMA for name in CONTROL_NAMES
    },
}
GENERATOR_INSTRUCTIONS = """You create controlled candidate predictions for calibrating a research-decision judge. You receive an observable project trajectory and its actual next annotated decision.

Return four prediction objects:

1. `exact_actual`: Copy the actual decision sentence verbatim. Infer compatible category, object, and operation fields.
2. `faithful_paraphrase`: Express the same decision in genuinely different wording while preserving its component, scientific operation, and all defining concrete settings.
3. `adjacent`: Write a plausible decision in the same broad research area, but deliberately act on a different component and use a related but distinct scientific operation. It should merit component_match=1 and operation_match=1, not an exact match.
4. `unrelated`: Write a plausible decision that acts on a clearly different part of the project and performs a clearly different scientific operation. It should merit component_match=0 and operation_match=0.

Every decision sentence must agree with its category, object, and operation fields. Use only the allowed enum values in the schema. Do not explain the controls and do not mention their intended scores in their decision text."""


def generator_custom_id(case_id: str) -> str:
    return f"calibration-controls__{case_id}"


def judge_custom_id(case_id: str, control: str) -> str:
    return f"judge-calibration__{case_id}__{control}"


def generator_input(case: dict) -> str:
    evaluation_input = case["evaluation_input"]
    if not evaluation_input.endswith(experiment.PREDICTION_SUFFIX):
        raise ValueError(f"{case['case_id']} has an unexpected evaluation input ending")
    observable_prefix = evaluation_input[: -len(experiment.PREDICTION_SUFFIX)]
    return "\n\n".join(
        [
            observable_prefix,
            "# Actual next annotated decision",
            f"Category: {case['target']['category']}",
            f"Decision: {case['target']['decision']}",
            "Generate the four calibration controls. Return the required JSON only.",
        ]
    )


def validate_controls(custom_id: str, text: str, cases_by_request: dict) -> None:
    controls = json.loads(text)
    if not isinstance(controls, dict) or set(controls) != set(CONTROL_NAMES):
        raise ValueError(f"{custom_id} has unexpected calibration controls")
    for name in CONTROL_NAMES:
        experiment.validate_prediction(f"{custom_id}__{name}", controls[name])

    target = cases_by_request[custom_id]["target"]
    if controls["faithful_paraphrase"]["decision"] == target["decision"]:
        raise ValueError(f"{custom_id} faithful_paraphrase is not a paraphrase")
    for name in ("adjacent", "unrelated"):
        if controls[name]["decision"] == target["decision"]:
            raise ValueError(f"{custom_id} {name} copies the target")


def pin_exact_control(controls: dict, target: dict) -> None:
    controls["exact_actual"]["category"] = target["category"]
    controls["exact_actual"]["decision"] = target["decision"]


def write_fixed(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"existing file differs from this run configuration: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def jsonl(items: list[dict]) -> str:
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)


def score_controls(setting: dict, judgments: dict) -> list[dict]:
    """One row per case and control: the judged scores next to the intended ones."""
    score_rows = []
    for case in setting["cases"]:
        for control in CONTROL_NAMES:
            custom_id = judge_custom_id(case["case_id"], control)
            judgment = judgments[custom_id]
            experiment.validate_judgment(custom_id, judgment)
            expected = EXPECTED_SCORES[control]
            score_rows.append(
                {
                    "case_id": case["case_id"],
                    "control": control,
                    "expected_trajectory_alignment": sum(expected.values()),
                    "trajectory_alignment": experiment.headline_score(judgment),
                    "expected_component_match": expected["component_match"],
                    "component_match": judgment["component_match"],
                    "expected_operation_match": expected["operation_match"],
                    "operation_match": judgment["operation_match"],
                    "specification_match": judgment["specification_match"],
                    "expected_scores_matched": (
                        judgment["component_match"] == expected["component_match"]
                        and judgment["operation_match"]
                        == expected["operation_match"]
                    ),
                    "justification": judgment["justification"],
                }
            )
    return score_rows


def summarize_controls(
    setting: dict, score_rows: list[dict], results_dir: Path
) -> tuple[dict, float]:
    """Write scores.csv; return per-control means and the strict expected-order rate."""
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    per_control = {}
    for control in CONTROL_NAMES:
        rows = [row for row in score_rows if row["control"] == control]
        per_control[control] = {
            "case_count": len(rows),
            "expected": {
                "trajectory_alignment": rows[0]["expected_trajectory_alignment"],
                "component_match": rows[0]["expected_component_match"],
                "operation_match": rows[0]["expected_operation_match"],
            },
            "mean": {
                metric: statistics.fmean(row[metric] for row in rows)
                for metric in experiment.REPORTED_METRICS
            },
            "exact_expected_rate": statistics.fmean(
                row["expected_scores_matched"] for row in rows
            ),
            "trajectory_mean_absolute_error": statistics.fmean(
                abs(
                    row["trajectory_alignment"]
                    - row["expected_trajectory_alignment"]
                )
                for row in rows
            ),
        }

    rows_by_case = {
        case["case_id"]: {
            row["control"]: row
            for row in score_rows
            if row["case_id"] == case["case_id"]
        }
        for case in setting["cases"]
    }
    strict_order_rate = statistics.fmean(
        min(
            rows["exact_actual"]["trajectory_alignment"],
            rows["faithful_paraphrase"]["trajectory_alignment"],
        )
        > rows["adjacent"]["trajectory_alignment"]
        > rows["unrelated"]["trajectory_alignment"]
        for rows in rows_by_case.values()
    )
    return per_control, strict_order_rate


def print_controls(per_control: dict, strict_order_rate: float) -> None:
    print(
        "control\texpected_trajectory\tmean_trajectory\tmean_object"
        "\tmean_operation\tmean_specification\texact_expected_rate\ttrajectory_mae"
    )
    for control in CONTROL_NAMES:
        row = per_control[control]
        print(
            "\t".join(
                [
                    control,
                    str(row["expected"]["trajectory_alignment"]),
                    f'{row["mean"]["trajectory_alignment"]:.3f}',
                    f'{row["mean"]["component_match"]:.3f}',
                    f'{row["mean"]["operation_match"]:.3f}',
                    f'{row["mean"]["specification_match"]:.3f}',
                    f'{row["exact_expected_rate"]:.1%}',
                    f'{row["trajectory_mean_absolute_error"]:.3f}',
                ]
            )
        )
    print(f"strict expected order rate: {strict_order_rate:.1%}")


def calibrate(args: argparse.Namespace) -> None:
    """Generate the controls with Opus and judge them through OpenRouter."""
    setting = experiment.load_setting(args.setting)
    if tuple(setting["methods"]) != ("baseline",):
        raise ValueError("calibration requires the prepared baseline-only setting")
    assets = experiment.load_setting_assets(args.setting, setting)
    cases_by_request = {
        generator_custom_id(case["case_id"]): case for case in setting["cases"]
    }
    generator_routing = experiment.provider_preferences(
        "openrouter",
        args.generator_max_prompt_price,
        args.generator_max_completion_price,
        False,
    )
    judge_routing = experiment.provider_preferences(
        "openrouter",
        args.judge_max_prompt_price,
        args.judge_max_completion_price,
        False,
    )

    generator_requests = [
        experiment.api_request(
            generator_custom_id(case["case_id"]),
            experiment.response_body(
                args.generator_model,
                GENERATOR_INSTRUCTIONS,
                generator_input(case),
                "research_decision_judge_calibration_controls",
                CONTROL_SCHEMA,
                args.generator_max_output_tokens,
                args.generator_reasoning_effort,
                generator_routing,
            ),
        )
        for case in setting["cases"]
    ]
    generator_request_path = args.run_dir / "control_requests.jsonl"
    generator_output_path = args.run_dir / "control_output.jsonl"
    write_fixed(generator_request_path, jsonl(generator_requests))
    experiment.run_concurrent_requests(
        generator_request_path,
        generator_output_path,
        args.run_dir / "control_errors.jsonl",
        "Calibration controls",
        "openrouter",
        args.generator_concurrency,
        args.max_attempts,
        args.confirm_submit,
        lambda custom_id, text: validate_controls(
            custom_id, text, cases_by_request
        ),
    )

    control_ids = set(cases_by_request)
    controls, generator_usage = experiment.read_response_output(
        generator_output_path, control_ids
    )
    for custom_id, generated in controls.items():
        validate_controls(
            custom_id,
            json.dumps(generated, ensure_ascii=False),
            cases_by_request,
        )
    for custom_id, generated in controls.items():
        target = cases_by_request[custom_id]["target"]
        pin_exact_control(generated, target)

    control_rows = [
        {
            "case_id": case["case_id"],
            "control": control,
            "actual_category": case["target"]["category"],
            "actual_decision": case["target"]["decision"],
            "prediction": controls[generator_custom_id(case["case_id"])][control],
        }
        for case in setting["cases"]
        for control in CONTROL_NAMES
    ]
    write_fixed(args.run_dir / "controls.jsonl", jsonl(control_rows))

    judge_instructions = assets["judge_instructions"].strip()
    judge_requests = []
    for case in setting["cases"]:
        generated = controls[generator_custom_id(case["case_id"])]
        for control in CONTROL_NAMES:
            judge_requests.append(
                experiment.judge_request(
                    judge_custom_id(case["case_id"], control),
                    judge_instructions,
                    case,
                    generated[control],
                    args.judge_model,
                    args.judge_max_output_tokens,
                    args.judge_reasoning_effort,
                    judge_routing,
                )
            )
    judge_request_path = args.run_dir / "judge_requests.jsonl"
    judge_output_path = args.run_dir / "judge_output.jsonl"
    write_fixed(judge_request_path, jsonl(judge_requests))
    experiment.run_concurrent_requests(
        judge_request_path,
        judge_output_path,
        args.run_dir / "judge_errors.jsonl",
        "Calibration judgments",
        "openrouter",
        args.judge_concurrency,
        args.max_attempts,
        args.confirm_submit,
        experiment.validate_judgment_response,
    )

    expected_judge_ids = {
        judge_custom_id(case["case_id"], control)
        for case in setting["cases"]
        for control in CONTROL_NAMES
    }
    judgments, judge_usage = experiment.read_response_output(
        judge_output_path, expected_judge_ids
    )
    score_rows = score_controls(setting, judgments)
    results_dir = args.run_dir / "results"
    per_control, strict_order_rate = summarize_controls(setting, score_rows, results_dir)
    summary = {
        "case_count": len(setting["cases"]),
        "control_count": len(score_rows),
        "generator_model": args.generator_model,
        "generator_reasoning_effort": args.generator_reasoning_effort,
        "judge_model": args.judge_model,
        "judge_reasoning_effort": args.judge_reasoning_effort,
        "per_control": per_control,
        "strict_expected_order_rate": strict_order_rate,
        "generator_usage": generator_usage,
        "judge_usage": judge_usage,
    }
    experiment.write_json(results_dir / "summary.json", summary)

    print_controls(per_control, strict_order_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    variance_parser = subparsers.add_parser(
        "variance", help="summarize repeated judgments of one fixed prediction set"
    )
    variance_parser.add_argument("--run-root", type=Path, required=True)
    variance_parser.add_argument("--repeats", type=int, default=3)
    variance_parser.add_argument("--method", default="baseline")
    variance_parser.add_argument("--output-dir", type=Path, required=True)

    calibrate_parser = subparsers.add_parser(
        "calibrate", help="generate controls with Opus and judge them through OpenRouter"
    )
    calibrate_parser.add_argument("--setting", type=Path, required=True)
    calibrate_parser.add_argument("--run-dir", type=Path, required=True)
    calibrate_parser.add_argument("--generator-model", required=True)
    calibrate_parser.add_argument("--judge-model", required=True)
    calibrate_parser.add_argument("--generator-reasoning-effort", default="high")
    calibrate_parser.add_argument("--judge-reasoning-effort", default="xhigh")
    calibrate_parser.add_argument("--generator-max-output-tokens", type=int, default=4000)
    calibrate_parser.add_argument("--judge-max-output-tokens", type=int, default=3000)
    calibrate_parser.add_argument("--generator-concurrency", type=int, default=4)
    calibrate_parser.add_argument("--judge-concurrency", type=int, default=16)
    calibrate_parser.add_argument("--max-attempts", type=int, default=5)
    calibrate_parser.add_argument("--generator-max-prompt-price", type=float, default=5)
    calibrate_parser.add_argument("--generator-max-completion-price", type=float, default=25)
    calibrate_parser.add_argument("--judge-max-prompt-price", type=float, default=10)
    calibrate_parser.add_argument("--judge-max-completion-price", type=float, default=50)
    calibrate_parser.add_argument("--confirm-submit", action="store_true")

    args = parser.parse_args()
    {"variance": analyze_variance, "calibrate": calibrate}[args.command](args)


if __name__ == "__main__":
    main()
