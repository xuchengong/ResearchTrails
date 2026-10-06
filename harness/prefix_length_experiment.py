#!/usr/bin/env python3
"""Evaluate next-decision prediction with no prefix, the last k observed decisions, or the full prefix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics

import experiment


HERE = Path(__file__).resolve().parent
DEFAULT_SETTING = (
    HERE / "experiments" / "prefix-length" / "setting.json"
)
DEFAULT_RUN_ROOT = (
    HERE
    / "runs"
    / "prefix-length-gemini-3.1-flash-lite-last-k-sol-medium"
)
PREFIX_LENGTHS = (1, 2, 3, 5, 10)
CONDITIONS = tuple(f"last_{length}" for length in PREFIX_LENGTHS)
ENDPOINT_CONDITIONS = ("no_prefix", "full")
ALL_CONDITIONS = ("no_prefix", *CONDITIONS, "full")
PREDICTION_SUFFIX = experiment.PREDICTION_SUFFIX
OBSERVED_MARKER = "Observed decisions:\n\n"
# The no-prefix point sends only this request after the prediction instructions.
NO_PREFIX_INPUT = "Predict the next research decision. Return the required JSON only."


def condition_name(length: int) -> str:
    return f"last_{length}"


def observed_blocks(case: dict) -> tuple[list[str], str]:
    evaluation_input = case["evaluation_input"]
    if not evaluation_input.endswith(PREDICTION_SUFFIX):
        raise ValueError(f"{case['case_id']} has an unexpected prediction suffix")
    observable = evaluation_input[: -len(PREDICTION_SUFFIX)]
    if observable.count(OBSERVED_MARKER) != 1:
        raise ValueError(f"{case['case_id']} has an unexpected observed-decision marker")
    _, body = observable.split(OBSERVED_MARKER)
    starts = list(re.finditer(r"(?m)^- T[0-9]{3} \|", body))
    blocks = [
        body[
            match.start() : starts[position + 1].start()
            if position + 1 < len(starts)
            else len(body)
        ].strip()
        for position, match in enumerate(starts)
    ]
    if len(blocks) != case["observed_decision_count"]:
        raise ValueError(
            f"{case['case_id']} renders {len(blocks)} decisions but declares "
            f"{case['observed_decision_count']}"
        )
    return blocks, evaluation_input


def recent_prefix(case: dict, length: int) -> str:
    if length <= 0:
        raise ValueError("prefix length must be positive")
    blocks, full_input = observed_blocks(case)
    if len(blocks) <= length:
        return full_input
    description = (
        f"Only the {length} most recent decisions introduced before the held-out "
        "decision are shown, in timezone-aware first-date order. Older observed "
        "decisions are omitted. Outcomes that were not established by the cutoff "
        "are withheld."
    )
    return "\n\n".join(
        [
            "# Evaluation trajectory prefix",
            description,
            "Observed decisions:",
            "\n\n".join(blocks[-length:]),
        ]
    ) + PREDICTION_SUFFIX


def submitted_keys(setting: dict) -> list[tuple[dict, int]]:
    return [
        (case, length)
        for case in setting["cases"]
        for length in PREFIX_LENGTHS
        if case["observed_decision_count"] > length
    ]


def load_submitted_predictions(setting: dict, output_path: Path) -> tuple[dict, dict]:
    return experiment.read_predictions(
        output_path,
        {
            experiment.prediction_custom_id(case["case_id"], condition_name(length))
            for case, length in submitted_keys(setting)
        },
    )


def prepare_predictions(args: argparse.Namespace) -> tuple[dict, Path]:
    if not args.model.strip():
        raise ValueError("prediction model must not be empty")
    if args.max_output_tokens <= 0:
        raise ValueError("prediction max output tokens must be positive")
    setting = experiment.load_setting(args.setting)
    assets = experiment.load_setting_assets(args.setting, setting)
    routing = experiment.routing_from_args(args)
    requests = [
        experiment.prediction_request(
            experiment.prediction_custom_id(case["case_id"], condition_name(length)),
            assets["prediction_instructions"].strip(),
            recent_prefix(case, length),
            args.model,
            args.max_output_tokens,
            args.reasoning_effort,
            routing,
        )
        for case, length in submitted_keys(setting)
    ]
    request_path = args.run_dir / "prediction_requests.jsonl"
    submitted_counts = {
        condition_name(length): sum(
            case["observed_decision_count"] > length
            for case in setting["cases"]
        )
        for length in PREFIX_LENGTHS
    }
    experiment.write_or_validate_plan(
        request_path,
        args.run_dir / "run.json",
        requests,
        {
            "format_version": 1,
            "setting": experiment.recorded_path(args.setting),
            "conditions": list(CONDITIONS),
            "prefix_lengths": list(PREFIX_LENGTHS),
            "prefix_selection": "most recent k observed decisions",
            "case_count": setting["case_count"],
            "submitted_prediction_counts": submitted_counts,
            "reused_full_prediction_count": (
                setting["case_count"] * len(PREFIX_LENGTHS) - len(requests)
            ),
            "prediction_model": args.model,
            "prediction_reasoning_effort": args.reasoning_effort,
            "prediction_max_output_tokens": args.max_output_tokens,
            "prediction_api_provider": args.api_provider,
            "prediction_provider_preferences": routing,
            "execution_mode": "concurrent",
            "stateless": True,
        },
    )
    return setting, request_path


def run_predictions(args: argparse.Namespace) -> None:
    setting, request_path = prepare_predictions(args)
    output_path = args.run_dir / "prediction_output.jsonl"
    experiment.run_requests(
        args, request_path, output_path, "Prefix-length predictions",
        experiment.validate_prediction_response,
    )
    predictions, _ = load_submitted_predictions(setting, output_path)
    print(f"validated {len(predictions)} new truncated-prefix predictions")


def prepare_no_prefix_predictions(args: argparse.Namespace) -> tuple[dict, Path]:
    if not args.model.strip():
        raise ValueError("prediction model must not be empty")
    if args.max_output_tokens <= 0:
        raise ValueError("prediction max output tokens must be positive")
    setting = experiment.load_setting(args.setting)
    assets = experiment.load_setting_assets(args.setting, setting)
    routing = experiment.routing_from_args(args)
    requests = [
        experiment.prediction_request(
            experiment.prediction_custom_id(case["case_id"], "no_prefix"),
            assets["prediction_instructions"].strip(),
            NO_PREFIX_INPUT,
            args.model,
            args.max_output_tokens,
            args.reasoning_effort,
            routing,
        )
        for case in setting["cases"]
    ]
    request_path = args.run_dir / "prediction_requests.jsonl"
    experiment.write_or_validate_plan(
        request_path,
        args.run_dir / "run.json",
        requests,
        {
            "format_version": 1,
            "setting": experiment.recorded_path(args.setting),
            "method": "no_prefix",
            "no_prefix_input": NO_PREFIX_INPUT,
            "prediction_model": args.model,
            "prediction_reasoning_effort": args.reasoning_effort,
            "prediction_max_output_tokens": args.max_output_tokens,
            "prediction_api_provider": args.api_provider,
            "prediction_provider_preferences": routing,
            "execution_mode": "concurrent",
            "stateless": True,
        },
    )
    return setting, request_path


def run_no_prefix_predictions(args: argparse.Namespace) -> None:
    setting, request_path = prepare_no_prefix_predictions(args)
    experiment.run_requests(
        args, request_path, args.run_dir / "prediction_output.jsonl", "No-prefix predictions",
        experiment.validate_prediction_response,
    )
    _, predictions = load_no_prefix_predictions(setting, args.run_dir)
    print(f"validated {len(predictions)} no-prefix predictions")


def load_no_prefix_predictions(
    setting: dict, run_dir: Path
) -> tuple[dict, dict[str, dict]]:
    prediction_run = experiment.read_json(run_dir / "run.json")
    if prediction_run.get("method") != "no_prefix":
        raise ValueError(f"expected a no-prefix prediction run: {run_dir}")
    predictions, _ = experiment.read_predictions(
        run_dir / "prediction_output.jsonl",
        {
            experiment.prediction_custom_id(case["case_id"], "no_prefix")
            for case in setting["cases"]
        },
    )
    return prediction_run, predictions


def load_full_prefix_predictions(
    setting: dict, run_dir: Path
) -> tuple[dict, dict[str, dict]]:
    """Baseline predictions of a runner run over the same cases, as the full point."""
    prediction_run = experiment.read_json(run_dir / "run.json")
    source_setting = experiment.read_json(experiment.resolve_recorded(prediction_run["setting"]))
    base_cases = {case["case_id"]: case for case in setting["cases"]}
    source_cases = {case["case_id"]: case for case in source_setting["cases"]}
    if set(base_cases) != set(source_cases):
        raise ValueError(f"full-prefix source case IDs differ: {run_dir}")
    for case_id, case in base_cases.items():
        source_case = source_cases[case_id]
        if (
            source_case["evaluation_input"] != case["evaluation_input"]
            or source_case["target"] != case["target"]
        ):
            raise ValueError(f"full-prefix source case differs for {case_id}: {run_dir}")
    expected_ids = {
        experiment.prediction_custom_id(case["case_id"], method)
        for case in source_setting["cases"]
        for method in source_setting["methods"]
    }
    all_predictions, _ = experiment.read_response_output(
        run_dir / "prediction_output.jsonl", expected_ids
    )
    predictions = {
        experiment.prediction_custom_id(case["case_id"], "full"): all_predictions[
            experiment.prediction_custom_id(case["case_id"], "baseline")
        ]
        for case in setting["cases"]
    }
    for custom_id, prediction in predictions.items():
        experiment.validate_prediction(custom_id, prediction)
    return prediction_run, predictions


def prepare_judging(args: argparse.Namespace) -> tuple[dict, Path]:
    if not args.model.strip():
        raise ValueError("judge model must not be empty")
    if args.max_output_tokens <= 0:
        raise ValueError("judge max output tokens must be positive")
    setting = experiment.load_setting(args.setting)
    assets = experiment.load_setting_assets(args.setting, setting)
    prediction_run = experiment.read_json(args.run_dir / "run.json")
    prediction_output = args.run_dir / "prediction_output.jsonl"
    if not prediction_output.is_file():
        raise FileNotFoundError(
            f"prediction output is missing: {prediction_output}; run predictions first"
        )
    predictions, prediction_usage = load_submitted_predictions(
        setting, prediction_output
    )
    no_prefix_run, no_prefix_predictions = load_no_prefix_predictions(
        setting, args.no_prefix_run_dir
    )
    full_run, full_predictions = load_full_prefix_predictions(
        setting, args.full_prefix_run_dir
    )
    for source_name, source_run in (
        ("no-prefix", no_prefix_run),
        ("full-prefix", full_run),
    ):
        for field in (
            "prediction_model",
            "prediction_reasoning_effort",
            "prediction_api_provider",
            "prediction_provider_preferences",
        ):
            if experiment.canonical_json(source_run.get(field)) != (
                experiment.canonical_json(prediction_run.get(field))
            ):
                raise ValueError(
                    f"ladder and {source_name} runs disagree on {field}"
                )

    routing = experiment.routing_from_args(args)
    judge_instructions = assets["judge_instructions"].strip()

    # Truncated predictions first, then both saved endpoints of every case; every
    # judgment sees the original full observable prefix.
    judged = [
        (case, condition_name(length), predictions)
        for case, length in submitted_keys(setting)
    ] + [
        (case, condition, endpoint_predictions)
        for case in setting["cases"]
        for condition, endpoint_predictions in (
            ("no_prefix", no_prefix_predictions),
            ("full", full_predictions),
        )
    ]
    requests = [
        experiment.judge_request(
            experiment.judge_custom_id(case["case_id"], condition),
            judge_instructions,
            case,
            source[experiment.prediction_custom_id(case["case_id"], condition)],
            args.model,
            args.max_output_tokens,
            args.reasoning_effort,
            routing,
        )
        for case, condition, source in judged
    ]
    request_path = args.run_dir / "judge_requests.jsonl"
    experiment.write_or_validate_plan(
        request_path,
        args.run_dir / "judge_run.json",
        requests,
        {
            "format_version": 1,
            "conditions": list(ALL_CONDITIONS),
            "prediction_output": experiment.recorded_path(prediction_output),
            "prediction_usage": prediction_usage,
            "no_prefix_source_run_dir": experiment.recorded_path(args.no_prefix_run_dir),
            "full_prefix_source_run_dir": experiment.recorded_path(args.full_prefix_run_dir),
            "judge_context": "original full observable prefix for every condition",
            "judge_model": args.model,
            "judge_reasoning_effort": args.reasoning_effort,
            "judge_max_output_tokens": args.max_output_tokens,
            "judge_api_provider": args.api_provider,
            "judge_provider_preferences": routing,
            "execution_mode": "concurrent",
            "stateless": True,
        },
    )
    return setting, request_path


def new_score_rows(setting: dict, judge_output: Path) -> tuple[dict, dict]:
    expected_ids = {
        experiment.judge_custom_id(case["case_id"], condition_name(length))
        for case, length in submitted_keys(setting)
    }
    expected_ids.update(
        experiment.judge_custom_id(case["case_id"], condition)
        for case in setting["cases"]
        for condition in ENDPOINT_CONDITIONS
    )
    judgments, usage = experiment.read_response_output(judge_output, expected_ids)
    rows = {}
    submitted = [
        (case, condition_name(length), length, "new_truncated_prefix_prediction")
        for case, length in submitted_keys(setting)
    ]
    endpoints = [
        (
            case,
            condition,
            0 if condition == "no_prefix" else case["observed_decision_count"],
            f"saved_{condition}_prediction_rejudged",
        )
        for case in setting["cases"]
        for condition in ENDPOINT_CONDITIONS
    ]
    for case, condition, displayed_count, score_source in (*submitted, *endpoints):
        custom_id = experiment.judge_custom_id(case["case_id"], condition)
        judgment = judgments[custom_id]
        experiment.validate_judgment(custom_id, judgment)
        rows[(case["case_id"], condition)] = experiment.score_row(
            case,
            condition,
            judgment,
            displayed_decision_count=displayed_count,
            score_source=score_source,
        )
    return rows, usage


def aggregate(setting_path: Path, run_dir: Path, judge_output: Path) -> None:
    setting = experiment.load_setting(setting_path)
    prediction_run = experiment.read_json(run_dir / "run.json")
    judge_run = experiment.read_json(run_dir / "judge_run.json")
    judged_rows, judge_usage = new_score_rows(setting, judge_output)
    rows = []
    for case in setting["cases"]:
        rows.append(judged_rows[(case["case_id"], "no_prefix")])
        for length in PREFIX_LENGTHS:
            condition = condition_name(length)
            key = (case["case_id"], condition)
            if case["observed_decision_count"] > length:
                rows.append(judged_rows[key])
                continue
            baseline = judged_rows[(case["case_id"], "full")]
            rows.append(
                {
                    **baseline,
                    "method": condition,
                    "displayed_decision_count": case["observed_decision_count"],
                    "score_source": "reused_current_full_prefix_judgment",
                }
            )
        rows.append(judged_rows[(case["case_id"], "full")])
    expected = setting["case_count"] * len(ALL_CONDITIONS)
    if len(rows) != expected:
        raise ValueError(f"assembled {len(rows)} ladder scores, expected {expected}")
    repository_rows, micro, macro, _ = experiment.summarize_rows(
        rows,
        setting["evaluation_indices"],
        ALL_CONDITIONS,
        [case["case_id"] for case in setting["cases"]],
    )
    results_dir = run_dir / "results"
    experiment.write_score_tables(results_dir, rows, repository_rows)
    summary = {
        "case_count": setting["case_count"],
        "conditions": list(ALL_CONDITIONS),
        "scored_condition_case_count": len(rows),
        "new_truncated_prediction_count": len(submitted_keys(setting)),
        "saved_endpoint_prediction_count": (
            setting["case_count"] * len(ENDPOINT_CONDITIONS)
        ),
        "judge_request_count": len(judged_rows),
        "reused_full_judgment_count": len(rows) - len(judged_rows),
        **experiment.run_models(prediction_run, judge_run),
        "judge_context": judge_run["judge_context"],
        "no_prefix_source_run_dir": judge_run["no_prefix_source_run_dir"],
        "full_prefix_source_run_dir": judge_run["full_prefix_source_run_dir"],
        "micro_average_over_all_cases": micro,
        "macro_average_over_repository_means": macro,
        "prediction_usage": judge_run["prediction_usage"],
        "judge_usage": judge_usage,
    }
    experiment.write_json(results_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_judging(args: argparse.Namespace) -> None:
    _, request_path = prepare_judging(args)
    output_path = args.run_dir / "judge_output.jsonl"
    experiment.run_requests(
        args, request_path, output_path, "Prefix-length judgments",
        experiment.validate_judgment_response,
    )
    aggregate(args.setting, args.run_dir, output_path)


def summarize_score_subset(
    scores: dict[str, dict], case_ids: list[str], case_by_id: dict[str, dict]
) -> tuple[dict, dict, int]:
    micro = {
        metric: statistics.fmean(scores[case_id][metric] for case_id in case_ids)
        for metric in experiment.REPORTED_METRICS
    }
    repository_ids = sorted(
        {case_by_id[case_id]["evaluation_idx"] for case_id in case_ids}
    )
    macro = {
        metric: statistics.fmean(
            statistics.fmean(
                scores[case_id][metric]
                for case_id in case_ids
                if case_by_id[case_id]["evaluation_idx"] == evaluation_idx
            )
            for evaluation_idx in repository_ids
        )
        for metric in experiment.REPORTED_METRICS
    }
    return micro, macro, len(repository_ids)


def load_ladder_scores(path: Path) -> dict[str, dict[str, dict]]:
    scores = {condition: {} for condition in ALL_CONDITIONS}
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            condition = raw["method"]
            if condition not in scores:
                raise ValueError(f"unexpected ladder condition {condition!r} in {path}")
            case_id = raw["case_id"]
            if case_id in scores[condition]:
                raise ValueError(f"duplicate {condition} score for {case_id} in {path}")
            scores[condition][case_id] = {
                **raw,
                **{
                    metric: int(raw[metric])
                    for metric in experiment.REPORTED_METRICS
                },
            }
    return scores


def repeat_aggregate(curve_rows: list[dict], analysis: str) -> dict:
    result = {}
    print(f"\n{analysis}")
    print("aggregation\tcondition\tmean_trajectory\tsample_sd\tmean_gap_to_full")
    for aggregation in ("micro", "macro"):
        result[aggregation] = {}
        full_rows = sorted(
            (
                row
                for row in curve_rows
                if row["analysis"] == analysis
                and row["aggregation"] == aggregation
                and row["condition"] == "full"
            ),
            key=lambda row: row["repeat"],
        )
        full_by_repeat = {
            row["repeat"]: row[experiment.HEADLINE_METRIC] for row in full_rows
        }
        for condition in ("no_prefix", *CONDITIONS, "full"):
            condition_rows = sorted(
                (
                    row
                    for row in curve_rows
                    if row["analysis"] == analysis
                    and row["aggregation"] == aggregation
                    and row["condition"] == condition
                ),
                key=lambda row: row["repeat"],
            )
            values = [row[experiment.HEADLINE_METRIC] for row in condition_rows]
            gaps = [
                row[experiment.HEADLINE_METRIC] - full_by_repeat[row["repeat"]]
                for row in condition_rows
            ]
            result[aggregation][condition] = {
                "case_count": condition_rows[0]["case_count"],
                "repository_count": condition_rows[0]["repository_count"],
                "repeat_values": values,
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "mean_condition_minus_full": statistics.fmean(gaps),
            }
            condition_result = result[aggregation][condition]
            print(
                f"{aggregation}\t{condition}\t{condition_result['mean']:.3f}\t"
                f"{condition_result['sample_sd']:.3f}\t"
                f"{condition_result['mean_condition_minus_full']:+.3f}"
            )
    return result


def summarize_runs(args: argparse.Namespace) -> None:
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    setting = experiment.load_setting(args.setting)
    case_by_id = {case["case_id"]: case for case in setting["cases"]}
    all_case_ids = list(case_by_id)
    common_case_ids = [
        case_id
        for case_id, case in case_by_id.items()
        if case["observed_decision_count"] > max(PREFIX_LENGTHS)
    ]
    if not common_case_ids:
        raise ValueError("no cases belong to the common prefix-length subset")
    curve_rows = []
    expected_settings = None
    for repeat in range(1, args.repeats + 1):
        ladder_dir = Path(f"{args.run_root}-r{repeat}")
        ladder_summary = experiment.read_json(ladder_dir / "results" / "summary.json")
        if ladder_summary.get("conditions") != list(ALL_CONDITIONS):
            raise ValueError(f"ladder conditions mismatch at repeat {repeat}")
        settings = {
            field: ladder_summary.get(field)
            for field in (
                "prediction_model",
                "prediction_api_provider",
                "prediction_reasoning_effort",
                "judge_model",
                "judge_api_provider",
                "judge_reasoning_effort",
            )
        }
        if expected_settings is None:
            expected_settings = settings
        elif experiment.canonical_json(settings) != experiment.canonical_json(
            expected_settings
        ):
            raise ValueError(
                f"model settings differ at repeat {repeat}: {settings}"
            )

        scores_by_condition = load_ladder_scores(
            ladder_dir / "results" / "scores.csv"
        )
        expected_cases = set(case_by_id)
        if any(
            set(scores) != expected_cases
            for scores in scores_by_condition.values()
        ):
            raise ValueError(f"ladder score cases differ at repeat {repeat}")
        for analysis, case_ids in (
            ("primary_common_observed_gt_10", common_case_ids),
            ("secondary_all_cases", all_case_ids),
        ):
            for condition, scores in scores_by_condition.items():
                micro, macro, repository_count = summarize_score_subset(
                    scores, case_ids, case_by_id
                )
                for aggregation, metrics in (("micro", micro), ("macro", macro)):
                    curve_rows.append(
                        {
                            "analysis": analysis,
                            "repeat": repeat,
                            "aggregation": aggregation,
                            "condition": condition,
                            "case_count": len(case_ids),
                            "repository_count": repository_count,
                            **{
                                metric: metrics[metric]
                                for metric in experiment.REPORTED_METRICS
                            },
                        }
                    )

    output_dir = Path(f"{args.run_root}-summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    table_names = {
        "primary_common_observed_gt_10": "primary_common_subset_curve.csv",
        "secondary_all_cases": "secondary_full_cohort_curve.csv",
    }
    for analysis, filename in table_names.items():
        experiment.write_csv(
            output_dir / filename, [row for row in curve_rows if row["analysis"] == analysis]
        )
    experiment.write_csv(
        output_dir / "primary_common_subset_cases.csv",
        [
            {
                "case_id": case_id,
                "evaluation_idx": case_by_id[case_id]["evaluation_idx"],
                "target_time_step_id": case_by_id[case_id]["target_time_step_id"],
                "observed_decision_count": case_by_id[case_id][
                    "observed_decision_count"
                ],
            }
            for case_id in common_case_ids
        ],
    )

    primary = repeat_aggregate(curve_rows, "primary_common_observed_gt_10")
    secondary = repeat_aggregate(curve_rows, "secondary_all_cases")
    experiment.write_json(
        output_dir / "summary.json",
        {
            "setting": experiment.recorded_path(args.setting),
            "repeats": args.repeats,
            "conditions": list(ALL_CONDITIONS),
            "run_root": experiment.recorded_path(args.run_root),
            "model_settings": expected_settings,
            "primary_analysis": {
                "subset_rule": "observed_decision_count > 10",
                "case_count": len(common_case_ids),
                "repository_count": len(
                    {
                        case_by_id[case_id]["evaluation_idx"]
                        for case_id in common_case_ids
                    }
                ),
                "curve": primary,
                "curve_table": table_names["primary_common_observed_gt_10"],
                "case_table": "primary_common_subset_cases.csv",
            },
            "secondary_analysis": {
                "subset_rule": "all prepared evaluation cases; conditions at or above the available history reuse full-prefix scores",
                "case_count": len(all_case_ids),
                "repository_count": len(setting["evaluation_indices"]),
                "curve": secondary,
                "curve_table": table_names["secondary_all_cases"],
            },
        },
    )
    print(f"\nwrote ladder summary to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prediction_parser = subparsers.add_parser(
        "run-predictions",
        help="run resumable last-k predictions, or no-prefix predictions with --no-prefix",
    )
    prediction_parser.add_argument("--setting", type=Path, default=DEFAULT_SETTING)
    prediction_parser.add_argument("--run-dir", type=Path, required=True)
    prediction_parser.add_argument(
        "--no-prefix", action="store_true",
        help="predict from the prediction instructions alone, the curve's no-prefix point",
    )
    experiment.add_execution_options(prediction_parser, 4000)

    judge_parser = subparsers.add_parser(
        "run-judging",
        help="judge truncated predictions and saved no/full-prefix endpoints",
    )
    judge_parser.add_argument("--setting", type=Path, default=DEFAULT_SETTING)
    judge_parser.add_argument("--run-dir", type=Path, required=True)
    judge_parser.add_argument("--no-prefix-run-dir", type=Path, required=True)
    judge_parser.add_argument("--full-prefix-run-dir", type=Path, required=True)
    experiment.add_execution_options(judge_parser, 3000)

    summarize_parser = subparsers.add_parser(
        "summarize", help="combine no-prefix, last-k, and full-prefix results"
    )
    summarize_parser.add_argument("--setting", type=Path, default=DEFAULT_SETTING)
    summarize_parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    summarize_parser.add_argument("--repeats", type=int, default=3)

    args = parser.parse_args()
    if args.command == "run-predictions":
        (run_no_prefix_predictions if args.no_prefix else run_predictions)(args)
    elif args.command == "run-judging":
        run_judging(args)
    else:
        summarize_runs(args)


if __name__ == "__main__":
    main()
