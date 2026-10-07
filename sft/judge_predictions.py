"""Judge saved SFT predictions with the concurrent harness's rubric/schema.

Prepare offline by default; --confirm-submit submits only unfinished judgments.
"""

import argparse
import csv
import fcntl
import json
from pathlib import Path
import sys

from sft.io import EVAL_SPLIT, ROOT, write_once, json_text, load_split, read_jsonl

# The harness uses sibling imports. Reuse its renderer, schema, validation, and
# transport directly, without importing the GPU training/prediction modules.
sys.path.insert(0, str(ROOT / "harness"))
import experiment


DEFAULT_RUBRIC = ROOT / "harness/prompts/judge_rubrics_long.md"


def prepare_method(args, method, cases, instructions, routing):
    source = args.run_dir / "predictions" / method
    predictions = read_jsonl(source / "predictions.jsonl")
    by_id = {row["case_id"]: row for row in predictions}
    expected = {case["case_id"] for case in cases}
    if set(by_id) != expected:
        raise ValueError(
            f"{source}: prediction case IDs differ from eval_{EVAL_SPLIT}; "
            f"missing={sorted(expected - set(by_id))[:5]}, "
            f"extra={sorted(set(by_id) - expected)[:5]}"
        )
    config = json.loads((source / "run.json").read_text())
    if config["split"] != EVAL_SPLIT:
        raise ValueError(f"{source}: predictions used a different split")
    model_info = json.loads((source / "model_info.json").read_text())
    comparison = {
        key: config[key] for key in (
            "base_model", "enable_thinking", "precision", "max_new_tokens"
        )
    }
    comparison["resolved_revision"] = model_info["resolved_revision"]
    requests, rows = [], []
    for case in cases:
        saved = by_id[case["case_id"]]
        if any(saved[key] != case[key] for key in ("project_id", "target_step")):
            raise ValueError(f"{source}: mismatched metadata for {case['case_id']}")
        if not isinstance(saved["raw_text"], str) or type(saved["valid_json"]) is not bool:
            raise ValueError(f"{source}: malformed prediction record for {case['case_id']}")
        # Verify the saved native-schema parse; malformed model outputs remain
        # semantic judge inputs, while malformed prediction records fail here.
        try:
            parsed = json.loads(saved["raw_text"])
        except json.JSONDecodeError:
            parsed = None
        valid = (
            isinstance(parsed, dict) and set(parsed) == {"category", "decision"}
            and parsed["category"] in experiment.TARGET_CATEGORIES
            and isinstance(parsed["decision"], str) and bool(parsed["decision"].strip())
        )
        if saved["valid_json"] != valid or saved["prediction"] != (parsed if valid else None):
            raise ValueError(f"{source}: inconsistent saved parse for {case['case_id']}")
        if [message["role"] for message in case["messages"]] != ["system", "user"]:
            raise ValueError(f"unexpected case messages for {case['case_id']}")
        judge_case = {
            "case_id": case["case_id"], "target": case["target"],
            "evaluation_input": case["messages"][1]["content"],
        }
        # No synthetic object/operation fields. For invalid JSON, serialize the
        # original text as a string instead of repairing it or dropping the case.
        prediction = saved["prediction"] if valid else saved["raw_text"]
        custom_id = experiment.judge_custom_id(case["case_id"], method.replace("/", "__"))
        requests.append({
            "custom_id": custom_id, "method": "POST", "url": "/v1/responses",
            "body": experiment.response_body(
                args.model, instructions.strip(), experiment.render_judge_input(judge_case, prediction),
                "individual_prediction_judgment", experiment.JUDGE_SCHEMA,
                args.max_output_tokens, args.reasoning_effort, routing,
            ),
        })
        rows.append({
            "custom_id": custom_id, "method": method, "case_id": case["case_id"],
            "project_id": case["project_id"], "target_step": case["target_step"],
            "target_category": case["target"]["category"], "target_decision": case["target"]["decision"],
            "prediction_valid_json": valid,
        })
    metadata = {
        "prediction_dir": str(source.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "split": EVAL_SPLIT, "cases": len(cases),
        "invalid_predictions": sum(not row["prediction_valid_json"] for row in rows),
    }
    return {"method": method, "requests": requests, "rows": rows, "metadata": metadata,
            "comparison": comparison}


def summarize_method(directory, plan):
    judgments, usage = experiment.read_response_output(
        directory / "judge_output.jsonl", {row["custom_id"] for row in plan["rows"]}
    )
    rows = []
    for source in plan["rows"]:
        judgment = judgments[source["custom_id"]]
        experiment.validate_judgment(source["custom_id"], judgment)
        rows.append({
            **source, experiment.HEADLINE_METRIC: experiment.headline_score(judgment),
            **{key: judgment[key] for key in (*experiment.JUDGE_METRICS, "justification")},
        })
    projects = sorted({row["project_id"] for row in rows})
    per_project = [
        {"project_id": project, **experiment.metric_summary([
            row for row in rows if row["project_id"] == project
        ])} for project in projects
    ]
    summary = {
        "method": plan["method"], "cases": len(rows), "projects": len(projects),
        "invalid_predictions": plan["metadata"]["invalid_predictions"],
        "micro": experiment.metric_summary(rows),
        "project_macro": experiment.metric_summary(per_project),
        "by_step": {
            str(step): experiment.metric_summary([row for row in rows if row["target_step"] == step])
            for step in sorted({row["target_step"] for row in rows})
        },
        "judge_usage": usage,
    }
    for name, records in (("scores.csv", rows), ("per_project.csv", per_project)):
        with (directory / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (directory / "summary.json").write_text(json_text(summary))
    return summary


def run(args):
    if min(args.concurrency, args.max_attempts, args.max_output_tokens) < 1 or not args.model.strip():
        raise ValueError("judge model must be nonempty; concurrency, attempts, and token limit must be positive")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("methods must be unique")
    instructions = args.judge_instructions.read_text()
    if not instructions.strip():
        raise ValueError("judge instructions are empty")
    routing = experiment.provider_preferences(
        args.api_provider, args.openrouter_max_prompt_price,
        args.openrouter_max_completion_price, args.allow_azure,
    )
    cases = load_split(args.data_dir, f"eval_{EVAL_SPLIT}")
    # An method is a directory under predictions/, e.g. one per training checkpoint.
    # Validate every selected run before making any paid requests.
    plans = [prepare_method(args, method, cases, instructions, routing) for method in args.methods]
    if any(plan["comparison"] != plans[0]["comparison"] for plan in plans[1:]):
        raise ValueError("selected prediction runs differ in base model/revision or decoding settings")
    output = args.output_dir or args.run_dir / "judging" / "judge_prediction"
    output.mkdir(parents=True, exist_ok=True)
    # One process owns preparation and submission, preventing duplicate paid calls.
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(output / "judge_config.json", json_text({
            "model": args.model, "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens, "api_provider": args.api_provider,
            "provider_preferences": routing, "schema": experiment.JUDGE_SCHEMA,
            "rubric_path": str(args.judge_instructions.resolve()),
            "invalid_output_policy": "Judge original raw text; include in all score averages; report JSON validity separately.",
        }))
        write_once(output / args.judge_instructions.name, instructions)
        pending = {}
        for plan in plans:
            directory = output / plan["method"]
            write_once(directory / "judge_run.json", json_text(plan["metadata"]))
            write_once(directory / "judge_requests.jsonl", "".join(
                json.dumps(item, ensure_ascii=False) + "\n" for item in plan["requests"]
            ))
            completed = experiment.completed_response_ids(
                directory / "judge_output.jsonl", {row["custom_id"] for row in plan["rows"]},
                experiment.validate_judgment_response,
            )
            pending[plan["method"]] = len(cases) - len(completed)
            print(f"{plan['method']}: {len(completed)}/{len(cases)} judged; "
                  f"{plan['metadata']['invalid_predictions']} invalid prediction JSON")
        print(f"{sum(pending.values())} pending judge requests; output: {output}")
        if any(pending.values()) and not args.confirm_submit:
            print("Prepared only. Submit with sft/run_checkpoint_generation.sh judge (or add --confirm-submit).")
            return
        summaries = []
        for plan in plans:
            directory = output / plan["method"]
            experiment.run_concurrent_requests(
                directory / "judge_requests.jsonl", directory / "judge_output.jsonl",
                directory / "judge_errors.jsonl", f"Judgments ({plan['method']})",
                args.api_provider, args.concurrency, args.max_attempts, args.confirm_submit,
                experiment.validate_judgment_response,
            )
            summaries.append(summarize_method(directory, plan))
        steps = sorted({row["target_step"] for row in cases})
        columns = [f"First-{step}" for step in steps]
        table = [
            "| Method | " + " | ".join(columns) + " | Overall | Project macro |",
            "| --- | " + " | ".join(["---:"] * (len(steps) + 2)) + " |",
        ]
        for summary in summaries:
            values = [summary["by_step"][str(step)][experiment.HEADLINE_METRIC] for step in steps]
            values.extend(summary[key][experiment.HEADLINE_METRIC] for key in ("micro", "project_macro"))
            table.append(f"| {summary['method']} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
        comparison = output / "comparisons" / "__".join(method.replace("/", "-") for method in args.methods)
        comparison.parent.mkdir(exist_ok=True)
        comparison.with_suffix(".json").write_text(json_text(summaries))
        comparison.with_suffix(".md").write_text("\n".join(table) + "\n")
        print("\n".join(table))


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "sft/runs/qwen3-8b")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "sft/data")
    parser.add_argument("--output-dir", type=Path, help="default: RUN_DIR/judging/judge_prediction")
    parser.add_argument("--methods", nargs="+", required=True,
                        help="prediction directories under predictions/ to judge")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--judge-instructions", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--max-output-tokens", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, default=experiment.DEFAULT_CONCURRENCY)
    parser.add_argument("--max-attempts", type=int, default=experiment.DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--confirm-submit", action="store_true")
    experiment.add_api_options(parser)
    return parser


if __name__ == "__main__":
    run(argument_parser().parse_args())
