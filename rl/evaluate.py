"""Compare a GRPO adapter with its SFT initialization on the held-out prefixes."""

import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys

from sft.io import EVAL_SPLIT, ROOT, write_once, json_text, load_split, read_jsonl

sys.path.insert(0, str(ROOT / "harness"))
import experiment


DEFAULT_RUN = ROOT / "rl/runs/grpo-lr1e-4"
RUBRIC = ROOT / "harness/prompts/judge_rubrics_long.md"


def prepare(args):
    source = args.run_dir.resolve()
    training = json.loads((source / "run.json").read_text())
    if training.get("algorithm") != "grpo":
        raise ValueError("--run-dir must be a GRPO training run")
    if args.adapter:
        # An exported epoch checkpoint carries its own completed export record,
        # so a run still in progress can be evaluated before it finishes.
        adapter = args.adapter.resolve()
        if adapter.parent != source:
            raise ValueError("--adapter must be an exported checkpoint of --run-dir")
        updates = json.loads((adapter / "export_complete.json").read_text())["updates"]
    else:
        completed = json.loads((source / "completed.json").read_text())
        adapter = source / completed["adapter"]
        updates = completed["updates"]
        if json.loads((adapter / "export_complete.json").read_text())["updates"] != updates:
            raise ValueError("final export does not match its completed training run")
    base = json.loads((adapter / "dpo_base.json").read_text())
    if training["adapter_mode"] != "merged_sft_fresh_lora" or base != training["merged_base"]:
        raise ValueError("adapter export does not match its training run")
    data_dir = Path(training["data_dir"])
    train_cases = load_split(data_dir, "train_" + training["condition"])
    dataset = "eval_" + EVAL_SPLIT
    cases = load_split(data_dir, dataset)
    if {r["project_id"] for r in cases} & {r["project_id"] for r in train_cases}:
        raise ValueError("held-out evaluation projects overlap training projects")
    output = args.output_dir.resolve() if args.output_dir else source.with_name(source.name + "-eval")
    for protected in (source, Path(base["sft_checkpoint"]).parent.resolve(), data_dir.resolve()):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("choose an evaluation output separate from training runs and data")
    instructions = RUBRIC.read_text()
    if not instructions.strip():
        raise ValueError("judge rubric is empty")
    config = {
        "source_run": str(source),
        "algorithm": "grpo", "policy_adapter": str(adapter),
        "policy_updates": updates, "merged_base": base,
        "data_dir": str(data_dir),
        "split": EVAL_SPLIT, "dataset": dataset,
        "cases": len(cases), "projects": len({r["project_id"] for r in cases}),
        "generation": {"seed": args.seed, "temperature": args.temperature,
                       "max_new_tokens": args.max_new_tokens, "enable_thinking": False},
        "judge": {"model": "openai/gpt-5.6-sol:floor", "reasoning_effort": "medium",
                  "max_output_tokens": 3000,
                  "routing": experiment.provider_preferences("openrouter", 4, 20, False),
                  "schema": experiment.JUDGE_SCHEMA},
    }
    return output, config, cases, instructions


def predict_methods(output, config, device, prepare_only):
    # Separate processes release each GPU model before loading the next method.
    for method in ("sft_initialization", config["algorithm"]):
        command = [sys.executable, "-m", "rl.predict", "--adapter", config["policy_adapter"],
                   "--data-dir", config["data_dir"],
                   "--output-dir", str(output / "predictions" / method), "--device", device,
                   "--seed", str(config["generation"]["seed"]),
                   "--temperature", str(config["generation"]["temperature"]),
                   "--max-new-tokens", str(config["generation"]["max_new_tokens"])]
        if method == "sft_initialization":
            command.append("--sft-initialization")
        if prepare_only:
            command.append("--prepare-only")
        subprocess.run(command, cwd=ROOT, check=True)


def prepare_judging(output, config, cases, instructions):
    requests, rows = [], []
    expected_ids = {case["case_id"] for case in cases}
    for method in ("sft_initialization", config["algorithm"]):
        directory = output / "predictions" / method
        predictions = read_jsonl(directory / "predictions.jsonl")
        by_id = {r["case_id"]: r for r in predictions}
        if set(by_id) != expected_ids:
            raise ValueError(f"{method}: predictions do not cover exactly the selected cases")
        prediction_config = json.loads((directory / "run.json").read_text())
        expected = {"policy": method, "adapter": config["policy_adapter"], "merged_base": config["merged_base"],
                    "split": config["split"], "dataset": config["dataset"], **config["generation"]}
        if any(prediction_config[k] != value for k, value in expected.items()):
            raise ValueError(f"{method}: prediction settings differ from the comparison")
        for case in cases:
            saved = by_id[case["case_id"]]
            if any(saved[k] != case[k] for k in ("project_id", "target_step")):
                raise ValueError(f"{method}: prediction metadata differs for {case['case_id']}")
            if not isinstance(saved["raw_text"], str) or type(saved["valid_json"]) is not bool:
                raise ValueError(f"{method}: malformed prediction record for {case['case_id']}")
            try:
                parsed = json.loads(saved["raw_text"])
            except json.JSONDecodeError:
                parsed = None
            valid = (isinstance(parsed, dict) and set(parsed) == {"category", "decision"}
                     and parsed["category"] in experiment.TARGET_CATEGORIES
                     and isinstance(parsed["decision"], str) and bool(parsed["decision"].strip()))
            if saved["valid_json"] != valid or saved["prediction"] != (parsed if valid else None):
                raise ValueError(f"{method}: inconsistent saved parse for {case['case_id']}")
            if [m["role"] for m in case["messages"]] != ["system", "user"]:
                raise ValueError(f"unexpected prefix messages: {case['case_id']}")
            judge_case = {"case_id": case["case_id"], "target": case["target"],
                          "evaluation_input": case["messages"][1]["content"]}
            custom_id = experiment.judge_custom_id(case["case_id"], method)
            requests.append({
                "custom_id": custom_id, "method": "POST", "url": "/v1/responses",
                "body": experiment.response_body(
                    config["judge"]["model"], instructions.strip(),
                    experiment.render_judge_input(judge_case, parsed if valid else saved["raw_text"]),
                    "individual_prediction_judgment", experiment.JUDGE_SCHEMA,
                    config["judge"]["max_output_tokens"], config["judge"]["reasoning_effort"],
                    config["judge"]["routing"]),
            })
            rows.append({"custom_id": custom_id, "method": method, "case_id": case["case_id"],
                         "project_id": case["project_id"], "target_step": case["target_step"],
                         "prediction_valid_json": valid})
    random.Random(config["generation"]["seed"]).shuffle(requests)
    write_once(output / "judge_requests.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests))
    write_once(output / "judge_cases.json", json_text(rows))
    return rows


def summarize(output, config, metadata):
    algorithm = config["algorithm"]
    label = algorithm.upper()
    judgments, usage = experiment.read_response_output(
        output / "judge_output.jsonl", {r["custom_id"] for r in metadata})
    rows = []
    for item in metadata:
        judgment = judgments[item["custom_id"]]
        experiment.validate_judgment(item["custom_id"], judgment)
        rows.append({**item, "trajectory_alignment": experiment.headline_score(judgment), **judgment})
    # The harness usage summary retains token counts; read actual paid costs too.
    costs = []
    with (output / "judge_output.jsonl").open() as handle:
        for line in handle:
            response = json.loads(line)
            _, consumed = experiment.extract_response_text(response)
            cost = consumed.get("cost")
            if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                raise ValueError(f"missing or invalid saved usage.cost: {response['custom_id']}")
            costs.append(cost)
    steps = sorted({r["target_step"] for r in rows})
    summaries = {}
    per_project = []
    for method in ("sft_initialization", algorithm):
        method_rows = [r for r in rows if r["method"] == method]
        projects = sorted({r["project_id"] for r in method_rows})
        project_rows = [{"method": method, "project_id": p, **experiment.metric_summary(
            [r for r in method_rows if r["project_id"] == p])} for p in projects]
        per_project.extend(project_rows)
        summaries[method] = {
            "cases": len(method_rows), "invalid_json": sum(not r["prediction_valid_json"] for r in method_rows),
            "micro": experiment.metric_summary(method_rows), "project_macro": experiment.metric_summary(project_rows),
            "by_step": {str(s): experiment.metric_summary([r for r in method_rows if r["target_step"] == s]) for s in steps},
        }
    scores = {(r["method"], r["case_id"]): r["trajectory_alignment"] for r in rows}
    paired = [{"case_id": r["case_id"], "project_id": r["project_id"], "target_step": r["target_step"],
               "sft_initialization": r["trajectory_alignment"], algorithm: scores[(algorithm, r["case_id"])],
               "delta": scores[(algorithm, r["case_id"])] - r["trajectory_alignment"]}
              for r in rows if r["method"] == "sft_initialization"]
    comparison = {"mean_delta": statistics.mean(r["delta"] for r in paired),
                  f"{algorithm}_higher": sum(r["delta"] > 0 for r in paired), "equal": sum(r["delta"] == 0 for r in paired),
                  f"{algorithm}_lower": sum(r["delta"] < 0 for r in paired)}
    for name, values in (("scores.csv", rows), ("per_project.csv", per_project), ("paired_scores.csv", paired)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (output / "summary.json").write_text(json_text({"dataset": config["dataset"], "methods": summaries, "paired": comparison,
                                                   "judge_usage": {**usage, "cost_usd": math.fsum(costs)}}))
    labels = [f"First-{s}" for s in steps]
    table = ["| Method | " + " | ".join(labels) + " | Overall | Project macro |",
             "| --- | " + " | ".join(["---:"] * (len(steps) + 2)) + " |"]
    values = {}
    for method, name in (("sft_initialization", "SFT initialization"), (algorithm, f"{label} final")):
        s = summaries[method]
        values[method] = [s["by_step"][str(step)]["trajectory_alignment"] for step in steps]
        values[method] += [s[key]["trajectory_alignment"] for key in ("micro", "project_macro")]
        table.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in values[method]) + " |")
    table.append(f"| {label} minus SFT | " + " | ".join(f"{d-s:+.3f}" for d, s in
                 zip(values[algorithm], values["sft_initialization"])) + " |")
    table += ["", f"Dataset: {config['dataset']} (held-out prefixes).",
              f"Per-case judge-score comparisons: {label} higher {comparison[algorithm + '_higher']}, "
              f"equal {comparison['equal']}, lower {comparison[algorithm + '_lower']}.",
              "These compare independent reference-based scores, not direct pairwise judge verdicts.",
              f"Invalid JSON: SFT {summaries['sft_initialization']['invalid_json']}, {label} {summaries[algorithm]['invalid_json']}. "
              "Original raw text is judged for invalid JSON; all cases remain in averages.",
              f"Judge: Sol medium, original integer rubric, component + operation (0–4). "
              f"Saved successful-response API cost: ${math.fsum(costs):.6f}."]
    report = "\n".join(table) + "\n"
    (output / "comparison.md").write_text(report)
    print(report)


def run(args):
    if (not math.isfinite(args.temperature) or args.temperature < 0 or args.max_new_tokens < 1
            or min(args.concurrency, args.max_attempts) < 1):
        raise ValueError("invalid generation or judging settings")
    output, config, cases, instructions = prepare(args)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(output / "comparison_config.json", json_text(config))
        write_once(output / RUBRIC.name, instructions)
        print(f"Compare SFT initialization and {config['algorithm'].upper()} ({config['policy_updates']} updates) on "
              f"{len(cases)} held-out cases / {config['projects']} projects; {2 * len(cases)} judgments.", flush=True)
        print(f"Output: {output}", flush=True)
        if args.command in ("prepare", "predict", "all"):
            predict_methods(output, config, args.device, prepare_only=True)
        if args.command == "prepare":
            print("Prepared only; no model loaded or paid calls submitted.")
            return
        if args.command in ("predict", "all"):
            predict_methods(output, config, args.device, prepare_only=False)
        metadata = prepare_judging(output, config, cases, instructions)
        completed = experiment.completed_response_ids(output / "judge_output.jsonl",
                    {r["custom_id"] for r in metadata}, experiment.validate_judgment_response)
        pending = len(metadata) - len(completed)
        print(f"{len(completed)}/{len(metadata)} judged; {pending} pending.", flush=True)
        if args.command == "predict" or (pending and not args.confirm_submit):
            print("Judge requests prepared. Submit them with rl/run_evaluation.sh judge (or judge --confirm-submit).")
            return
        experiment.run_concurrent_requests(
            output / "judge_requests.jsonl", output / "judge_output.jsonl", output / "judge_errors.jsonl",
            f"SFT vs {config['algorithm'].upper()} judgments", "openrouter", args.concurrency, args.max_attempts,
            args.confirm_submit, experiment.validate_judgment_response)
        summarize(output, config, metadata)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "predict", "judge", "all"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--adapter", type=Path,
                        help="exported epoch checkpoint inside --run-dir; "
                             "default: the completed run's final adapter")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--confirm-submit", action="store_true")
    return parser


if __name__ == "__main__":
    run(argument_parser().parse_args())
