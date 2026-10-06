#!/usr/bin/env python3
"""Compare plain research advice with pattern-augmented advice, without a judge."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
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

ENDPOINT = "https://openrouter.ai/api/v1/responses"
MODELS = ("openai/gpt-5.6-sol",)
# case: (query in queries/, pattern in patterns/)
CASES = {
    "bug_repair": ("bug_repair.md", "intermediate_target.md"),
    "answer_judge": ("answer_judge.md", "premise_audit.md"),
}
INSTRUCTIONS = (
    "You are a research collaborator. Answer the user's question in clear, concrete "
    "language. Respect the stated resource constraints. Distinguish reported "
    "observations from assumptions; do not invent experimental results."
)


def prepare_requests(args) -> dict:
    query = args.query.read_text(encoding="utf-8")
    pattern = args.pattern.read_text(encoding="utf-8")
    if not query.strip() or not pattern.strip():
        raise ValueError("query and pattern files must both contain text")
    if len(set(args.models)) != len(args.models):
        raise ValueError("model names must be unique")
    if min(args.repetitions, args.max_output_tokens, args.concurrency, args.timeout) <= 0:
        raise ValueError("repetitions, token budget, concurrency, and timeout must be positive")
    requests = []
    for model in args.models:
        for repetition in range(1, args.repetitions + 1):
            for condition in ("without_pattern", "with_pattern"):
                instructions = INSTRUCTIONS
                if condition == "with_pattern":
                    instructions += "\n\nConsider the following research pattern when it fits the user's situation:\n\n" + pattern
                requests.append({
                    "custom_id": f"{re.sub(r'[^a-zA-Z0-9_-]', '-', model)}__r{repetition}__{condition}",
                    "condition": condition,
                    "repetition": repetition,
                    "body": {
                        "model": model,
                        "instructions": instructions,
                        "input": query,
                        "store": False,
                        "max_output_tokens": args.max_output_tokens,
                        "reasoning": {"effort": args.reasoning_effort},
                        "provider": {"sort": "price", "require_parameters": True,
                                     "ignore": ["azure"],
                                     "max_price": {"prompt": 5, "completion": 25}},
                    },
                })
    if len({r['custom_id'] for r in requests}) != len(requests):
        raise ValueError("model names produce conflicting output IDs")
    return {"endpoint": ENDPOINT, "query_file": recorded_path(args.query),
            "pattern_file": recorded_path(args.pattern), "requests": requests}


def post_response(body: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                 "Content-Type": "application/json",
                 "X-OpenRouter-Title": "research-user-simulation"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def answer_text(body: dict) -> str:
    if body.get("error") or body.get("status") != "completed":
        raise ValueError(f"response not completed: {body.get('status')}; {body.get('error') or body.get('incomplete_details')}")
    text = "".join(part["text"] for output in body["output"] if output["type"] == "message"
                   for part in output["content"] if part["type"] == "output_text")
    if not text.strip():
        raise ValueError("response contains no answer text")
    return text


def generate(request: dict, timeout: int, transport) -> dict:
    result = {"custom_id": request["custom_id"], "status": "failed"}
    try:
        result["response"] = transport(request["body"], timeout)
        answer_text(result["response"])
        result["status"] = "completed"
    except urllib.error.HTTPError as exc:
        result["error"] = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def export_comparison(directory: Path, plan: dict, latest: dict) -> None:
    lines = ["# Research advice: qualitative comparison", "",
             "Original generated answers; no judge, scores, or editorial rewriting.", "",
             "## User query", "", plan["requests"][0]["body"]["input"], ""]
    usage = []
    for request in plan["requests"]:
        cid = request["custom_id"]
        result = latest.get(cid)
        lines.extend([f"## {request['body']['model']} · r{request['repetition']} · {request['condition']}", ""])
        if result is None:
            lines.extend(["Not submitted.", ""])
        elif result["status"] == "completed":
            lines.extend([answer_text(result["response"]), ""])
            usage.append({"custom_id": cid, "response_id": result["response"]["id"],
                          "model": result["response"]["model"],
                          "usage": result["response"].get("usage", {})})
        else:
            lines.extend([f"Status: {result['status']}. See attempts.jsonl.", ""])
    (directory / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    (directory / "usage.json").write_text(json.dumps(usage, indent=2) + "\n", encoding="utf-8")


def run(args, transport=post_response) -> None:
    plan = prepare_requests(args)
    directory = args.run_dir
    directory.mkdir(parents=True, exist_ok=True)
    # One launcher per run protects against duplicate paid requests.
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process is using {directory}") from exc
        prepared = directory / "requests.json"
        if prepared.exists():
            if json.loads(prepared.read_text()) != plan:
                raise ValueError("query, pattern, or generation settings changed; use a new --run-dir")
        else:
            with prepared.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        requests = {r["custom_id"]: r for r in plan["requests"]}
        ledger = directory / "attempts.jsonl"
        latest = {}
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                record = json.loads(line)
                if record["custom_id"] not in requests or record["status"] not in {"started", "completed", "failed"}:
                    raise ValueError("attempt ledger does not match prepared requests")
                if record["status"] == "completed":
                    answer_text(record["response"])
                latest[record["custom_id"]] = record
        remaining = [r for cid, r in requests.items() if cid not in latest or latest[cid]["status"] != "completed"]
        export_comparison(directory, plan, latest)
        if args.dry_run:
            print(f"Prepared {len(requests)} requests; {len(remaining)} pending. No API calls made.\n{prepared}")
            return
        if not remaining:
            print(f"All {len(requests)} answers already completed; no API calls made.\n{directory / 'COMPARISON.md'}")
            return
        if any(r["custom_id"] in latest for r in remaining) and not args.retry_errors:
            raise ValueError("prior failed or interrupted attempts exist; inspect attempts.jsonl, then use --retry-errors to resubmit")
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY is required; export it or source the repository .env")
        with ledger.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {}
            for request in remaining:
                started = {"custom_id": request["custom_id"], "status": "started",
                           "recorded_at": datetime.now(timezone.utc).isoformat()}
                handle.write(json.dumps(started) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                latest[request["custom_id"]] = started
                print(f"Submitting {request['custom_id']}", flush=True)
                futures[pool.submit(generate, request, args.timeout, transport)] = request["custom_id"]
            for future in as_completed(futures):
                result = future.result()
                result["recorded_at"] = datetime.now(timezone.utc).isoformat()
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                latest[result["custom_id"]] = result
                export_comparison(directory, plan, latest)
                print(f"{result['status']}: {result['custom_id']}", flush=True)
        failed = [cid for cid, result in latest.items() if result["status"] != "completed"]
        if failed:
            raise RuntimeError(f"{len(failed)} requests failed. Raw responses/errors saved in {ledger}")
        print(directory / "COMPARISON.md")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES, help="the query and its pattern")
    parser.add_argument("--run-dir", type=Path, help="default: runs/<case>")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args(argv)
    query, pattern = CASES[args.case]
    args.query = HERE / "queries" / query
    args.pattern = HERE / "patterns" / pattern
    args.run_dir = args.run_dir or HERE / "runs" / args.case
    return args


if __name__ == "__main__":
    run(parse_args())
