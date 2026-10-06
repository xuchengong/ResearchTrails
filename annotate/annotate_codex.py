"""Run the research-decision annotation pipeline through Codex CLI.

This entry point reuses ``annotate_openrouter.py`` for evidence batching, rolling
decision state, citation validation, lifecycle handling, resumability, and final
annotation construction. Each model call invokes ``codex exec`` noninteractively
in an empty read-only workspace with user configuration, repository instructions,
web search, and shell tools disabled. Codex receives the same system instructions,
prepared payload, and JSON Schema as the OpenRouter backend.

The default model is ``gpt-6-astra`` with ``xhigh`` reasoning effort. The default
output is ``annotation_codex.json`` with a separate
``.state.annotation_codex.json`` sidecar. Because this is a comparison run, it
does not replace the canonical annotation status in the filtered CSV.

Changing the model resumes the saved commit-mapping progress. Historical calls retain
their original model in the usage ledger; subsequent calls use the new model.
With ``--omit_decision_history_citation``, mapping requests also omit citations
from the five recent decisions. Local citations and open-thread evidence remain
intact.

This runner is intended for a ChatGPT account that includes Codex usage. It refuses
to run when ``OPENAI_API_KEY`` or ``CODEX_API_KEY`` is set, because either key can
switch the CLI from ChatGPT-managed usage to separately billed API usage.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import annotate_openrouter as annotate


DEFAULT_MODEL = os.environ.get("CODEX_MODEL", "gpt-6-astra")
DEFAULT_CONTEXT_WINDOW = int(os.environ.get("CODEX_CONTEXT_TOKENS", "1050000"))
DEFAULT_EFFORT = os.environ.get("CODEX_EFFORT", "xhigh")
DEFAULT_TIMEOUT = int(os.environ.get("CODEX_TIMEOUT", "1800"))


class FatalCodexError(annotate.AnnotationError):
    """Authentication or subscription exhaustion that should stop the batch."""


class CodexResultError(annotate.ModelResultError):
    """A quota-consuming Codex call that did not return usable JSON."""


def normalize_codex_usage(
    events, requested_model, stage, duration_ms, metadata=None
):
    completed_turns = [
        event
        for event in events
        if isinstance(event, dict) and event.get("type") == "turn.completed"
    ]
    usages = [
        event.get("usage")
        for event in completed_turns
        if isinstance(event.get("usage"), dict)
    ]
    prompt_tokens = sum(
        annotate.numeric(usage.get("input_tokens"), integer=True)
        for usage in usages
    )
    completion_tokens = sum(
        annotate.numeric(usage.get("output_tokens"), integer=True)
        for usage in usages
    )
    cached_tokens = sum(
        annotate.numeric(usage.get("cached_input_tokens"), integer=True)
        for usage in usages
    )
    reasoning_tokens = sum(
        annotate.numeric(usage.get("reasoning_output_tokens"), integer=True)
        for usage in usages
    )
    thread = next(
        (
            event
            for event in events
            if isinstance(event, dict) and event.get("type") == "thread.started"
        ),
        {},
    )
    entry = {
        "stage": stage,
        "at": annotate.utc_now(),
        "usage_reported": bool(usages),
        "requested_model": requested_model,
        "reported_model": requested_model,
        "response_id": thread.get("thread_id", ""),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": 0,
        "reasoning_tokens": reasoning_tokens,
        "cost": 0,
        "cost_details": {
            "basis": "chatgpt_managed_codex_usage_not_incremental_api_charge"
        },
        "uncached_prompt_tokens": max(prompt_tokens - cached_tokens, 0),
        "duration_ms": annotate.numeric(duration_ms, integer=True),
        "turn_count": len(completed_turns),
    }
    if metadata:
        entry["call_metadata"] = metadata
    return entry


def parse_codex_events(stdout):
    events = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Codex emitted invalid JSONL on line {line_number}: {line[:300]}"
            ) from exc
    return events


def codex_error_detail(events, stderr):
    details = []
    for event in events:
        if event.get("type") not in {"error", "turn.failed"}:
            continue
        error = event.get("error")
        if isinstance(error, dict):
            details.append(str(error.get("message") or error))
        elif error:
            details.append(str(error))
        elif event.get("message"):
            details.append(str(event["message"]))
    if stderr.strip():
        details.append(stderr.strip())
    return " | ".join(details)[:1000] or "Codex exited without an error message"


def call_codex(
    client,
    model,
    system_prompt,
    user_payload,
    schema,
    schema_name,
    usage_stage,
    usage_metadata=None,
    max_tokens=annotate.DEFAULT_MAX_TOKENS,
):
    # Codex CLI has no per-run max-output flag. The shared batch planner uses
    # max_tokens before this boundary to keep each requested delta bounded.
    del max_tokens
    with tempfile.TemporaryDirectory(prefix="trajectory-codex-") as temporary:
        workspace = Path(temporary)
        schema_path = workspace / f"{schema_name}.schema.json"
        output_path = workspace / "result.json"
        schema_path.write_text(json.dumps(schema, ensure_ascii=False))
        command = [
            client["executable"],
            "exec",
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--cd",
            str(workspace),
            "--model",
            model,
            "--config",
            f'model_reasoning_effort="{client["effort"]}"',
            "--config",
            'model_reasoning_summary="none"',
            "--config",
            f"developer_instructions={json.dumps(system_prompt)}",
            "--config",
            'web_search="disabled"',
            "--config",
            "features.shell_tool=false",
            "--config",
            "features.apps=false",
            "--config",
            "features.multi_agent=false",
            "--config",
            "features.remote_plugin=false",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "-",
        ]
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(user_payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=client["timeout"],
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise annotate.AnnotationError(
                f"Codex CLI timed out after {client['timeout']} seconds during "
                f"{usage_stage}"
            ) from exc
        duration_ms = round((time.monotonic() - started) * 1000)

        try:
            events = parse_codex_events(completed.stdout)
        except ValueError as exc:
            usage_entry = normalize_codex_usage(
                [], model, usage_stage, duration_ms, usage_metadata
            )
            raise CodexResultError(str(exc), usage_entry) from exc
        usage_entry = normalize_codex_usage(
            events, model, usage_stage, duration_ms, usage_metadata
        )
        failed = any(event.get("type") == "turn.failed" for event in events)
        if completed.returncode or failed:
            detail = codex_error_detail(events, completed.stderr)
            normalized = detail.lower()
            if any(
                marker in normalized
                for marker in (
                    "authentication",
                    "not logged in",
                    "login required",
                    "unauthorized",
                    "forbidden",
                    "usage limit",
                    "rate limit",
                    "quota",
                    "billing",
                )
            ):
                raise FatalCodexError(f"Codex CLI cannot continue: {detail}")
            raise CodexResultError(
                f"Codex CLI did not complete successfully: {detail}", usage_entry
            )
        if not output_path.exists():
            raise CodexResultError(
                "Codex CLI completed without writing its structured output",
                usage_entry,
            )
        raw_output = output_path.read_text()
        try:
            structured = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise CodexResultError(
                f"Codex CLI wrote invalid JSON structured output: {raw_output[:500]}",
                usage_entry,
            ) from exc
        if not isinstance(structured, dict):
            raise CodexResultError(
                "Codex CLI structured output is not a JSON object", usage_entry
            )
        return structured, usage_entry


def recorded_codex_call(
    state,
    state_path,
    client,
    model,
    system_prompt,
    user_payload,
    schema,
    schema_name,
    usage_stage,
    usage_metadata=None,
    max_tokens=annotate.DEFAULT_MAX_TOKENS,
):
    try:
        result, usage_entry = call_codex(
            client,
            model,
            system_prompt,
            user_payload,
            schema,
            schema_name,
            usage_stage,
            usage_metadata,
            max_tokens,
        )
    except CodexResultError as exc:
        annotate.append_usage(state, exc.usage_entry)
        annotate.checkpoint_mapping(state, state_path)
        raise
    annotate.append_usage(state, usage_entry)
    annotate.checkpoint_mapping(state, state_path)
    return result


def codex_client():
    executable = shutil.which(os.environ.get("CODEX_CLI", "codex"))
    if not executable:
        sys.exit("Codex CLI is not installed or CODEX_CLI is incorrect")
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY"):
        sys.exit(
            "OPENAI_API_KEY or CODEX_API_KEY is set. Unset both before this "
            "ChatGPT-managed Codex run to avoid separately billed API usage."
        )
    if DEFAULT_EFFORT not in {"minimal", "low", "medium", "high", "xhigh"}:
        sys.exit("CODEX_EFFORT must be minimal, low, medium, high, or xhigh")

    status = subprocess.run(
        [executable, "login", "status"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    status_text = "\n".join(
        text.strip() for text in (status.stdout, status.stderr) if text.strip()
    )
    if status.returncode or "logged in using chatgpt" not in status_text.lower():
        sys.exit(
            "Codex CLI is not logged in through ChatGPT; run `codex login` first. "
            f"Status: {status_text}"
        )

    version = subprocess.run(
        [executable, "--version"],
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    ).stdout.strip()
    print(
        f"{version}; auth=ChatGPT; model={DEFAULT_MODEL}; effort={DEFAULT_EFFORT}",
        flush=True,
    )
    return {
        "executable": executable,
        "effort": DEFAULT_EFFORT,
        "timeout": DEFAULT_TIMEOUT,
    }


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--omit_decision_history_citation",
        action="store_true",
        help="omit recent decisions' historical citations from mapping requests; keep them locally",
    )
    args = annotate.parse_args(
        description=__doc__,
        default_model=DEFAULT_MODEL,
        default_annotation_file="annotation_codex.json",
        context_tokens_env="CODEX_CONTEXT_TOKENS",
        parents=[parser],
    )
    args.backend = "codex-cli"
    args.reasoning_effort = DEFAULT_EFFORT
    args.resume_on_model_change = True
    if args.context_window_tokens is None:
        args.context_window_tokens = DEFAULT_CONTEXT_WINDOW
    args.input_token_budget = math.floor(
        (args.context_window_tokens - annotate.DEFAULT_MAX_TOKENS)
        * args.context_utilization
    )
    annotate.run_annotation_batch(
        args,
        codex_client(),
        recorded_call=recorded_codex_call,
        fatal_error_type=FatalCodexError,
        backend_name="Codex CLI",
        respect_annotation_status=True,
    )


if __name__ == "__main__":
    main()
