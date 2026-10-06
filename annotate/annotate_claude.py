"""Run the research-decision annotation pipeline through Claude Code.

This entry point reuses ``annotate_openrouter.py`` for evidence batching, rolling decision
state, citation validation, lifecycle handling, resumability, and final annotation
construction. Each model call invokes Claude Code noninteractively with tools and
session persistence disabled. Claude receives only the same prepared payload that
the OpenRouter backend would receive and must return the same JSON Schema.

The default output is ``annotation_claude.json`` with a separate
``.state.annotation_claude.json`` sidecar. Because this is a comparison run, it does
not replace the canonical annotation status in the filtered CSV.

This runner is intended for a Claude Max subscription. It refuses to run when
``ANTHROPIC_API_KEY`` is set, because Claude Code would then use separately billed
API usage rather than the subscription.
"""

import json
import math
import os
import shutil
import subprocess
import sys

import annotate_openrouter as annotate


DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
DEFAULT_CONTEXT_WINDOW = int(os.environ.get("CLAUDE_CONTEXT_TOKENS", "1000000"))
DEFAULT_EFFORT = os.environ.get("CLAUDE_EFFORT", "xhigh")
DEFAULT_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", "1800"))


class FatalClaudeError(annotate.AnnotationError):
    """Authentication or subscription exhaustion that should stop the batch."""


class ClaudeResultError(annotate.ModelResultError):
    """A paid or quota-consuming Claude call that did not return usable JSON."""


def normalize_claude_usage(response, requested_model, stage, metadata=None):
    usage = response.get("usage") if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = annotate.numeric(usage.get("input_tokens"), integer=True)
    cache_write_tokens = annotate.numeric(
        usage.get("cache_creation_input_tokens"), integer=True
    )
    cached_tokens = annotate.numeric(
        usage.get("cache_read_input_tokens"), integer=True
    )
    prompt_tokens = input_tokens + cache_write_tokens + cached_tokens
    completion_tokens = annotate.numeric(usage.get("output_tokens"), integer=True)
    model_usage = response.get("modelUsage") or response.get("model_usage") or {}
    reported_models = sorted(model_usage) if isinstance(model_usage, dict) else []
    entry = {
        "stage": stage,
        "at": annotate.utc_now(),
        "usage_reported": bool(usage),
        "requested_model": requested_model,
        "reported_model": ", ".join(reported_models),
        "response_id": response.get("session_id", ""),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "reasoning_tokens": 0,
        "cost": annotate.numeric(response.get("total_cost_usd")),
        "cost_details": {
            "basis": "claude_cli_reported_usd_not_incremental_subscription_charge"
        },
        "uncached_prompt_tokens": input_tokens + cache_write_tokens,
        "duration_ms": annotate.numeric(response.get("duration_ms"), integer=True),
        "api_duration_ms": annotate.numeric(
            response.get("duration_api_ms"), integer=True
        ),
        "turn_count": annotate.numeric(response.get("num_turns"), integer=True),
    }
    if metadata:
        entry["call_metadata"] = metadata
    return entry


def call_claude(
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
    command = [
        client["executable"],
        "--print",
        "--output-format",
        "json",
        "--model",
        model,
        "--effort",
        client["effort"],
        "--tools",
        "",
        "--safe-mode",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--system-prompt",
        system_prompt,
        "--json-schema",
        json.dumps(schema, separators=(",", ":")),
    ]
    payload = json.dumps(user_payload, ensure_ascii=False)
    try:
        completed = subprocess.run(
            command,
            input=payload,
            text=True,
            capture_output=True,
            timeout=client["timeout"],
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise annotate.AnnotationError(
            f"Claude CLI timed out after {client['timeout']} seconds during {usage_stage}"
        ) from exc

    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = (completed.stderr or completed.stdout).strip()[:500]
        usage_entry = normalize_claude_usage(
            {}, model, usage_stage, metadata=usage_metadata
        )
        raise ClaudeResultError(
            f"Claude returned invalid JSON from its CLI envelope: {detail}", usage_entry
        ) from exc

    usage_entry = normalize_claude_usage(
        response, model, usage_stage, metadata=usage_metadata
    )
    if completed.returncode or response.get("is_error"):
        detail = str(response.get("result") or completed.stderr or response)[:500]
        normalized = detail.lower()
        if any(
            marker in normalized
            for marker in (
                "authentication",
                "not logged in",
                "login required",
                "login expired",
                "hit your limit",
                "usage limit",
                "billing",
            )
        ):
            raise FatalClaudeError(f"Claude Code cannot continue: {detail}")
        raise ClaudeResultError(
            f"Claude returned invalid JSON structured output: {detail}", usage_entry
        )

    structured = response.get("structured_output")
    if not isinstance(structured, dict):
        raise ClaudeResultError(
            "Claude returned invalid JSON structured output: "
            f"subtype={response.get('subtype', 'unknown')}",
            usage_entry,
        )
    return structured, usage_entry


def recorded_claude_call(
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
        result, usage_entry = call_claude(
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
    except ClaudeResultError as exc:
        annotate.append_usage(state, exc.usage_entry)
        annotate.checkpoint_mapping(state, state_path)
        raise
    annotate.append_usage(state, usage_entry)
    annotate.checkpoint_mapping(state, state_path)
    return result


def claude_client():
    executable = shutil.which(os.environ.get("CLAUDE_CLI", "claude"))
    if not executable:
        sys.exit("Claude Code is not installed or CLAUDE_CLI is incorrect")
    if os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ANTHROPIC_API_KEY is set. Unset it before this Claude Max run to avoid "
            "separately billed Anthropic API usage."
        )
    if DEFAULT_EFFORT not in {"low", "medium", "high", "xhigh", "max"}:
        sys.exit("CLAUDE_EFFORT must be low, medium, high, xhigh, or max")

    status = subprocess.run(
        [executable, "auth", "status", "--json"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    try:
        auth = json.loads(status.stdout)
    except json.JSONDecodeError:
        sys.exit(f"could not read Claude Code auth status: {status.stderr.strip()}")
    if status.returncode or not auth.get("loggedIn"):
        sys.exit("Claude Code is not logged in; run `claude auth login` first")

    version = subprocess.run(
        [executable, "--version"],
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    ).stdout.strip()
    print(
        f"Claude Code {version}; auth={auth.get('authMethod', 'unknown')}; "
        f"effort={DEFAULT_EFFORT}",
        flush=True,
    )
    return {
        "executable": executable,
        "effort": DEFAULT_EFFORT,
        "timeout": DEFAULT_TIMEOUT,
    }


def main():
    args = annotate.parse_args(
        description=__doc__,
        default_model=DEFAULT_MODEL,
        default_annotation_file="annotation_claude.json",
        context_tokens_env="CLAUDE_CONTEXT_TOKENS",
    )
    args.backend = "claude-code"
    if args.context_window_tokens is None:
        args.context_window_tokens = DEFAULT_CONTEXT_WINDOW
    args.input_token_budget = math.floor(
        (args.context_window_tokens - annotate.DEFAULT_MAX_TOKENS)
        * args.context_utilization
    )
    annotate.run_annotation_batch(
        args,
        claude_client(),
        recorded_call=recorded_claude_call,
        fatal_error_type=FatalClaudeError,
        backend_name="Claude Code",
    )


if __name__ == "__main__":
    main()
