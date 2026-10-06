"""The next-decision prediction prompt: instructions, annotation checks, and observed prefixes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from render_demonstrations import decision_action_text


HERE = Path(__file__).resolve().parent
TASK_PROMPT = HERE / "prompts" / "next_decision_system.md"
DEMONSTRATIONS = HERE / "prompts" / "demonstrations.md"
SKILL = HERE / "skills" / "trajectory" / "SKILL.md"


def parse_date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp is not timezone-aware: {value}")
    return parsed


def validate_annotation(path: Path, annotation: dict) -> list[dict]:
    decisions = annotation.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError(f"{path} has no decisions")
    if [decision.get("time_step_id") for decision in decisions] != list(range(len(decisions))):
        raise ValueError(f"{path} has stale or nonsequential time_step_id values")
    first_dates = []
    for decision in decisions:
        first_dates.append(parse_date(decision["first_date"]))
        parse_date(decision["last_date"])
    if first_dates != sorted(first_dates):
        raise ValueError(f"{path} decisions are not ordered by timezone-aware first_date")
    return decisions


def skill_body(path: Path = SKILL) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"{path} is missing YAML frontmatter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"{path} has unterminated YAML frontmatter") from exc
    return "\n".join(lines[end + 1 :]).strip()


def observed_prefix(decisions: list[dict], target_step: int) -> list[dict]:
    """Every decision introduced before the target, in first_date order."""
    if target_step <= 0 or target_step >= len(decisions):
        raise ValueError(f"target step must be between 1 and {len(decisions) - 1}")
    return decisions[:target_step]


def completed_prior_decisions(decisions: list[dict], target_step: int) -> list[dict]:
    """Earlier decisions that ended before the target began."""
    target_start = parse_date(decisions[target_step]["first_date"])
    return [
        decision
        for decision in observed_prefix(decisions, target_step)
        if parse_date(decision["last_date"]) < target_start
    ]


def render_prefix(decisions: list[dict], target_step: int) -> str:
    observed = observed_prefix(decisions, target_step)
    target_start = parse_date(decisions[target_step]["first_date"])
    observed_by_id = {decision["decision_id"]: decision for decision in observed}
    prefix_description = (
        "All decisions introduced before the held-out decision are shown in "
        "timezone-aware first-date order. Outcomes that were not established "
        "by the cutoff are withheld."
    )
    lines = [
        "# Evaluation trajectory prefix",
        "",
        prefix_description,
        "",
        "Observed decisions:",
        "",
    ]
    for decision in observed:
        completed = parse_date(decision["last_date"]) < target_start
        replacement = observed_by_id.get(decision.get("superseded_by"))
        if not completed:
            state = "active at cutoff"
        elif decision["outcome"] == "retained":
            state = "retained by cutoff"
        elif decision["outcome"] == "abandoned":
            state = "abandoned by cutoff"
        elif decision["outcome"] == "superseded":
            if replacement:
                state = f"superseded by observed T{replacement['time_step_id']:03d}"
            else:
                state = "retained by cutoff"
        else:
            raise ValueError(
                f"decision {decision['decision_id']} has invalid outcome "
                f"{decision['outcome']!r}"
            )
        lines.extend(
            [
                f"- T{decision['time_step_id']:03d} | {decision['category']} | {state}",
                f"  {decision_action_text(decision['decision'])}",
                "",
            ]
        )
    lines.append("Predict the next research decision. Return the required JSON only.")
    return "\n".join(lines)
