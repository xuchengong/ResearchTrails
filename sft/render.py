"""Leakage-aware v1 prefix rendering, independent of the experiment harness.

The action sanitizer and cutoff-local states intentionally preserve the existing
evaluation semantics. They do not make annotation-level text leakage impossible:
annotations remain retrospective labels and should still be audited.
"""

from datetime import datetime
import json
from pathlib import Path
import re

CATEGORIES = ["method", "experiment", "ablation"]

HISTORY = re.compile(
    r"^(?:the earlier\b|the later\b|the experiment began\b|the model set expanded\b|"
    r"the protocol iterated\b|the run selection was\b|both .*\bhistorically\b|"
    r"this revises the earlier\b)", re.I,
)
RETROSPECTIVE = re.compile(
    r"(?:\b(?:temporarily|briefly|initially|ultimately|finalized)\b|\bfor later\b|"
    r",\s*with\s+the\s+(?:final|latest|retained|mature)\b|\bremains?\s+operative\b|"
    r"\b(?:later|subsequently|eventually)\s+"
    r"(?:remov\w*|delet\w*|disabl\w*|discontinu\w*|abandon\w*|supersed\w*|"
    r"replac\w*|restor\w*|return\w*|revert\w*)\b|"
    r"\b(?:was|were|is|are)\s+(?:later|subsequently|eventually)\s+"
    r"(?:remov\w*|delet\w*|disabl\w*|discontinu\w*|abandon\w*|supersed\w*|"
    r"replac\w*|commented\s+out|restor\w*|return\w*|revert\w*)\b|"
    r"\b(?:then|before being)\s+(?:remov\w*|delet\w*|disabl\w*|discontinu\w*|"
    r"abandon\w*|supersed\w*|replac\w*|restor\w*|return\w*|revert\w*)\b|"
    r"\b(?:branch|arm|suite|comparison|condition|computation|invocation|launcher|"
    r"route|paths?|implementation|baseline|experiment|protocol)\b.{0,100}\b"
    r"(?:was|were|is|are)\s+(?:removed|deleted|disabled|discontinued|abandoned|"
    r"superseded|commented\s+out)\b|\b(?:was|were)\s+historically\s+"
    r"(?:included|covered|tested|used)\b|\b(?:was|were)\s+(?:\w+[\s,/.-]+){0,20}"
    r"historically\b|\b(?:included|covered|tested|reported|run)\s+historically\b)", re.I,
)
STATUS_SUBJECT = re.compile(
    r"^(?:the\s+)?(?:latest|retained|mature)\b.*\b"
    r"(?:retains?|removes?|drops?|excludes?|discontinues?|replaces?)\b", re.I,
)


def action_text(text: str) -> str:
    """Remove retrospective outcome prose, retaining the annotated action."""
    sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        clauses = []
        for clause in re.split(r";\s+", sentence):
            clause = re.sub(r"^(?:Temporarily|Initially|Briefly|Ultimately)\s+", "", clause, flags=re.I)
            # Preserve an action qualified by a temporary duration, including
            # actions after a task-specific introductory phrase.
            clause, temporary_action = re.subn(
                r"(\bby\s+|^For [^,]+,\s*)(?:temporarily|briefly)\s+",
                r"\1", clause, flags=re.I,
            )
            if HISTORY.search(clause) or STATUS_SUBJECT.search(clause):
                continue
            marker = RETROSPECTIVE.search(clause)
            if marker:
                prefix = clause[:marker.start()].rstrip(" ,;:\u2014-")
                if temporary_action:
                    prefix = re.sub(r",?\s+(?:and|but)$", "", prefix, flags=re.I)
                if (len(prefix.split()) < 6 or HISTORY.search(prefix)
                        or re.match(r"^(?:the|this|that|these|those|a|an|both)\b", clause, re.I)):
                    continue
                clause = prefix
            clause = re.sub(r"\b(?:latest|retained|mature)\s+(?!(?:and|or)\b)", "", clause, flags=re.I)
            clause = re.sub(
                r"\bfinal\s+(?=(?:pipeline|protocol|configuration|variant|comparison|"
                r"suite|route|supplied objective|three signals)\b)", "", clause, flags=re.I,
            )
            clause = re.sub(r"\b(?:historical|historically|subsequent)\s+", "", clause, flags=re.I)
            clause = re.sub(r"\s{2,}", " ", clause).strip(" ,;")
            if clause:
                clauses.append(clause)
        if clauses:
            normalized = "; ".join(clauses)
            normalized = normalized[0].upper() + normalized[1:]
            sentences.append(normalized + ("" if normalized[-1] in ".!?" else "."))
    result = " ".join(sentences)
    if not result or RETROSPECTIVE.search(result) or STATUS_SUBJECT.search(result):
        raise ValueError(f"cannot render action-only decision: {text}")
    return result


def date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware: {value}")
    return parsed


def read_decisions(path: Path) -> list[dict]:
    decisions = json.loads(path.read_text())["decisions"]
    if not decisions or [d["time_step_id"] for d in decisions] != list(range(len(decisions))):
        raise ValueError(f"missing/nonsequential decisions: {path}")
    if len({d["decision_id"] for d in decisions}) != len(decisions):
        raise ValueError(f"duplicate decision IDs: {path}")
    starts = [date(d["first_date"]) for d in decisions]
    if starts != sorted(starts):
        raise ValueError(f"decisions not in first-date order: {path}")
    for decision, start in zip(decisions, starts):
        if (date(decision["last_date"]) < start or decision["category"] not in CATEGORIES
                or decision["outcome"] not in {"retained", "abandoned", "superseded"}):
            raise ValueError(f"invalid dates/category/outcome: {path}, {decision['decision_id']}")
    return decisions



def render_prefix(decisions: list[dict], step: int) -> str:
    if not 1 <= step < len(decisions):
        raise ValueError(f"target step outside trajectory: {step}")
    observed = decisions[:step]
    cutoff = date(decisions[step]["first_date"])
    by_id = {decision["decision_id"]: decision for decision in observed}
    blocks = []
    for decision in observed:
        if date(decision["last_date"]) >= cutoff:
            state = "active at cutoff"
        elif decision["outcome"] == "superseded":
            replacement = by_id.get(decision.get("superseded_by"))
            state = (f"superseded by observed T{replacement['time_step_id']:03d}"
                     if replacement else "retained by cutoff")
        else:
            state = decision["outcome"] + " by cutoff"
        blocks.append(
            f"- T{decision['time_step_id']:03d} | {decision['category']} | {state}\n"
            f"  {action_text(decision['decision'])}"
        )
    return (
        "# Evaluation trajectory prefix\n\n"
        "All decisions introduced before the held-out decision are shown in "
        "timezone-aware first-date order. Outcomes that were not established "
        "by the cutoff are withheld.\n\nObserved decisions:\n\n"
        + "\n\n".join(blocks)
        + "\n\nPredict the next research decision. Return the required JSON only."
    )
