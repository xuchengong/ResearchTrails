#!/usr/bin/env python3
"""Render selected annotations as evidence-free research trajectory demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
ANNOTATION_ROOT = REPO_ROOT / "annotations" / "neurips_2025"
SPLITS = REPO_ROOT / "annotate" / "harness_splits.json"
DEFAULT_OUTPUT = HERE / "prompts" / "demonstrations.md"
DEFAULT_SHUFFLED_OUTPUT = HERE / "prompts" / "shuffled_demonstrations.md"
DEFAULT_SHUFFLE_METADATA = HERE / "prompts" / "shuffled_demonstrations.metadata.json"
DEFAULT_SHUFFLE_SEED = 20260909
DEMONSTRATION_HEADER = """# Reference research decision trajectories

These records are inputs for learning a next-decision prediction procedure. Decision descriptions state the scientific action without retrospective outcome clauses. Evidence excerpts, commit metadata, abstracts, and dates are omitted. Derive portable guidance from the records without copying domain-specific solutions.
"""

HISTORY_CLAUSE_START = re.compile(
    r"^(?:"
    r"the earlier\b|the later\b|the experiment began\b|the model set expanded\b|"
    r"the protocol iterated\b|the run selection was\b|"
    r"both .*\bhistorically\b|this revises the earlier\b"
    r")",
    re.IGNORECASE,
)
RETROSPECTIVE_OUTCOME_MARKER = re.compile(
    r"(?:"
    r"\b(?:temporarily|briefly|initially|ultimately|finalized)\b|"
    r"\bfor later\b|"
    r",\s*with\s+the\s+(?:final|latest|retained|mature)\b|"
    r"\bremains?\s+operative\b|"
    r"\b(?:later|subsequently|eventually)\s+"
    r"(?:remov\w*|delet\w*|disabl\w*|discontinu\w*|abandon\w*|"
    r"supersed\w*|replac\w*|restor\w*|return\w*|revert\w*)\b|"
    r"\b(?:was|were|is|are)\s+(?:later|subsequently|eventually)\s+"
    r"(?:remov\w*|delet\w*|disabl\w*|discontinu\w*|abandon\w*|"
    r"supersed\w*|replac\w*|commented\s+out|restor\w*|return\w*|revert\w*)\b|"
    r"\b(?:then|before being)\s+"
    r"(?:remov\w*|delet\w*|disabl\w*|discontinu\w*|abandon\w*|"
    r"supersed\w*|replac\w*|restor\w*|return\w*|revert\w*)\b|"
    r"\b(?:branch|arm|suite|comparison|condition|computation|invocation|launcher|"
    r"route|implementation|baseline|experiment|protocol)\b.{0,100}\b"
    r"(?:was|were|is|are)\s+"
    r"(?:removed|deleted|disabled|discontinued|abandoned|superseded|commented\s+out)\b|"
    r"\b(?:was|were)\s+historically\s+(?:included|covered|tested|used)\b|"
    r"\b(?:was|were)\s+(?:\w+[\s,/.-]+){0,20}historically\b|"
    r"\b(?:included|covered|tested|reported|run)\s+historically\b"
    r")",
    re.IGNORECASE,
)
STATUS_SUBJECT = re.compile(
    r"^(?:the\s+)?(?:latest|retained|mature)\b.*\b"
    r"(?:retains?|removes?|drops?|excludes?|discontinues?|replaces?)\b",
    re.IGNORECASE,
)


def load_splits(path: Path = SPLITS) -> dict[str, tuple[int, ...]]:
    """The train, test, and prefix_length project indices in annotations/neurips_2025/."""
    data = json.loads(path.read_text(encoding="utf-8"))
    splits = {}
    for name in ("train", "test", "prefix_length"):
        indices = data[name]["neurips_2025"]
        if any(type(index) is not int for index in indices) or len(indices) != len(set(indices)):
            raise ValueError(f"{path} {name} must list distinct integer indices")
        splits[name] = tuple(indices)
    overlap = sorted(set(splits["train"]) & set(splits["test"]))
    if overlap:
        raise ValueError(f"{path} assigns indices to both train and test: {overlap}")
    outside = sorted(set(splits["prefix_length"]) - set(splits["test"]))
    if outside:
        raise ValueError(f"{path} lists prefix_length indices outside test: {outside}")
    return splits


SPLIT_INDICES = load_splits()
TRAIN_INDICES = SPLIT_INDICES["train"]
TEST_INDICES = SPLIT_INDICES["test"]
PREFIX_LENGTH_INDICES = SPLIT_INDICES["prefix_length"]


def decision_action_text(text: str) -> str:
    """Remove prose that retrospectively states how a decision later fared."""
    sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        clauses = []
        for clause in re.split(r";\s+", sentence):
            clause = re.sub(
                r"^(?:Temporarily|Initially|Briefly|Ultimately)\s+",
                "",
                clause,
                flags=re.IGNORECASE,
            )
            if HISTORY_CLAUSE_START.search(clause) or STATUS_SUBJECT.search(clause):
                continue

            marker = RETROSPECTIVE_OUTCOME_MARKER.search(clause)
            if marker:
                prefix = clause[: marker.start()].rstrip(" ,;:\u2014-")
                if (
                    len(prefix.split()) < 6
                    or HISTORY_CLAUSE_START.search(prefix)
                    or re.match(
                        r"^(?:the|this|that|these|those|a|an|both)\b",
                        clause,
                        re.IGNORECASE,
                    )
                ):
                    continue
                clause = prefix

            clause = re.sub(
                r"\b(?:latest|retained|mature)\s+(?!(?:and|or)\b)",
                "",
                clause,
                flags=re.IGNORECASE,
            )
            clause = re.sub(
                r"\bfinal\s+(?=(?:pipeline|protocol|configuration|variant|"
                r"comparison|suite|route|supplied objective|three signals)\b)",
                "",
                clause,
                flags=re.IGNORECASE,
            )
            clause = re.sub(
                r"\b(?:historical|historically|subsequent)\s+",
                "",
                clause,
                flags=re.IGNORECASE,
            )
            clause = re.sub(r"\s{2,}", " ", clause).strip(" ,;")
            if clause:
                clauses.append(clause)

        if clauses:
            normalized = "; ".join(clauses)
            normalized = normalized[0].upper() + normalized[1:]
            if normalized[-1] not in ".!?":
                normalized += "."
            sentences.append(normalized)

    result = " ".join(sentences)
    if not result:
        raise ValueError(f"decision has no action after removing outcome history: {text}")
    if RETROSPECTIVE_OUTCOME_MARKER.search(result) or STATUS_SUBJECT.search(result):
        raise ValueError(f"decision still contains retrospective outcome wording: {result}")
    return result


def annotation_path(index: int) -> Path:
    matches = sorted(ANNOTATION_ROOT.glob(f"{index}-*/annotation.json"))
    if len(matches) != 1:
        raise ValueError(f"idx={index} requires exactly one annotation.json; found {len(matches)}")
    return matches[0]


def render_annotation(index: int) -> str:
    path = annotation_path(index)
    annotation = json.loads(path.read_text(encoding="utf-8"))
    decisions = annotation.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError(f"{path} has no decisions")
    expected_steps = list(range(len(decisions)))
    actual_steps = [decision.get("time_step_id") for decision in decisions]
    if actual_steps != expected_steps:
        raise ValueError(f"{path} has stale or nonsequential time_step_id values")

    lines = [
        f"## Demonstration idx={index}",
        "",
        f"Project: {annotation['title']}",
        "",
        f"Trajectory insight: {annotation['trajectory_insight']}",
        "",
        "Decisions:",
        "",
    ]
    for decision in decisions:
        outcome = decision["outcome"]
        if decision.get("superseded_by"):
            outcome += f" by {decision['superseded_by']}"
        lines.extend(
            [
                f"- T{decision['time_step_id']:03d} {decision['decision_id']} "
                f"| {decision['category']} | {outcome}",
                f"  {decision_action_text(decision['decision'])}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def render_demonstrations() -> str:
    sections = [render_annotation(index) for index in TRAIN_INDICES]
    return DEMONSTRATION_HEADER.rstrip() + "\n\n" + "\n\n".join(sections) + "\n"


def derived_seed(seed: int, index: int, purpose: str) -> int:
    digest = hashlib.sha256(f"{seed}:{index}:{purpose}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def remap_identifiers(
    text: str,
    decision_id_map: dict[str, str],
    time_step_id_map: dict[int, str],
) -> str:
    def replace_decision(match: re.Match) -> str:
        original = match.group(0)
        if original not in decision_id_map:
            raise ValueError(f"decision text refers to unknown decision ID {original}")
        return decision_id_map[original]

    def replace_time_step(match: re.Match) -> str:
        original = int(match.group(1))
        if original not in time_step_id_map:
            raise ValueError(f"decision text refers to unknown time-step ID {match.group(0)}")
        return time_step_id_map[original]

    text = re.sub(r"\bD\d{3}\b", replace_decision, text)
    return re.sub(r"\bT(\d{2,3})\b", replace_time_step, text)


def render_shuffled_annotation(index: int, seed: int) -> tuple[str, dict]:
    path = annotation_path(index)
    annotation = json.loads(path.read_text(encoding="utf-8"))
    decisions = annotation.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError(f"{path} has no decisions")

    order_seed = derived_seed(seed, index, "order")
    permutation = list(range(len(decisions)))
    random.Random(order_seed).shuffle(permutation)
    if permutation == list(range(len(decisions))):
        permutation = permutation[1:] + permutation[:1]

    decision_id_map = {
        decisions[original_position]["decision_id"]: f"D{shuffled_position + 1:03d}"
        for shuffled_position, original_position in enumerate(permutation)
    }
    time_step_id_map = {
        decisions[original_position]["time_step_id"]: f"T{shuffled_position:03d}"
        for shuffled_position, original_position in enumerate(permutation)
    }
    outcome_seed = derived_seed(seed, index, "outcome")
    outcome_rng = random.Random(outcome_seed)
    supersession_seed = derived_seed(seed, index, "supersession")
    supersession_rng = random.Random(supersession_seed)
    randomized_outcomes = {}
    randomized_superseded_by = {}
    new_ids = [f"D{position + 1:03d}" for position in range(len(permutation))]
    for shuffled_position, new_id in enumerate(new_ids):
        if outcome_rng.random() < 0.5:
            outcome = "retained"
        elif shuffled_position == len(new_ids) - 1:
            outcome = "abandoned"
        else:
            outcome = outcome_rng.choice(("abandoned", "superseded"))
        randomized_outcomes[new_id] = outcome
        if outcome == "superseded":
            randomized_superseded_by[new_id] = supersession_rng.choice(
                new_ids[shuffled_position + 1 :]
            )

    lines = [
        f"## Demonstration idx={index}",
        "",
        f"Project: {annotation['title']}",
        "",
        "Decisions:",
        "",
    ]
    for shuffled_position, original_position in enumerate(permutation):
        decision = decisions[original_position]
        new_id = decision_id_map[decision["decision_id"]]
        outcome = randomized_outcomes[new_id]
        if outcome == "superseded":
            outcome += f" by {randomized_superseded_by[new_id]}"
        lines.extend(
            [
                f"- T{shuffled_position:03d} {new_id} "
                f"| {decision['category']} | {outcome}",
                "  "
                + remap_identifiers(
                    decision_action_text(decision["decision"]),
                    decision_id_map,
                    time_step_id_map,
                ),
                "",
            ]
        )

    metadata = {
        "source_annotation": path.relative_to(REPO_ROOT).as_posix(),
        "order_seed": order_seed,
        "outcome_seed": outcome_seed,
        "supersession_seed": supersession_seed,
        "shuffled_position_to_original_time_step_id": [
            decisions[position]["time_step_id"] for position in permutation
        ],
        "original_decision_id_to_shuffled_decision_id": decision_id_map,
        "randomized_outcomes": randomized_outcomes,
        "randomized_superseded_by": randomized_superseded_by,
    }
    return "\n".join(lines).rstrip(), metadata


def render_shuffled_demonstrations(seed: int) -> tuple[str, dict]:
    sections = []
    trajectories = {}
    for index in TRAIN_INDICES:
        section, metadata = render_shuffled_annotation(index, seed)
        sections.append(section)
        trajectories[str(index)] = metadata
    rendered = DEMONSTRATION_HEADER.rstrip() + "\n\n" + "\n\n".join(sections) + "\n"
    metadata = {
        "format_version": 2,
        "seed": seed,
        "training_indices_source": SPLITS.relative_to(REPO_ROOT).as_posix(),
        "seed_derivation": "uint64_be(sha256(f'{seed}:{index}:{purpose}')[:8])",
        "permutation_scope": "independent within each training trajectory",
        "decision_id_rule": "D001..Dnnn assigned in shuffled presentation order",
        "time_step_id_rule": "T000..Tnnn assigned in shuffled presentation order",
        "decision_text_rule": "scientific action only; retrospective outcome clauses removed before identifier remapping",
        "outcome_rule": "50% retained; otherwise equally likely abandoned or superseded, except the final decision is abandoned when non-retained",
        "supersession_rule": "each randomly superseded decision targets a seeded random decision later in shuffled presentation order",
        "trajectories": trajectories,
    }
    return rendered, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shuffled", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_SHUFFLE_METADATA)
    args = parser.parse_args()
    if args.shuffled:
        output = args.output or DEFAULT_SHUFFLED_OUTPUT
        rendered, metadata = render_shuffled_demonstrations(args.seed)
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote shuffle metadata to {args.metadata_output}")
    else:
        output = args.output or DEFAULT_OUTPUT
        rendered = render_demonstrations()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"wrote {len(TRAIN_INDICES)} demonstrations to {output}")


if __name__ == "__main__":
    main()
