"""The GRPO reward: Sol judges each sampled decision against the training target.

Judge calls go through OpenRouter in the harness's request format. Each response is
saved to its own file, so an interrupted batch resumes without paying twice.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import sys

from sft.io import ROOT, atomic_json, json_text, write_once

sys.path.insert(0, str(ROOT / "harness"))
import experiment


def score_groups(directory, rows, rollouts, config, instructions):
    """Judge unique valid answers once; preserve all sampled group members.

    Invalid JSON and truncated completions receive zero reward. Duplicates share
    a judgment, so judge randomness cannot create preferences between identical
    decision answers. Even one unique valid answer is judged.
    """
    by_id = {row["case_id"]: row for row in rows}
    requests, identities, reward_sources = [], {}, {}
    for case_index, rollout in enumerate(rollouts):
        row = by_id[rollout["case_id"]]
        unique = {}
        for candidate in rollout["candidates"]:
            identity = (row["case_id"], candidate["candidate_id"])
            if not candidate["token_ids"]:
                raise ValueError(f"empty GRPO completion: {identity}")
            if not candidate["finished"] or candidate["prediction"] is None:
                reward_sources[identity] = None
                continue
            key = json.dumps(candidate["prediction"], sort_keys=True, ensure_ascii=False)
            if key not in unique:
                unique[key] = identity
                custom_id = f"case-{case_index:04d}-candidate-{candidate['candidate_id']:02d}"
                identities[custom_id] = identity
                case = {"case_id": row["case_id"], "target": row["target"],
                        "evaluation_input": row["messages"][1]["content"]}
                requests.append({
                    "custom_id": custom_id, "method": "POST", "url": "/v1/responses",
                    "body": experiment.response_body(
                        config["model"], instructions.strip(),
                        experiment.render_judge_input(case, candidate["prediction"]),
                        "individual_prediction_judgment", experiment.JUDGE_SCHEMA,
                        config["max_output_tokens"], config["reasoning_effort"], config["routing"],
                    ),
                })
            reward_sources[identity] = unique[key]
    scores, usage = execute_judgments(directory, requests, identities, config)
    rewards = {identity: 0.0 if source is None else scores[source] for identity, source in reward_sources.items()}
    write_once(directory / "rewards.json", json_text([
        {"case_id": identity[0], "candidate_id": identity[1], "reward": rewards[identity],
         "judged_candidate_id": source[1] if source else None,
         "reward_source": "judge" if source else "invalid_or_truncated"}
        for identity, source in reward_sources.items()
    ]))
    return rewards, usage


def execute_judgments(directory, requests, identities, config):
    """Resume paid requests, persist responses atomically, and account their cost."""
    write_once(directory / "judge_requests.json", json_text(requests))
    response_dir = directory / "judge_responses"
    response_dir.mkdir(exist_ok=True)
    unknown = {p.stem for p in response_dir.glob("*.json")} - set(identities)
    if unknown:
        raise ValueError(f"unknown saved judge responses: {unknown}")
    responses, pending = {}, []
    for request in requests:
        path = response_dir / (request["custom_id"] + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["custom_id"] != request["custom_id"]:
                raise ValueError(f"saved judge response identity differs: {path}")
            text, _ = experiment.extract_response_text(saved)
            experiment.validate_judgment_response(request["custom_id"], text)
            responses[request["custom_id"]] = saved
        else:
            pending.append(request)
    if pending:
        key = experiment.api_key("openrouter")
        failures = []
        with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
            futures = {pool.submit(
                experiment.execute_request, request, "openrouter", key,
                config["max_attempts"], experiment.validate_judgment_response,
            ): request["custom_id"] for request in pending}
            for future in as_completed(futures):
                custom_id = futures[future]
                try:
                    response = future.result()
                    atomic_json(response_dir / (custom_id + ".json"), response)
                    responses[custom_id] = response
                except Exception as exc:
                    failures.append({"custom_id": custom_id, "error": str(exc)})
        if failures:
            atomic_json(directory / "judge_errors.json", failures)
            raise RuntimeError(f"{len(failures)} judgments failed; resume to retry missing responses: {directory}")
    scores, judgments, costs = {}, [], []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for custom_id, identity in identities.items():
        response = responses[custom_id]
        if response["custom_id"] != custom_id:
            raise ValueError(f"saved judge response identity differs: {custom_id}")
        text, consumed = experiment.extract_response_text(response)
        cost = consumed.get("cost")
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            raise ValueError(f"missing or invalid usage.cost in saved judge response: {response_dir / (custom_id + '.json')}")
        costs.append(cost)
        judgment = json.loads(text)
        experiment.validate_judgment(custom_id, judgment)
        score = experiment.headline_score(judgment)
        scores[identity] = score
        judgments.append({"case_id": identity[0], "candidate_id": identity[1],
                          "score": score, **judgment})
        for name in usage:
            usage[name] += int(consumed.get(name, 0))
    write_once(directory / "judgments.json", json_text(judgments))
    return scores, {"responses": len(responses), **usage, "cost_usd": math.fsum(costs)}
