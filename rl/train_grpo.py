"""Online GRPO from the merged SFT model, on one GPU or several.

Run it under torchrun (`rl/run_grpo.sh` uses every visible GPU) or directly with
one visible GPU. The world size is deliberately absent from `run.json`: the
experiment is defined by its data, hyperparameters and seeds, not by how many GPUs
executed it, so one run directory can be resumed on a different number of GPUs.
Each collection round records the world size that produced it in `metrics.json`.
`--continue-from` keeps training an exported epoch adapter instead of attaching a
fresh LoRA.
"""

import argparse
import fcntl
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import signal
import time

from peft import get_peft_model_state_dict
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

from rl import grpo
from rl.core import activate, prompt_tokens, sample_candidates
from rl.reward import experiment, score_groups
from rl.modeling import initialize_policy
from sft.io import ROOT, atomic_json, write_once, json_text, load_split
from sft.modeling import model_dtype


DEFAULT_SFT_CHECKPOINT = ROOT / "sft/runs/qwen3-8b/seed-42/checkpoint-72"


def prepare(args):
    """Resolve and record the SFT initialization, training-only data and any continued adapter."""
    if min(args.epochs, args.batch_size, args.gradient_accumulation_steps, args.candidates, args.max_new_tokens,
           args.judge_concurrency, args.judge_max_attempts, args.judge_max_output_tokens) < 1:
        raise ValueError("epochs, batch size, gradient accumulation steps, candidate count and request limits must be positive")
    if args.candidates < 2 or not 0 < args.clip_epsilon < 1:
        raise ValueError("need at least two candidates and clip-epsilon in (0, 1)")
    if not math.isfinite(args.beta) or args.beta < 0:
        raise ValueError("KL beta must be finite and nonnegative")
    if not all(math.isfinite(x) and x > 0 for x in
               (args.learning_rate, args.temperature, args.max_grad_norm, args.length_normalizer)):
        raise ValueError("learning rate, temperature, gradient norm and length normalizer must be finite and positive")
    if (args.max_length is not None and args.max_length < 1) or not args.judge_model.strip():
        raise ValueError("max-length must be positive and judge-model must be nonempty")
    initial = args.sft_checkpoint.resolve()
    source = initial.parent
    original = json.loads((source / "run.json").read_text())
    info = json.loads((source / "model_info.json").read_text())
    json.loads((source / "completed.json").read_text())  # Require a finished SFT run.
    adapter_config = json.loads((initial / "adapter_config.json").read_text())
    if (adapter_config["peft_type"] != "LORA" or adapter_config.get("bias", "none") != "none"
            or adapter_config.get("modules_to_save") or adapter_config.get("use_dora")
            or adapter_config.get("use_rslora") or adapter_config.get("rank_pattern")
            or adapter_config.get("alpha_pattern")):
        raise ValueError("requires the SFT pipeline's standard fixed-rank LoRA adapter with frozen base weights")
    data_dir = (args.data_dir or ROOT / original["data_dir"]).resolve()
    condition = original["condition"]
    rows = load_split(data_dir, "train_" + condition)
    projects = {row["project_id"] for row in rows}
    manifest = json.loads((data_dir / "manifest.json").read_text())
    for name in manifest["datasets"]:
        if name.startswith("eval_"):
            if projects & {row["project_id"] for row in load_split(data_dir, name)}:
                raise ValueError(f"training and {name} projects overlap")
    for row in rows:
        if ([m["role"] for m in row["messages"]] != ["system", "user"]
                or not row["target"]["decision"].strip()
                or row["target"]["category"] not in experiment.TARGET_CATEGORIES):
            raise ValueError(f"malformed training example: {row['case_id']}")
        experiment.render_judge_input({
            "case_id": row["case_id"], "target": row["target"],
            "evaluation_input": row["messages"][1]["content"],
        }, row["target"])
    # The final adapter's saved tokenizer is shared by all SFT epoch checkpoints.
    tokenizer_dir = source / "adapter"
    if not any(p.name.startswith(("tokenizer", "special_tokens", "chat_template"))
               for p in tokenizer_dir.iterdir()):
        raise ValueError("SFT run has no saved tokenizer")
    output = args.output_dir.resolve()
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output)
           for p in (source, data_dir)):
        raise ValueError("GRPO output must be separate from SFT results and datasets")
    instructions = (ROOT / "harness/prompts/judge_rubrics_long.md").read_text()
    judge = {
        "model": args.judge_model, "reasoning_effort": args.judge_reasoning_effort,
        "max_output_tokens": args.judge_max_output_tokens, "concurrency": args.judge_concurrency,
        "max_attempts": args.judge_max_attempts,
        "routing": experiment.provider_preferences(
            "openrouter", args.judge_max_prompt_price, args.judge_max_completion_price, False,
        ),
        "score": "component_match + operation_match", "schema": experiment.JUDGE_SCHEMA,
    }
    config = {
        "algorithm": "grpo", "sft_checkpoint": str(initial), "sft_run": str(source),
        "tokenizer_dir": str(tokenizer_dir), "data_dir": str(data_dir),
        "condition": condition, "cases": len(rows), "projects": len(projects),
        "adapter_mode": "merged_sft_fresh_lora",
        "merged_base": {
            "model": info["base_model"], "revision": info["resolved_revision"],
            "sft_checkpoint": str(initial), "precision": args.precision,
        },
        "fresh_lora": {
            "task_type": "CAUSAL_LM", "r": adapter_config["r"],
            "lora_alpha": adapter_config["lora_alpha"],
            "target_modules": adapter_config["target_modules"],
            "lora_dropout": 0.0, "bias": "none", "init_lora_weights": True,
        },
        "enable_thinking": False,
        "epochs": args.epochs, "batch_size": args.batch_size, "candidates": args.candidates,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "batch_unit": "prefix groups with nonzero reward variance",
        "invalid_reward": 0.0,
        "advantage": "group rewards minus the group mean; no standard-deviation scaling",
        "num_iterations": 1, "loss_type": "dr_grpo",
        "length_normalizer": args.length_normalizer,
        "final_remainder": "discard; carry pending gradients across earlier epoch boundaries",
        "seed": args.seed, "learning_rate": args.learning_rate, "beta": args.beta,
        "temperature": args.temperature, "top_p": 1.0, "max_new_tokens": args.max_new_tokens,
        "max_length": args.max_length or original["max_length"],
        "clip_epsilon": args.clip_epsilon, "max_grad_norm": args.max_grad_norm,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "device": args.device, "judge": judge,
        "lr_scheduler_type": "constant", "dropout": 0.0,
        "objective": "Dr. GRPO: mean-centered group rewards; completion-token sum over a constant normalizer; fixed SFT reference KL; temperature-scaled policy",
        "packages": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
    }
    if args.continue_from:
        config = grpo.continue_from(config, args.continue_from)
    return config, rows, instructions


def training_order(rows, config):
    """One deterministic shuffled pass per epoch; collection sizes are adaptive."""
    examples = []
    for epoch in range(1, config["epochs"] + 1):
        order = list(range(len(rows)))
        random.Random(config["seed"] + epoch).shuffle(order)
        examples.extend(rows[i] for i in order)
    return examples


def save_resume(path, model, optimizer, cursor, updates, history, accumulation):
    """Commit collected prefixes and pending gradients between optimizer updates."""
    state = {
        "policy": {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()},
        "optimizer": optimizer.state_dict(), "cursor": cursor, "updates": updates,
        "history": history, "torch_rng": torch.get_rng_state(),
        "accumulation": accumulation,
        "gradients": {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()
                      if p.grad is not None},
        "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def export_adapter(path, model, tokenizer, metadata, base_spec):
    """Keep all epoch adapters, but only one rolling optimizer checkpoint."""
    complete = path / "export_complete.json"
    if complete.exists():
        if json.loads(complete.read_text()) != metadata:
            raise ValueError(f"adapter export metadata changed: {path}")
        return
    path.mkdir(parents=True, exist_ok=True)
    activate(model)
    model.save_pretrained(path, selected_adapters=["default"], save_embedding_layers=False)
    tokenizer.save_pretrained(path)
    # This adapter must be applied to the merged SFT base, not original Qwen.
    write_once(path / "dpo_base.json", json_text(base_spec))
    atomic_json(complete, metadata)


def log_epoch_judge_cost(output, history, epoch):
    """Rebuild completed-epoch costs from committed batches, including on resume."""
    epochs = []
    for number in range(1, epoch + 1):
        usage = [record["judge_usage"] for record in history if record["epoch"] == number]
        epochs.append({"epoch": number, "responses": sum(item["responses"] for item in usage),
                       "cost_usd": math.fsum(item["cost_usd"] for item in usage)})
    total = math.fsum(item["cost_usd"] for item in epochs)
    atomic_json(output / "judge_costs.json", {"epochs": epochs, "total_cost_usd": total})
    current = epochs[-1]
    tqdm.write(f"Epoch {epoch} judge API cost: ${current['cost_usd']:.6f} "
               f"({current['responses']} saved responses); cumulative: ${total:.6f}")


def train(args):
    rank, size = grpo.start(args.device)
    lead = rank == 0
    try:
        config, rows, instructions = prepare(args)
        output = args.output_dir.resolve()
        if lead:
            output.mkdir(parents=True, exist_ok=True)
        grpo.barrier(size)
        lock = (output / ".lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB) if lead else None
        if lead:
            existing = [p for p in output.iterdir() if p.name != ".lock"]
            if existing and not (args.resume or args.prepare_only):
                raise ValueError("output directory is not empty; use --resume or choose a new directory")
            write_once(output / "run.json", json_text(config))
            write_once(output / "judge_rubrics_long.md", instructions)
            write_once(output / "model_info.json", json_text({
                "merged_base": config["merged_base"], "enable_thinking": False,
            }))
        grpo.barrier(size)
        examples = training_order(rows, config)
        target_groups = config["batch_size"] * config["gradient_accumulation_steps"]
        say = tqdm.write if lead else (lambda *a, **k: None)
        if lead:
            print(f"Online GRPO on {size} rank(s): {len(rows)} training prefixes per epoch, "
                  f"{config['epochs']} epochs; up to "
                  f"{len(rows) * config['epochs'] * config['candidates']} candidate judgments.", flush=True)
            print(f"Each optimizer update requires {target_groups} informative groups; "
                  f"generation and backward are sharded, gradients summed then averaged once.", flush=True)
            lineage = config.get("initialized_from")
            print(f"Policy: continuing {lineage['checkpoint']} "
                  f"(epoch {lineage['source_epoch']}, {lineage['source_updates']} updates at "
                  f"LR {lineage['source_learning_rate']:g}) at LR {config['learning_rate']:g}; "
                  "KL reference stays the merged SFT base." if lineage else
                  f"Policy: fresh LoRA on the merged SFT base at LR {config['learning_rate']:g}.",
                  flush=True)
        if (output / "completed.json").exists():
            if lead:
                print(f"Already complete: {output / 'adapter'}")
            return
        if args.prepare_only or not args.confirm_submit:
            if lead:
                print("Prepared only; no model loaded and no paid calls. To train, rerun "
                      "rl/run_grpo.sh without --dry-run (or add --resume --confirm-submit).")
            return

        experiment.api_key("openrouter")
        model_dtype(config["merged_base"]["precision"], config["device"])
        set_seed(config["seed"])
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_dir"], local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompts = {row["case_id"]: prompt_tokens(row, tokenizer) for row in rows}
        if any(len(t) + config["max_new_tokens"] > config["max_length"] for t in prompts.values()):
            raise ValueError("a prompt + max-new-tokens exceeds max-length; increase max-length")
        # A continued run keeps training an exported adapter; a fresh one
        # attaches zero-output factors to the same frozen merged base.
        model = (grpo.load_continued_policy(config) if "initialized_from" in config
                 else initialize_policy(config))
        if config["max_length"] > model.config.max_position_embeddings:
            raise ValueError("max-length exceeds the model context window")
        # start() narrowed this rank to a single visible GPU, so it is cuda:0.
        model.to("cuda:0" if config["device"] == "cuda" else config["device"])
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        if config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        activate(model, training=True)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config["learning_rate"], weight_decay=0.0)
        resume_path = output / "latest.pt"
        cursor, updates, history = 0, 0, []
        accumulation = {"groups": 0, "skipped": 0, "training_seconds": 0.0}
        optimizer.zero_grad(set_to_none=True)
        if not resume_path.exists() and lead:
            # Adam's moments describe the policy being continued, so they carry
            # over; every rank then picks them up from this first checkpoint.
            if "initialized_from" in config:
                grpo.adopt_optimizer(optimizer, grpo.continued_optimizer_state(config),
                                              config["learning_rate"])
            save_resume(resume_path, model, optimizer, 0, 0, [], accumulation)
        grpo.barrier(size)
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        cursor, updates, history, accumulation = grpo.restore(state, model, optimizer, rank)
        if not 0 <= cursor <= len(examples) or sum(r["prefixes"] for r in history) != cursor:
            raise ValueError("invalid resume position")
        if not 0 <= accumulation["groups"] < target_groups or accumulation["skipped"] < 0:
            raise ValueError("invalid saved gradient accumulation counts")
        total_cost = math.fsum(r["judge_usage"]["cost_usd"] for r in history)
        if lead and cursor and cursor % len(rows) == 0:
            epoch = cursor // len(rows)
            export_adapter(output / f"checkpoint-epoch-{epoch}", model, tokenizer,
                           {"epoch": epoch, "batches": len(history), "updates": updates},
                           config["merged_base"])
            log_epoch_judge_cost(output, history, epoch)
        halt = False

        def request_stop(signum, frame):
            nonlocal halt
            halt = True
            say(f"Signal {signum}: will stop after committing the current collection round.")

        handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
        progress = tqdm(total=target_groups, initial=accumulation["groups"], unit="group",
                        desc=f"Update {updates + 1}: accepted", dynamic_ncols=True,
                        disable=not lead or cursor == len(examples))
        try:
            while cursor < len(examples):
                batch_started = time.perf_counter()
                batch_id = len(history)
                epoch = cursor // len(rows) + 1
                count = min(config["batch_size"], target_groups - accumulation["groups"],
                            epoch * len(rows) - cursor)
                batch_rows = examples[cursor:cursor + count]
                epoch_end = cursor + count == epoch * len(rows)
                progress.set_postfix(epoch=epoch, skipped=accumulation["skipped"],
                                     usd=f"{total_cost:.4f}", phase="generating")
                directory = output / "batches" / f"{batch_id:06d}"
                if lead:
                    directory.mkdir(parents=True, exist_ok=True)
                grpo.barrier(size)
                rollout_path = directory / "rollouts.json"
                if rollout_path.exists():
                    rollouts = json.loads(rollout_path.read_text())
                    if [r["case_id"] for r in rollouts] != [r["case_id"] for r in batch_rows]:
                        raise ValueError(f"saved rollout batch differs: {directory}")
                else:
                    rollouts = grpo.sample_shard(sample_candidates, model, tokenizer, batch_rows,
                                                    prompts, batch_id, config, rank, size)
                    if lead:
                        atomic_json(rollout_path, rollouts)
                if any(len(r["candidates"]) != config["candidates"] for r in rollouts):
                    raise ValueError(f"saved GRPO group size differs: {directory}")
                progress.set_postfix(epoch=epoch, skipped=accumulation["skipped"],
                                     usd=f"{total_cost:.4f}", phase="judging")
                # One rank pays for and records the judgments; the rest wait.
                # One rank scores and selects, then broadcasts the decision itself.
                # Recomputing the selection per rank would make every rank's entry
                # into the dense reduction depend on four independent evaluations
                # agreeing; broadcasting makes them identical by construction.
                selection = None
                if lead:
                    rewards, usage = score_groups(directory, batch_rows, rollouts,
                                                  config["judge"], instructions)
                    selection = (rewards, usage, *grpo.select_groups(rollouts, rewards),
                                 [max(rewards[(r["case_id"], c["candidate_id"])]
                                      for c in r["candidates"]) for r in rollouts],
                                 sum(not c["finished"] or c["prediction"] is None
                                     for r in rollouts for c in r["candidates"]))
                rewards, usage, groups, skipped, best_of_k, invalid = grpo.share(selection, size)
                total_cost += usage["cost_usd"]
                skipped_in_update = accumulation["skipped"] + skipped
                progress.update(len(groups))
                progress.set_postfix(epoch=epoch, skipped=skipped_in_update,
                                     usd=f"{total_cost:.4f}", phase="training")
                if lead:
                    write_once(directory / "groups.json", json_text([
                        {"case_id": g["case_id"], "rewards": g["rewards"],
                         "advantages": g["advantages"], "reward_std": g["reward_std"]}
                        for g in groups]))
                window_end = accumulation["groups"] + len(groups) == target_groups
                metrics = grpo.update_policy(model, optimizer, groups, prompts, config,
                                                 rank, size,
                                                 accumulated_groups=accumulation["groups"],
                                                 step=window_end)
                if config["device"] == "cuda":
                    torch.cuda.synchronize()
                training_seconds = time.perf_counter() - batch_started
                update_training_seconds = accumulation["training_seconds"] + training_seconds
                accumulation = {"groups": metrics["accumulated_groups"],
                                "skipped": 0 if window_end else skipped_in_update,
                                "training_seconds": 0.0 if window_end else update_training_seconds}
                cursor += count
                updates += int(metrics["updated"])
                discarded = 0
                if cursor == len(examples):
                    discarded = accumulation["groups"]
                    optimizer.zero_grad(set_to_none=True)
                    accumulation = {"groups": 0, "skipped": 0, "training_seconds": 0.0}
                    metrics["accumulated_groups"] = 0
                record = {
                    "batch": batch_id, "epoch": epoch, "updates": updates,
                    "prefixes": len(batch_rows), "groups": len(groups), "skipped": skipped,
                    "mean_candidate_score": sum(rewards.values()) / len(rewards),
                    "best_of_k_score": sum(best_of_k) / len(best_of_k),
                    "invalid_candidates": invalid,
                    "mean_reward_std": sum(g["reward_std"] for g in groups) / len(groups) if groups else 0.0,
                    "judge_usage": usage, "learning_rate": config["learning_rate"], **metrics,
                    "training_seconds": training_seconds, "world_size": size,
                }
                if window_end:
                    record["update_training_seconds"] = update_training_seconds
                if discarded:
                    record["discarded_groups"] = discarded
                history.append(record)
                if lead:
                    save_resume(resume_path, model, optimizer, cursor, updates, history, accumulation)
                    atomic_json(directory / "metrics.json", record)
                    atomic_json(output / "history.json", history)
                    if epoch_end:
                        export_adapter(output / f"checkpoint-epoch-{epoch}", model, tokenizer,
                                       {"epoch": epoch, "batches": batch_id + 1, "updates": updates},
                                       config["merged_base"])
                        log_epoch_judge_cost(output, history, epoch)
                say(f"Epoch {epoch}, collection {batch_id + 1}, prefix {cursor % len(rows) or len(rows)}/{len(rows)}: "
                    f"{len(groups)}/{len(batch_rows)} groups, loss={metrics['grpo_loss']}, updates={updates}, "
                    f"reward={record['mean_candidate_score']:.3f}/best-of-k={record['best_of_k_score']:.3f}, "
                    f"invalid={invalid}, gap={metrics['advantage_logp_gap']}, kl={metrics['kl']}, "
                    f"ranks={size}, total_cost=${total_cost:.4f}, "
                    f"batch_time={time.perf_counter() - batch_started:.1f}s")
                if window_end:
                    say(f"Update {updates}: {target_groups} accepted, {skipped_in_update} skipped; "
                        f"training_time={update_training_seconds:.1f}s")
                    if cursor < len(examples):
                        progress.reset()
                        progress.set_description(f"Update {updates + 1}: accepted")
                if discarded:
                    say(f"Discarded {discarded}/{target_groups} informative groups in the final batch.")
                # Every rank must leave the loop together or the next collective hangs.
                if grpo.share(bool(halt) if lead else None, size):
                    if cursor < len(examples):
                        say("Stopped after committing collected prefixes and gradients; "
                            "rerun the same command to resume.")
                        return
        finally:
            progress.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        if not updates:
            raise RuntimeError(f"only {sum(r['groups'] for r in history)} informative groups in the "
                               f"entire run; need {target_groups} for a full optimizer update")
        if lead:
            atomic_json(output / "history.json", history)
            export_adapter(output / "adapter", model, tokenizer,
                           {"epoch": config["epochs"], "batches": len(history), "updates": updates},
                           config["merged_base"])
            atomic_json(output / "completed.json", {
                "adapter": "adapter", "epochs": config["epochs"], "batches": len(history),
                "updates": updates,
                "accepted_groups": sum(r["groups"] for r in history),
                "skipped_prefixes": sum(r["skipped"] for r in history),
                "discarded_groups": sum(r.get("discarded_groups", 0) for r in history)})
            print(f"Saved online GRPO adapter: {output / 'adapter'}", flush=True)
    finally:
        grpo.stop(size)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-checkpoint", type=Path, default=DEFAULT_SFT_CHECKPOINT,
                        help="SFT checkpoint or adapter; default: the seed-42 SFT run's epoch 3 (checkpoint-72)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, help="defaults to the SFT run's dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="informative prefix groups per collection; forwards use one completion at a time")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="effective prefix groups = batch-size times this value")
    parser.add_argument("--candidates", type=int, default=8, help="sampled completions per prefix group")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.04, help="fixed SFT reference KL coefficient; zero disables KL")
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--length-normalizer", type=float, default=64,
                        help="constant divisor for the summed completion-token loss; "
                             "set near the observed completion length, not the generation cap")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-length", type=int, help="defaults to the SFT run's max-length")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--judge-model", default="openai/gpt-5.6-sol:floor")
    parser.add_argument("--judge-reasoning-effort", default="medium")
    parser.add_argument("--judge-concurrency", type=int, default=16)
    parser.add_argument("--judge-max-attempts", type=int, default=5)
    parser.add_argument("--judge-max-output-tokens", type=int, default=3000)
    parser.add_argument("--judge-max-prompt-price", type=float, default=4)
    parser.add_argument("--judge-max-completion-price", type=float, default=20)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-submit", action="store_true", help="enable GPU training and paid OpenRouter judging")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-from", type=Path,
                        help="exported GRPO epoch checkpoint to keep training in place; "
                             "omit to attach a fresh LoRA to the merged SFT base")
    return parser


if __name__ == "__main__":
    train(argument_parser().parse_args())
