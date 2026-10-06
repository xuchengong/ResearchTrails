"""Group-relative, completion-token GRPO with a fixed merged-SFT reference.

Training runs on one GPU or, under torchrun, data-parallel over several. Ranks
shard prefixes for generation and informative groups for the backward pass.
Candidate seeds depend only on (seed, batch id, case id, candidate id), never on
rank or position, so a sharded rollout is the same rollout one process would have
drawn. Partial gradients are summed across ranks and then divided by the global
group count, which reproduces the single-process mean exactly; every rank steps
from the same reduced gradient, so the replicas never drift apart and any one of
them can be checkpointed. Only rank 0 calls the paid judge and writes to the run
directory. Launched without torchrun, the run is a single rank.

A run can also continue an exported GRPO adapter instead of attaching a fresh one.
The frozen base is unchanged: the original Qwen3 revision with the SFT LoRA merged
into its weights. Only the adapter on top differs, and the KL reference stays the
merged SFT model, so beta keeps measuring drift from SFT and the logged KL stays
comparable across runs. `adapter_mode` keeps its value because `rl.predict` and
`rl.evaluate` gate on it; the lineage lives in `initialized_from`.
"""

from datetime import timedelta
import json
import math
import os
from pathlib import Path

from peft import PeftModel, set_peft_model_state_dict
import torch
import torch.distributed as dist
import torch.nn.functional as F

from rl.core import activate
from rl.modeling import load_merged_sft


def group_advantages(rewards):
    """Mean-centered rewards (Dr. GRPO); identical rewards carry no signal.

    Dividing by the group standard deviation would rescale every group to unit
    spread, so a one-point judge fluctuation in an otherwise flat group would
    carry the same gradient magnitude as a genuine four-point win. Centering
    alone keeps advantages on the reward's own scale. The standard deviation is
    still returned, to skip groups that carry no signal and to log spread.
    """
    if len(rewards) < 2 or any(not math.isfinite(r) or not 0 <= r <= 4 for r in rewards):
        raise ValueError("GRPO requires at least two finite rewards in [0, 4]")
    values = torch.tensor(rewards, dtype=torch.float32)
    return (values - values.mean()).tolist(), values.std(unbiased=False).item()


def select_groups(rollouts, rewards):
    groups, skipped = [], 0
    for rollout in rollouts:
        values = [rewards[(rollout["case_id"], c["candidate_id"])] for c in rollout["candidates"]]
        advantages, std = group_advantages(values)
        if std == 0:
            skipped += 1
            continue
        groups.append({**rollout, "rewards": values, "advantages": advantages, "reward_std": std})
    return groups, skipped


def completion_token_logps(model, prompt, completion, temperature):
    """Match the actual temperature-scaled sampler (top-p=1, top-k=0)."""
    if not prompt or not completion:
        raise ValueError("prompt and completion must both contain tokens")
    device = next(model.parameters()).device
    ids = torch.tensor([prompt + completion[:-1]], device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   logits_to_keep=len(completion), use_cache=False).logits[0].float()
    return -F.cross_entropy(logits / temperature, torch.tensor(completion, device=device), reduction="none")


def grpo_loss(logps, old_logps, ref_logps, advantage, clip_epsilon, beta, length_normalizer):
    """Dr. GRPO: token sum over a fixed constant, then equal completion weights.

    Dividing by each completion's own length would give a long low-advantage
    completion a smaller per-token penalty than a short one, which rewards
    padding wrong answers. A constant normalizer removes that pressure. Set it
    near the observed completion length so the gradient keeps the scale a token
    mean produced, rather than the generation cap, which would shrink it.
    """
    ratio = torch.exp(logps - old_logps)
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
    if beta:
        log_ratio = ref_logps - logps
        kl = torch.expm1(log_ratio) - log_ratio
    else:
        kl = torch.zeros_like(logps)
    return (-surrogate + beta * kl).sum() / length_normalizer, kl.mean()


# Picklable payloads travel on their own gloo group. Under NCCL the object
# collectives size a receive buffer from an uninitialized device tensor filled
# by a preceding broadcast, and any drift between that broadcast and the dense
# gradient all-reduces on the same stream makes the next allocation read garbage
# (observed as a multi-exabyte request). Gloo keeps them on the CPU, away from
# the gradient stream; the payloads are a few kilobytes, so the cost is nil.
_OBJECTS = None


def topology():
    """Rank, local rank and world size as torchrun sets them."""
    return (int(os.environ.get("RANK", "0")),
            int(os.environ.get("LOCAL_RANK", "0")),
            int(os.environ.get("WORLD_SIZE", "1")))


def start(device):
    """Join the process group, leaving each rank exactly one visible GPU.

    `sft.modeling.model_dtype` requires a single visible device, and NCCL warns
    that an unknown rank-to-GPU mapping can hang, so under torchrun each rank
    narrows CUDA_VISIBLE_DEVICES to its own card before CUDA initializes. Every
    rank then addresses that card as cuda:0 and reports one device. Launched
    without torchrun, the caller's own CUDA_VISIBLE_DEVICES is left untouched.
    """
    rank, local_rank, size = topology()
    if device == "cuda" and "LOCAL_RANK" in os.environ:
        visible = [part for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part]
        if visible and len(visible) < size:
            raise RuntimeError(f"{size} ranks requested but only {len(visible)} GPU(s) visible: "
                               f"{','.join(visible)}")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible[local_rank] if visible else str(local_rank)
    global _OBJECTS
    if size > 1:
        if device == "cuda":
            # Generation and judging can hold a rank far longer than the ten
            # minute default before the next collective, so allow an hour.
            dist.init_process_group(backend="nccl", device_id=torch.device("cuda:0"),
                                    timeout=timedelta(hours=1))
            torch.cuda.set_device(0)
            _OBJECTS = dist.new_group(backend="gloo", timeout=timedelta(hours=1))
            print(f"[rank {rank}] physical GPU {os.environ['CUDA_VISIBLE_DEVICES']}, "
                  f"{torch.cuda.device_count()} visible, using cuda:0", flush=True)
        else:
            dist.init_process_group(backend="gloo", timeout=timedelta(hours=1))
            _OBJECTS = None
    return rank, size


def stop(size):
    global _OBJECTS
    if size > 1 and dist.is_initialized():
        _OBJECTS = None
        dist.destroy_process_group()


def barrier(size):
    if size > 1:
        dist.barrier()


def share(value, size, source=0):
    """Give every rank the source rank's copy of a picklable value."""
    if size == 1:
        return value
    holder = [value]
    dist.broadcast_object_list(holder, src=source, group=_OBJECTS)
    return holder[0]


def collect(value, size):
    """Every rank's value, in rank order, on every rank."""
    if size == 1:
        return [value]
    holder = [None] * size
    dist.all_gather_object(holder, value, group=_OBJECTS)
    return holder


def agree(value, size, label):
    """Fail loudly where the ranks disagree, instead of hanging in a collective.

    Every rank must reach the dense reduction the same number of times. When
    they do not, NCCL reports only a sequence number after its timeout, minutes
    later and on the wrong rank, so the disagreement is checked first on the
    object group where it can still be named.
    """
    if size == 1:
        return value
    seen = collect(value, size)
    if len(set(seen)) != 1:
        raise RuntimeError(f"ranks disagree on {label}: {seen}")
    return value


def shard(items, rank, size):
    """Round robin, so a batch shorter than the world still spreads out."""
    return items[rank::size]


def sample_shard(sampler, model, tokenizer, rows, prompts, batch_id, config, rank, size):
    """Generate this rank's prefixes, then reassemble the batch's own order."""
    produced = {r["case_id"]: r for r in
                sampler(model, tokenizer, shard(rows, rank, size), prompts, batch_id, config)}
    merged = {}
    for part in collect(produced, size):
        merged.update(part)
    if set(merged) != {row["case_id"] for row in rows}:
        raise ValueError(f"sharded generation did not cover the batch: {batch_id}")
    return [merged[row["case_id"]] for row in rows]


def update_policy(model, optimizer, groups, prompts, config, rank, size,
                  accumulated_groups=0, step=True):
    """One on-policy update per collected window; forward one completion at a time.

    No optimizer step occurs while a window is collected, so detached current
    log probabilities are exactly the rollout policy's old log probabilities.
    Clipping is inert for this single-update setting (as in num_iterations=1).
    `groups` is the whole collection on every rank. Each rank backpropagates its
    own shard, and `accumulated_groups` counts globally, so the division at the
    accumulation boundary is the one a single process would apply.
    """
    # Generation leaves the adapter frozen (sample_candidates calls activate with
    # training off). Re-enable it here rather than inside the loop: a rank whose
    # shard is empty would otherwise hold no trainable parameters, contribute
    # nothing, and never enqueue the reduction the other ranks are waiting on.
    activate(model, training=True)
    losses, kls, lengths, token_logps = [], [], [], []
    rewarded, penalized = [], []
    for group in shard(groups, rank, size):
        prompt = prompts[group["case_id"]]
        for candidate, advantage in zip(group["candidates"], group["advantages"]):
            reference = None
            if config["beta"]:
                activate(model)
                with model.disable_adapter(), torch.no_grad():
                    reference = completion_token_logps(model, prompt, candidate["token_ids"],
                                                       config["temperature"])
            activate(model, training=True)
            logps = completion_token_logps(model, prompt, candidate["token_ids"],
                                           config["temperature"])
            loss, kl = grpo_loss(logps, logps.detach(), reference, advantage,
                                 config["clip_epsilon"], config["beta"],
                                 config["length_normalizer"])
            if not torch.isfinite(loss):
                raise ValueError("non-finite GRPO loss; collection has not been committed")
            (loss / len(group["candidates"])).backward()
            losses.append(loss.detach().item())
            kls.append(kl.detach().item())
            lengths.append(len(candidate["token_ids"]))
            # Falling mean token log probability means the policy is collapsing
            # onto fewer completions; a constant normalizer cannot inflate length.
            mean_logp = logps.detach().mean().item()
            token_logps.append(mean_logp)
            # The surrogate is constant while the ratio is one, so progress shows
            # here instead: above-average completions must gain probability
            # relative to below-average ones.
            if advantage > 0:
                rewarded.append(mean_logp)
            elif advantage < 0:
                penalized.append(mean_logp)
    # Diagnostics describe the whole collection, not one rank's share. One
    # collective carries all six series so the ranks cannot drift between them.
    gathered = collect((losses, kls, lengths, token_logps, rewarded, penalized), size)
    losses, kls, lengths, token_logps, rewarded, penalized = (
        [value for part in parts for value in part] for parts in zip(*gathered))

    accumulated_groups += len(groups)
    update_groups = accumulated_groups if step else 0
    agree(update_groups, size, "the number of groups in this optimizer step")
    norm = None
    if update_groups:
        parameters = [p for p in model.parameters() if p.requires_grad]
        if size > 1:
            # Sum the ranks' partial gradients before the single shared division,
            # so the result matches one process holding every group. A rank that
            # drew no group still contributes its zeros to the collective.
            # One flattened reduction rather than one per parameter: 504 separate
            # collectives per update are 504 chances for the ranks to desynchronize,
            # and NCCL only reports the sequence number when they do.
            for parameter in parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
            gradients = [parameter.grad for parameter in parameters]
            flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            offset = 0
            for gradient in gradients:
                gradient.copy_(flat[offset:offset + gradient.numel()].view_as(gradient))
                offset += gradient.numel()
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(update_groups)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"],
                                                    error_if_nonfinite=True))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        accumulated_groups = 0
    average = lambda series: sum(series) / len(series) if series else None
    return {"grpo_loss": average(losses), "kl": average(kls), "grad_norm": norm,
            "mean_completion_tokens": average(lengths),
            "mean_token_logp": average(token_logps),
            "rewarded_token_logp": average(rewarded),
            "penalized_token_logp": average(penalized),
            "advantage_logp_gap": (average(rewarded) - average(penalized))
                                  if rewarded and penalized else None,
            "updated": bool(update_groups), "update_groups": update_groups,
            "accumulated_groups": accumulated_groups}


def restore(state, model, optimizer, rank):
    """Reload a checkpoint onto every rank, whatever world size wrote it.

    CUDA generator states are saved per visible device, so a run moved between
    a one-GPU and a four-GPU allocation cannot restore them positionally. Only
    the CPU generator is required for equivalence: rollouts reseed per candidate
    from (seed, batch id, case id, candidate id), and dropout is disabled.
    """
    set_peft_model_state_dict(model, state["policy"], adapter_name="default")
    optimizer.load_state_dict(state["optimizer"])
    # Pending gradients are already the sum over every rank's earlier shards,
    # so exactly one rank may carry them into the next reduction; the others
    # start empty or the window's accumulated gradient is multiplied by the world.
    if rank == 0:
        parameters = dict(model.named_parameters())
        for name, gradient in state["gradients"].items():
            parameters[name].grad = gradient.to(parameters[name])
    torch.set_rng_state(state["torch_rng"])
    saved = state["cuda_rng"]
    if saved and torch.cuda.is_available() and len(saved) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(saved)
    return state["cursor"], state["updates"], state["history"], state["accumulation"]


def continue_from(config, checkpoint):
    """Record the exported adapter this run continues, after checking it fits the base."""
    checkpoint = checkpoint.resolve()
    exported = json.loads((checkpoint / "export_complete.json").read_text())
    base = json.loads((checkpoint / "dpo_base.json").read_text())
    if any(base.get(key) != value for key, value in config["merged_base"].items()):
        raise ValueError("the checkpoint was trained on a different merged SFT base")
    adapter = json.loads((checkpoint / "adapter_config.json").read_text())
    shape = config["fresh_lora"]
    if (adapter["r"] != shape["r"] or adapter["lora_alpha"] != shape["lora_alpha"]
            or set(adapter["target_modules"]) != set(shape["target_modules"])):
        raise ValueError("the checkpoint's LoRA shape differs from this run's configuration")
    source = json.loads((checkpoint.parent / "run.json").read_text())
    if source["condition"] != config["condition"] or source["data_dir"] != config["data_dir"]:
        raise ValueError("the checkpoint was trained on a different split; choose a matching one")
    config["initialized_from"] = {
        "checkpoint": str(checkpoint), "source_run": str(checkpoint.parent),
        "source_algorithm": source.get("algorithm"), "source_epoch": exported["epoch"],
        "source_updates": exported["updates"],
        "source_learning_rate": source["learning_rate"],
        "adapter": "trained further in place; not merged into the base",
        "reference": "merged SFT base, unchanged",
    }
    return config


def load_continued_policy(config):
    """Merged SFT base with the recorded adapter loaded as the trainable policy."""
    checkpoint = Path(config["initialized_from"]["checkpoint"])
    merged = load_merged_sft(config["merged_base"])
    return PeftModel.from_pretrained(merged, checkpoint, is_trainable=True)


def continued_optimizer_state(config):
    """The source run's AdamW moments, required to match the continued export.

    `latest.pt` holds the state at the point the source run stopped, so it only
    describes the final epoch's adapter. Continuing an earlier epoch with these
    moments would pair second-order estimates with weights that never produced
    them, so that combination is refused rather than silently accepted.
    """
    lineage = config["initialized_from"]
    path = Path(lineage["source_run"]) / "latest.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["updates"] != lineage["source_updates"]:
        raise ValueError(
            f"{path} holds the state after {state['updates']} updates, but the continued "
            f"checkpoint was exported at {lineage['source_updates']}; continue from the "
            "checkpoint that matches the source run's final state")
    if state["gradients"]:
        raise ValueError(f"{path} carries pending gradients; the source run stopped "
                         "mid-accumulation and its moments do not describe a committed step")
    return state["optimizer"]


def adopt_optimizer(optimizer, saved, learning_rate):
    """Carry the moments and the step count, but not the source's step size.

    AdamW's saved `param_groups` include the learning rate that produced them,
    which `load_state_dict` would reinstate over this run's own setting.
    """
    trainable = list(optimizer.param_groups[0]["params"])
    shapes = [tuple(saved["state"][key]["exp_avg"].shape) for key in sorted(saved["state"])]
    if len(trainable) != len(shapes) or any(tuple(p.shape) != s for p, s in zip(trainable, shapes)):
        raise ValueError("the continued adapter's parameters do not match the saved moments")
    optimizer.load_state_dict(saved)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
