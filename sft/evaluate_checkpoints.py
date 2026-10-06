"""Evaluate baseline and every saved epoch checkpoint using gold completion loss.

No generation or judge API calls. Writes only to a separate evaluation directory.
"""

import argparse
import csv
import fcntl
from importlib.metadata import version
import json
import math
import os
from pathlib import Path

from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft.evaluation import completion_loss
from sft.io import EVAL_SPLIT, ROOT, json_text, load_split
from sft.modeling import CompletionCollator, CompletionDataset, model_dtype


def prepare_evaluation(args):
    """Validate completed inputs before loading a model or writing results."""
    run = json.loads((args.run_dir / "run.json").read_text())
    info = json.loads((args.run_dir / "model_info.json").read_text())
    completed = json.loads((args.run_dir / "completed.json").read_text())
    data_dir = args.data_dir if args.data_dir is not None else ROOT / run["data_dir"]
    datasets = {"eval": load_split(data_dir, "eval_" + EVAL_SPLIT)}
    train_rows = load_split(data_dir, "train_" + run["condition"])
    if {row["project_id"] for row in train_rows} & {row["project_id"] for row in datasets["eval"]}:
        raise ValueError("training and evaluation projects overlap")
    checkpoints = []
    for path in sorted(args.run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
        state = json.loads((path / "trainer_state.json").read_text())
        if state["global_step"] != int(path.name.split("-")[-1]):
            raise ValueError(f"checkpoint step differs from its directory: {path}")
        checkpoints.append({
            "checkpoint": path.name, "epoch": state["epoch"], "global_step": state["global_step"],
        })
    if not checkpoints or len(checkpoints) != math.ceil(completed["epoch"]):
        raise ValueError("expected every epoch checkpoint; one or more saved checkpoints are missing")
    if checkpoints[-1]["global_step"] != completed["global_step"]:
        raise ValueError("final checkpoint does not match completed.json")
    if any(not i < item["epoch"] <= i + 1 for i, item in enumerate(checkpoints)):
        raise ValueError("checkpoints do not contain one saved result per epoch")
    plan = {
        "run_dir": str(args.run_dir.resolve()), "data_dir": str(data_dir.resolve()),
        "model": info["base_model"], "revision": info["resolved_revision"],
        "split": EVAL_SPLIT, "batch_size": args.batch_size,
        "max_length": args.max_length if args.max_length is not None else run["max_length"],
        "precision": args.precision if args.precision is not None else run["precision"],
        "device": args.device, "checkpoints": checkpoints,
        "cases": {name: len(rows) for name, rows in datasets.items()},
        "packages": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
        "objective": "completion-token-weighted cross entropy; gold JSON answer and terminator; no prompt/padding loss",
    }
    return plan, datasets


def write_summary(output, results):
    with (output / "losses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    table = [
        "| Checkpoint | Epoch | Eval loss | Eval perplexity |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in results:
        table.append(
            f"| {row['checkpoint']} | {row['epoch']:g} | "
            f"{row['eval_loss']:.4f} | {row['eval_perplexity']:.3f} |"
        )
    text = "\n".join(table) + "\n"
    (output / "losses.md").write_text(text)
    print(text, flush=True)


def evaluate(args):
    if args.batch_size < 1 or (args.max_length is not None and args.max_length < 1):
        raise ValueError("batch size and max length must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("checkpoint evaluation supports one process only")
    plan, rows = prepare_evaluation(args)
    targets = [{"checkpoint": "baseline", "epoch": 0.0, "global_step": 0}, *plan["checkpoints"]]
    output = args.output_dir or args.run_dir / "evaluation_loss"
    # Prevent an explicit output path from adding files inside saved adapters.
    if output.resolve() == args.run_dir.resolve() or any(
        output.resolve().is_relative_to((args.run_dir / name).resolve())
        for name in ["adapter", *(item["checkpoint"] for item in plan["checkpoints"])]
    ):
        raise ValueError("choose a separate evaluation output directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = output / "evaluation_run.json"
        if record.exists() and record.read_text() != json_text(plan):
            # Results computed under other settings are stale: recompute every checkpoint.
            for item in targets:
                (output / (item["checkpoint"] + ".json")).unlink(missing_ok=True)
        record.write_text(json_text(plan))
        results, pending = {}, []
        for item in targets:
            path = output / (item["checkpoint"] + ".json")
            if path.exists():
                row = json.loads(path.read_text())
                if any(row[key] != item[key] for key in ("checkpoint", "epoch", "global_step")):
                    raise ValueError(f"saved evaluation identity differs: {path}")
                for name in rows:
                    if (row[f"{name}_examples"] != len(rows[name]) or row[f"{name}_completion_tokens"] <= 0
                            or not math.isfinite(row[f"{name}_loss"]) or row[f"{name}_loss"] < 0
                            or not math.isfinite(row[f"{name}_perplexity"])):
                        raise ValueError(f"invalid saved evaluation metrics: {path}")
                results[item["checkpoint"]] = row
            else:
                pending.append(item)
        print(f"{len(results)}/{len(targets)} evaluations saved; {len(pending)} remaining. Output: {output}", flush=True)
        if args.prepare_only:
            print("Prepared only; no model loaded or GPU work started.")
            return
        if pending:
            dtype = model_dtype(plan["precision"], args.device)
            # Tokenizer saved with the final adapter uses the original template.
            tokenizer = AutoTokenizer.from_pretrained(args.run_dir / "adapter")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            batches = {
                name: DataLoader(
                    CompletionDataset(data, tokenizer, plan["max_length"]), batch_size=args.batch_size,
                    shuffle=False, collate_fn=CompletionCollator(tokenizer.pad_token_id),
                ) for name, data in rows.items()
            }
            model = AutoModelForCausalLM.from_pretrained(
                plan["model"], revision=plan["revision"], dtype=dtype, attn_implementation="sdpa",
            ).to(args.device)
            if plan["max_length"] > model.config.max_position_embeddings:
                raise ValueError("max length exceeds the model context window")
            model.config.use_cache = False
            active_adapter = None
            for item in pending:
                name = item["checkpoint"]
                if name != "baseline":
                    path = args.run_dir / name
                    if active_adapter is None:
                        model = PeftModel.from_pretrained(model, path, adapter_name=name, is_trainable=False)
                    else:
                        loaded = model.load_adapter(path, adapter_name=name, is_trainable=False)
                        if loaded.missing_keys or loaded.unexpected_keys:
                            raise ValueError(f"adapter weights do not match: {path}: {loaded}")
                        model.set_adapter(name)
                        model.delete_adapter(active_adapter)
                    active_adapter = name
                model.requires_grad_(False).eval()
                row = {key: item[key] for key in ("checkpoint", "epoch", "global_step")}
                for split, loader in batches.items():
                    print(f"Evaluating {name}: {split}, {len(rows[split])} examples", flush=True)
                    metrics = completion_loss(model, loader, dtype if plan["precision"] != "fp32" else None)
                    row.update({f"{split}_{key}": value for key, value in metrics.items()})
                # Commit only complete results; interrupted checkpoints can be
                # rerun without changing previously completed evaluations.
                destination = output / (name + ".json")
                temporary = destination.with_suffix(".tmp")
                temporary.write_text(json_text(row))
                temporary.replace(destination)
                results[name] = row
                write_summary(output, [results[t["checkpoint"]] for t in targets if t["checkpoint"] in results])
        write_summary(output, [results[item["checkpoint"]] for item in targets])


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="completed training run containing checkpoint-* directories")
    parser.add_argument("--data-dir", type=Path, help="default: the data directory recorded by training")
    parser.add_argument("--output-dir", type=Path, help="default: RUN_DIR/evaluation_loss")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, help="default: training max length; no truncation")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], help="default: training precision")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--prepare-only", action="store_true", help="validate inputs and write the run record without loading models")
    return parser


if __name__ == "__main__":
    evaluate(argument_parser().parse_args())
