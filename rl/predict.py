"""Generate decisions on the held-out prefixes after rebuilding a GRPO adapter's merged SFT base."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from peft import PeftModel
import torch
from transformers import AutoTokenizer, GenerationConfig, set_seed

from rl.core import prompt_tokens
from rl.modeling import load_merged_sft
from sft.io import EVAL_SPLIT, write_once, json_text, load_split
from sft.modeling import model_dtype
from sft.predict import parse_prediction


def predict(args):
    if args.temperature < 0 or args.max_new_tokens < 1:
        raise ValueError("temperature must be nonnegative; max-new-tokens must be positive")
    adapter = args.adapter.resolve()
    run_dir = adapter.parent
    training = json.loads((run_dir / "run.json").read_text())
    if training.get("algorithm") != "grpo" or training["adapter_mode"] != "merged_sft_fresh_lora":
        raise ValueError("expected a fresh GRPO adapter trained on a merged SFT base")
    json.loads((adapter / "export_complete.json").read_text())
    base_spec = json.loads((adapter / "merged_base.json").read_text())
    if base_spec != training["merged_base"]:
        raise ValueError("exported adapter's merged-base identity differs from its training run")
    data_dir = (args.data_dir or Path(training["data_dir"])).resolve()
    dataset = "eval_" + EVAL_SPLIT
    rows = load_split(data_dir, dataset)
    output_dir = args.output_dir.resolve()
    if output_dir == run_dir or any(output_dir.is_relative_to(p) for p in
                                   (adapter, Path(base_spec["sft_checkpoint"]).parent, data_dir)):
        raise ValueError("choose a prediction directory separate from checkpoints and data")
    config = {
        "adapter": str(adapter), "merged_base": base_spec,
        "policy": "sft_initialization" if args.sft_initialization else "grpo",
        "data_dir": str(data_dir),
        "split": EVAL_SPLIT, "dataset": dataset, "seed": args.seed, "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens, "device": args.device, "enable_thinking": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(output_dir / "run.json", json_text(config))
        if args.prepare_only:
            print(f"Prepared {len(rows)} prediction cases; no model loaded: {output_dir}")
            return
        valid_ids = {row["case_id"] for row in rows}
        saved = {}
        with (output_dir / "predictions.jsonl").open("a+") as handle:
            handle.seek(0)
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    handle.seek(offset)
                    handle.truncate()
                    break
                record = json.loads(line)
                if record["case_id"] not in valid_ids or record["case_id"] in saved:
                    raise ValueError(f"unknown/duplicate prediction: {record['case_id']}")
                saved[record["case_id"]] = record
            if len(saved) < len(rows):
                model_dtype(base_spec["precision"], args.device)
                tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                # The merged base is the exact SFT initialization/reference used
                # in GRPO. Apply the trained fresh adapter only for the GRPO method.
                model = load_merged_sft(base_spec)
                if not args.sft_initialization:
                    model = PeftModel.from_pretrained(model, adapter)
                model.to(args.device).eval()
                generation = GenerationConfig(
                    max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
                    **({"temperature": args.temperature, "top_p": 1.0, "top_k": 0}
                       if args.temperature > 0 else {}),
                    pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=model.generation_config.eos_token_id, use_cache=True,
                )
                for row in rows:
                    if row["case_id"] in saved:
                        continue
                    digest = hashlib.sha256(f"{args.seed}:{row['case_id']}".encode()).digest()
                    set_seed(int.from_bytes(digest[:4], "big"))
                    prompt = prompt_tokens(row, tokenizer)
                    if len(prompt) + args.max_new_tokens > model.config.max_position_embeddings:
                        raise ValueError(f"context limit exceeded: {row['case_id']}; no truncation allowed")
                    ids = torch.tensor([prompt], device=args.device)
                    with torch.inference_mode():
                        # Transformers merges the checkpoint's generation_config.json into an
                        # explicitly passed config for every field left at a library default, so
                        # Qwen3's do_sample/temperature/top_k/top_p would silently override these.
                        tokens = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                                generation_config=generation,
                                                use_model_defaults=False)[0, len(prompt):]
                    raw = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    parsed = parse_prediction(raw)
                    record = {
                        "case_id": row["case_id"], "project_id": row["project_id"],
                        "target_step": row["target_step"], "prediction": parsed,
                        "raw_text": raw, "valid_json": parsed is not None, "generated_tokens": len(tokens),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    saved[row["case_id"]] = record
                    print(f"Predictions {len(saved)}/{len(rows)}", flush=True)
        invalid = sum(not row["valid_json"] for row in saved.values())
        write_once(output_dir / "summary.json", json_text({
            "cases": len(saved), "invalid_json": invalid, "valid_json_rate": 1 - invalid / len(saved),
        }))
        print(f"Saved predictions: {output_dir / 'predictions.jsonl'}")


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True, help="GRPO epoch checkpoint or final adapter directory")
    parser.add_argument("--sft-initialization", action="store_true",
                        help="generate with this GRPO checkpoint's merged SFT base, without its fresh GRPO adapter")
    parser.add_argument("--data-dir", type=Path, help="default: the GRPO run's data directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


if __name__ == "__main__":
    predict(argument_parser().parse_args())
