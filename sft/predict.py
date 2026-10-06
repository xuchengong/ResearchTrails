"""Generate held-out predictions with the untuned model or a saved SFT adapter."""

import argparse
import fcntl
import json
import os
from pathlib import Path

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from sft.io import EVAL_SPLIT, HERE, write_once, json_text, load_split
from sft.modeling import model_dtype
from sft.render import CATEGORIES


def parse_prediction(raw_text: str):
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if (not isinstance(value, dict) or set(value) != {"category", "decision"}
            or value["category"] not in CATEGORIES
            or not isinstance(value["decision"], str) or not value["decision"].strip()):
        return None
    return value


def predict(args):
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    dtype = model_dtype(args.precision, args.device)
    rows = load_split(args.data_dir, "eval_" + EVAL_SPLIT)
    model_id, revision = args.model, args.revision
    if args.adapter:
        run_dir = args.adapter.parent
        if not (run_dir / "completed.json").exists():
            raise ValueError("adapter must be the final adapter from a completed sft.train run")
        model_info = json.loads((run_dir / "model_info.json").read_text())
        model_id, revision = model_info["base_model"], model_info["resolved_revision"]
        if args.model is not None and args.model != model_id:
            raise ValueError("--model does not match the adapter's base model")
    elif model_id is None:
        model_id = "Qwen/Qwen3-8B"
    config = {
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items() if key != "output_dir"},
        "split": EVAL_SPLIT, "base_model": model_id, "requested_revision": revision,
        "enable_thinking": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(args.output_dir / "run.json", json_text(config))
        output = args.output_dir / "predictions.jsonl"
        valid_ids = {row["case_id"] for row in rows}
        saved = {}
        with output.open("a+") as handle:
            handle.seek(0)
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    # A killed write did not commit this last record. Retry it.
                    handle.seek(offset)
                    handle.truncate()
                    break
                record = json.loads(line)
                case_id = record["case_id"]
                if case_id not in valid_ids or case_id in saved:
                    raise ValueError(f"unknown/duplicate saved prediction: {case_id}")
                saved[case_id] = record
            if len(saved) < len(rows):
                info_path = args.output_dir / "model_info.json"
                if info_path.exists():
                    revision = json.loads(info_path.read_text())["resolved_revision"]
                tokenizer = AutoTokenizer.from_pretrained(
                    str(args.adapter) if args.adapter else model_id,
                    **({} if args.adapter else {"revision": revision}),
                )
                if not tokenizer.chat_template or tokenizer.eos_token_id is None:
                    raise ValueError("model must provide a chat template and EOS token")
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, revision=revision, dtype=dtype, attn_implementation="sdpa",
                )
                if args.adapter:
                    model = PeftModel.from_pretrained(model, args.adapter)
                model.to(args.device).eval()
                model.config.use_cache = True
                write_once(info_path, json_text({
                    "base_model": model_id,
                    "resolved_revision": getattr(model.config, "_commit_hash", None) or revision,
                    "enable_thinking": False,
                }))
                # Do not inherit Qwen's sampling defaults for a greedy comparison.
                generation = GenerationConfig(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=model.generation_config.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                )
                for row in rows:
                    if row["case_id"] in saved:
                        continue
                    input_ids = tokenizer.apply_chat_template(
                        row["messages"], tokenize=True, add_generation_prompt=True,
                        return_tensors="pt", enable_thinking=False,
                    ).to(args.device)
                    if input_ids.shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
                        raise ValueError(f"context limit exceeded: {row['case_id']}; no truncation is allowed")
                    with torch.inference_mode():
                        # Transformers merges the checkpoint's generation_config.json into an
                        # explicitly passed config for every field left at a library default, so
                        # Qwen3's do_sample/temperature/top_k/top_p would silently override these.
                        generated = model.generate(
                            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                            generation_config=generation, use_model_defaults=False,
                        )[0, input_ids.shape[1]:]
                    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
                    parsed = parse_prediction(raw)
                    record = {
                        "case_id": row["case_id"], "project_id": row["project_id"],
                        "target_step": row["target_step"], "prediction": parsed,
                        "raw_text": raw, "valid_json": parsed is not None,
                        "generated_tokens": len(generated),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    saved[row["case_id"]] = record
                    print(f"Predictions {len(saved)}/{len(rows)}", flush=True)
        invalid = sum(not row["valid_json"] for row in saved.values())
        write_once(args.output_dir / "summary.json", json_text({
            "cases": len(saved), "invalid_json": invalid, "valid_json_rate": 1 - invalid / len(saved),
        }))
        print(f"Saved {len(saved)} predictions ({invalid} invalid JSON): {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--adapter", type=Path, help="completed training run's adapter directory; omit for baseline")
    parser.add_argument("--model", help="baseline model; defaults to Qwen/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    predict(parser.parse_args())


if __name__ == "__main__":
    main()
