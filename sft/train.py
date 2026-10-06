"""Train the LoRA adapter on the first-1/2/3 training cases."""

import argparse
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path

from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from sft.io import HERE, write_once, json_text, load_split
from sft.modeling import CompletionCollator, CompletionDataset, model_dtype

CONDITION = "early_only"


def training_config(args):
    """The run.json of a training run; resuming requires the same configuration."""
    return {
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items() if key not in {"resume", "output_dir"}},
        "condition": CONDITION,
        "packages": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
        "objective": "completion-token cross entropy; uniform example sampling; no packing",
        "checkpoint_selection": "final epoch; evaluation cohort is never used for selection",
        "enable_thinking": False,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
    }


def training_arguments(args):
    """Cosine decay after linear warmup; one checkpoint per epoch and no in-training evaluation.
    Held-out loss is measured afterwards from the checkpoints (sft.evaluate_checkpoints)."""
    return TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine", warmup_ratio=args.warmup_ratio,
        weight_decay=0.0, max_grad_norm=1.0, optim="adamw_torch",
        bf16=args.precision == "bf16", fp16=args.precision == "fp16", use_cpu=args.device == "cpu",
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="epoch", save_total_limit=None,
        logging_strategy="steps", logging_steps=1, report_to="none",
        seed=args.seed, data_seed=args.seed, dataloader_num_workers=0,
        remove_unused_columns=False, label_names=["labels"],
    )


def train(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This launcher supports one GPU per run; launch seeds independently, not with torchrun.")
    if args.epochs <= 0 or args.batch_size < 1 or args.gradient_accumulation < 1 or args.max_length < 1:
        raise ValueError("epochs, batch size, gradient accumulation and max length must be positive")
    if args.learning_rate <= 0 or args.rank < 1 or args.alpha < 1 or not 0 <= args.dropout < 1:
        raise ValueError("invalid learning rate or LoRA parameters")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup ratio must be in [0, 1)")
    dtype = model_dtype(args.precision, args.device)
    rows = load_split(args.data_dir, "train_" + CONDITION)
    config = training_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = [p for p in args.output_dir.iterdir() if p.name != ".lock"]
        if existing and not args.resume:
            raise ValueError("output directory is not empty; explicitly use --resume or choose another directory")
        write_once(args.output_dir / "run.json", json_text(config))
        if (args.output_dir / "completed.json").exists():
            print(f"Already complete: {args.output_dir / 'adapter'}")
            return
        checkpoint = get_last_checkpoint(str(args.output_dir)) if args.resume else None
        set_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model_info_path = args.output_dir / "model_info.json"
        # A resumed run uses the original resolved Hub commit, not a moving 'main'.
        revision = (json.loads(model_info_path.read_text())["resolved_revision"]
                    if model_info_path.exists() else args.revision)
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
        if not tokenizer.chat_template or tokenizer.eos_token_id is None:
            raise ValueError("model tokenizer must provide a chat template and an EOS token")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        dataset = CompletionDataset(rows, tokenizer, args.max_length)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=revision, dtype=dtype, attn_implementation="sdpa",
        )
        if args.max_length > model.config.max_position_embeddings:
            raise ValueError("--max-length exceeds the model context window")
        write_once(model_info_path, json_text({
            "base_model": args.model,
            "resolved_revision": getattr(model.config, "_commit_hash", None) or args.revision,
            "enable_thinking": False,
        }))
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=args.rank, lora_alpha=args.alpha,
            lora_dropout=args.dropout, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
        model.print_trainable_parameters()
        lengths = [len(example["input_ids"]) for example in dataset.examples]
        completion_tokens = sum(sum(label != -100 for label in ex["labels"]) for ex in dataset.examples)
        write_once(args.output_dir / "token_stats.json", json_text({
            "examples": len(dataset), "projects": len({row["project_id"] for row in rows}),
            "max_tokens": max(lengths), "total_tokens_per_epoch": sum(lengths),
            "supervised_tokens_per_epoch": completion_tokens,
        }))
        print(
            f"Training {args.model}: {len(dataset)} examples, max {max(lengths)} tokens; "
            f"microbatch {args.batch_size} x accumulation {args.gradient_accumulation} "
            f"= effective batch {args.batch_size * args.gradient_accumulation}", flush=True,
        )
        trainer = Trainer(
            model=model, args=training_arguments(args), train_dataset=dataset,
            data_collator=CompletionCollator(tokenizer.pad_token_id), processing_class=tokenizer,
        )
        result = trainer.train(resume_from_checkpoint=checkpoint)
        if args.device == "cuda":
            # Peak includes model loading and training in this process; on resume
            # it describes this invocation, not earlier checkpoint-producing jobs.
            result.metrics.update(
                gpu_name=torch.cuda.get_device_name(),
                gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_cuda_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            )
            print(
                f"Peak CUDA memory: {result.metrics['peak_cuda_allocated_gib']:.2f} GiB allocated, "
                f"{result.metrics['peak_cuda_reserved_gib']:.2f} GiB reserved", flush=True,
            )
        adapter = args.output_dir / "adapter"
        trainer.save_model(str(adapter))
        tokenizer.save_pretrained(adapter)
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        write_once(args.output_dir / "completed.json", json_text({
            "adapter": "adapter", "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch, "train_loss": result.training_loss,
        }))
        print(f"Saved LoRA adapter: {adapter.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume latest epoch checkpoint, or skip completed run")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
