"""Reconstruct the frozen merged SFT model before adding the fresh GRPO LoRA adapter."""

from pathlib import Path

from peft import LoraConfig, PeftModel, get_peft_model
import torch
from transformers import AutoModelForCausalLM


def load_merged_sft(spec):
    """Merge in memory in the recorded dtype, identically for train and predict."""
    checkpoint = Path(spec["sft_checkpoint"])
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[spec["precision"]]
    base = AutoModelForCausalLM.from_pretrained(
        spec["model"], revision=spec["revision"], dtype=dtype, attn_implementation="sdpa",
    )
    if base.config.model_type != "qwen3":
        raise ValueError("this pipeline targets the existing Qwen3 SFT model")
    sft = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
    merged = sft.merge_and_unload(safe_merge=True)
    # PEFT 0.18 removes the LoRA layers but leaves this registration on the
    # underlying Transformer. Remove the old registration before fresh LoRA.
    del merged.peft_config
    merged.requires_grad_(False)
    return merged


def initialize_policy(config):
    merged = load_merged_sft(config["merged_base"])
    # Default LoRA initialization: random A and zero B, so the initial
    # correction is zero without making gradients to both factors vanish.
    return get_peft_model(merged, LoraConfig(**config["fresh_lora"]))
