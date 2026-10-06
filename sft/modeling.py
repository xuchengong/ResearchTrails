"""Completion-only chat encoding and padding; no cross-example packing."""

import json

import torch
from torch.utils.data import Dataset


def encode_example(row: dict, tokenizer, max_length: int) -> dict:
    messages = row["messages"]
    if [message["role"] for message in messages] != ["system", "user"]:
        raise ValueError(f"expected system + user messages: {row['case_id']}")
    target = json.dumps(row["target"], ensure_ascii=False)
    # Qwen3's non-thinking input includes an empty think block. Mask it as
    # part of the prompt, so the supervised continuation starts at the JSON.
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    full_ids = tokenizer.apply_chat_template(
        # A scientific decision may literally mention </think>. Explicitly
        # empty reasoning prevents Qwen from stripping that part of the JSON.
        messages + [{"role": "assistant", "content": target, "reasoning_content": ""}],
        tokenize=True, add_generation_prompt=False, enable_thinking=False,
    )
    if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError(f"chat template does not support completion-only masking: {row['case_id']}")
    if len(full_ids) > max_length:
        raise ValueError(
            f"{row['case_id']} has {len(full_ids)} tokens, exceeding --max-length {max_length}; "
            "increase the limit. This experiment does not silently truncate prefixes or targets."
        )
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
    }


class CompletionDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.examples = [encode_example(row, tokenizer, max_length) for row in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        width = max(len(feature["input_ids"]) for feature in features)
        return {
            key: torch.tensor([
                feature[key] + [pad] * (width - len(feature[key])) for feature in features
            ], dtype=torch.long)
            for key, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))
        }


def model_dtype(precision: str, device: str):
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use a GPU node or explicit --device cpu --precision fp32")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "This pipeline requires exactly one visible GPU. Preserve the scheduler's "
                "CUDA_VISIBLE_DEVICES, or explicitly select one allocated GPU."
            )
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support bf16; explicitly select --precision fp16")
    elif precision != "fp32":
        raise ValueError("CPU runs require --precision fp32")
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
