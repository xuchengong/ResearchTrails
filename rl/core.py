"""Prompt encoding and seeded candidate sampling for GRPO training and merged-model prediction."""

import hashlib

import torch
from transformers import GenerationConfig

from sft.predict import parse_prediction


def activate(model, training=False):
    model.set_adapter("default")
    model.set_requires_grad("default", training)
    model.train(training)


def prompt_tokens(row, tokenizer):
    if [m["role"] for m in row["messages"]] != ["system", "user"]:
        raise ValueError(f"expected system + user messages: {row['case_id']}")
    return tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )


def sample_candidates(model, tokenizer, rows, prompts, batch_id, config):
    """Only messages enter generation. Keep original token IDs, including EOS."""
    activate(model)
    device = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else eos
    if not eos_ids:
        raise ValueError("generation requires EOS token IDs")
    generation = GenerationConfig(
        do_sample=True, temperature=config["temperature"], top_p=config["top_p"], top_k=0,
        max_new_tokens=config["max_new_tokens"], eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
    )
    result = []
    for row in rows:
        prompt = prompts[row["case_id"]]
        candidates = []
        for candidate_id in range(config["candidates"]):
            identity = f"{config['seed']}:{batch_id}:{row['case_id']}:{candidate_id}"
            seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
            # Stable per-candidate seeds make interrupted generation reproducible.
            with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                torch.manual_seed(seed)
                ids = torch.tensor([prompt], device=device)
                with torch.no_grad():
                    # Transformers merges the checkpoint's generation_config.json into an
                    # explicitly passed config for every field left at a library default, so
                    # Qwen3's do_sample/temperature/top_k/top_p would silently override these.
                    generated = model.generate(
                        input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=generation,
                        use_model_defaults=False,
                    )[0, len(prompt):].tolist()
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            candidates.append({
                "candidate_id": candidate_id, "token_ids": generated, "raw_text": raw,
                "prediction": parse_prediction(raw), "finished": bool(generated and generated[-1] in eos_ids),
            })
        result.append({"case_id": row["case_id"], "candidates": candidates})
    return result
