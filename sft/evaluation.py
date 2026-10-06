"""Token-weighted, completion-only loss for checkpoint evaluation."""

import math

import torch


def completion_loss(model, batches, autocast_dtype=None):
    """Score gold answer tokens without generation, gradients, or RNG side effects."""
    device = next(model.parameters()).device
    was_training = model.training
    total_nll, tokens, examples = 0.0, 0, 0
    model.eval()
    try:
        # DataLoader iteration also consumes RNG; keep evaluation free of RNG side effects.
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
            ):
                for batch in batches:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    count = int(batch["labels"][:, 1:].ne(-100).sum())
                    if not count:
                        raise ValueError("evaluation batch has no supervised completion tokens")
                    # The causal LM shifts labels internally. Its mean CE includes
                    # only non-masked labels; multiply back to NLL before pooling.
                    loss = float(model(**batch, use_cache=False).loss)
                    if not math.isfinite(loss):
                        raise ValueError("non-finite completion loss")
                    total_nll += loss * count
                    tokens += count
                    examples += batch["input_ids"].shape[0]
    finally:
        model.train(was_training)
    if not tokens:
        raise ValueError("evaluation dataset is empty")
    loss = total_nll / tokens
    return {"loss": loss, "perplexity": math.exp(loss),
            "completion_tokens": tokens, "examples": examples}
