"""Routing diagnostics for trained / mid-training MoLE models.

The single most important diagnostic for a frozen-experts + learned-router
setup is "where does each task's tokens get routed?". Collapse to a
single expert is the dominant failure mode (Wu et al. 2024 Fig 5), and
plain loss curves don't reveal it. This module installs forward hooks on
every `MoLEQKVConv1D.mole_gate_proj`, runs the model on a small batch
budget, and aggregates per-layer + overall routing statistics.

Two complementary numbers are reported:

- ``mean_softmax`` — the average gate probability assigned to each
  expert, regardless of routing mode. Surfaces gate beliefs.
- ``argmax_fraction`` — the fraction of tokens whose argmax landed on
  each expert. Under top-1 routing this is what actually controls the
  contribution; for soft routing it is still useful as a "winning
  expert" histogram.

A third scalar, ``mean_entropy``, is the average per-token Shannon
entropy of the gate distribution (in nats). High entropy near log(N) =
indecisive / near-uniform; low entropy = sharp / collapsed.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from lora_gpt2.mole_layers import MoLEQKVConv1D


def _aggregate(layer_stats: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Merge per-layer stats into one set of corpus-level numbers.

    Token counts vary by layer in principle (they shouldn't here — every
    layer sees the same batch — but we're defensive in case future
    callers feed partial-layer captures), so we recombine using token-
    weighted sums rather than averaging the per-layer means.
    """
    if not layer_stats:
        return {
            "mean_softmax": [],
            "argmax_fraction": [],
            "mean_entropy": 0.0,
            "tokens": 0,
        }
    num_experts = len(next(iter(layer_stats.values()))["mean_softmax"])
    softmax_sum = torch.zeros(num_experts)
    argmax_sum = torch.zeros(num_experts)
    entropy_sum = 0.0
    total_tokens = 0
    for stats in layer_stats.values():
        tokens = stats["tokens"]
        total_tokens += tokens
        softmax_sum += torch.tensor(stats["mean_softmax"]) * tokens
        argmax_sum += torch.tensor(stats["argmax_fraction"]) * tokens
        entropy_sum += stats["mean_entropy"] * tokens
    if total_tokens == 0:
        return {
            "mean_softmax": [0.0] * num_experts,
            "argmax_fraction": [0.0] * num_experts,
            "mean_entropy": 0.0,
            "tokens": 0,
        }
    return {
        "mean_softmax": (softmax_sum / total_tokens).tolist(),
        "argmax_fraction": (argmax_sum / total_tokens).tolist(),
        "mean_entropy": entropy_sum / total_tokens,
        "tokens": int(total_tokens),
    }


def routing_stats_from_logits(
    logits_by_layer: dict[int, torch.Tensor],
    attention_mask: torch.Tensor | None,
) -> dict[str, Any]:
    """Compute routing stats from already-captured per-layer gate logits.

    Pure function over tensors, kept separate from the hook plumbing so
    the math is easy to unit-test without spinning up GPT-2.

    Each entry of ``logits_by_layer`` has shape ``(B, L, N)``.
    ``attention_mask`` (``(B, L)`` of 0/1, optional) masks padding tokens.
    """
    layer_stats: dict[int, dict[str, Any]] = {}
    if attention_mask is not None:
        valid = attention_mask.bool()
    else:
        valid = None

    for layer_idx, logits in logits_by_layer.items():
        if logits.dim() != 3:
            raise ValueError(
                f"Expected (B, L, N) gate logits for layer {layer_idx}; got {tuple(logits.shape)}"
            )
        probs = F.softmax(logits.float(), dim=-1)
        if valid is not None:
            flat = probs[valid]
        else:
            flat = probs.reshape(-1, probs.size(-1))
        if flat.numel() == 0:
            continue
        num_experts = flat.size(-1)
        mean_softmax = flat.mean(dim=0)
        argmax_idx = flat.argmax(dim=-1)
        argmax_counts = torch.zeros(num_experts)
        for k in range(num_experts):
            argmax_counts[k] = (argmax_idx == k).float().sum()
        argmax_fraction = argmax_counts / argmax_counts.sum().clamp_min(1.0)
        # Per-token entropy in nats. clamp_min keeps log finite for the
        # fully-saturated softmax(prob[k]≈1) case.
        per_token_entropy = -(flat * flat.clamp_min(1e-12).log()).sum(dim=-1)
        mean_entropy = float(per_token_entropy.mean())
        layer_stats[layer_idx] = {
            "mean_softmax": mean_softmax.tolist(),
            "argmax_fraction": argmax_fraction.tolist(),
            "mean_entropy": mean_entropy,
            "tokens": int(flat.size(0)),
        }

    return {
        "per_layer": layer_stats,
        "aggregate": _aggregate(layer_stats),
    }


@torch.no_grad()
def collect_routing_stats(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run the model over a dataloader and aggregate per-layer routing stats.

    Forward hooks are installed on every ``mole_gate_proj`` so we capture
    the raw per-layer logits without modifying the forward path. We
    accumulate sums incrementally per batch (rather than concatenating
    all batches in memory) so this is safe to run on a large validation
    set.
    """
    was_training = model.training
    model.eval()

    mole_blocks: list[tuple[int, MoLEQKVConv1D]] = []
    for layer_idx, block in enumerate(model.transformer.h):
        if isinstance(block.attn.c_attn, MoLEQKVConv1D):
            mole_blocks.append((layer_idx, block.attn.c_attn))
    if not mole_blocks:
        raise RuntimeError("No MoLEQKVConv1D modules found.")

    num_experts = mole_blocks[0][1].num_experts
    softmax_sum: dict[int, torch.Tensor] = {
        layer_idx: torch.zeros(num_experts) for layer_idx, _ in mole_blocks
    }
    argmax_sum: dict[int, torch.Tensor] = {
        layer_idx: torch.zeros(num_experts) for layer_idx, _ in mole_blocks
    }
    entropy_sum: dict[int, float] = {layer_idx: 0.0 for layer_idx, _ in mole_blocks}
    token_count: dict[int, int] = {layer_idx: 0 for layer_idx, _ in mole_blocks}

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_module, _inp, out):
            captured[layer_idx] = out.detach()
        return hook

    handles = [
        mole.mole_gate_proj.register_forward_hook(make_hook(layer_idx))
        for layer_idx, mole in mole_blocks
    ]

    try:
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            captured.clear()
            _ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            mask = batch.get("attention_mask")
            valid = mask.bool() if mask is not None else None
            for layer_idx, _ in mole_blocks:
                logits = captured.get(layer_idx)
                if logits is None:
                    continue
                probs = F.softmax(logits.float(), dim=-1).cpu()
                if valid is not None:
                    flat = probs[valid.cpu()]
                else:
                    flat = probs.reshape(-1, num_experts)
                if flat.numel() == 0:
                    continue
                softmax_sum[layer_idx] += flat.sum(dim=0)
                argmax_idx = flat.argmax(dim=-1)
                for k in range(num_experts):
                    argmax_sum[layer_idx][k] += float((argmax_idx == k).sum())
                per_token_entropy = -(flat * flat.clamp_min(1e-12).log()).sum(dim=-1)
                entropy_sum[layer_idx] += float(per_token_entropy.sum())
                token_count[layer_idx] += int(flat.size(0))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    layer_stats: dict[int, dict[str, Any]] = {}
    for layer_idx, _ in mole_blocks:
        tokens = token_count[layer_idx]
        if tokens == 0:
            continue
        layer_stats[layer_idx] = {
            "mean_softmax": (softmax_sum[layer_idx] / tokens).tolist(),
            "argmax_fraction": (argmax_sum[layer_idx] / tokens).tolist(),
            "mean_entropy": entropy_sum[layer_idx] / tokens,
            "tokens": tokens,
        }

    return {
        "per_layer": layer_stats,
        "aggregate": _aggregate(layer_stats),
    }
