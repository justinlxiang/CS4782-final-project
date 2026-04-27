"""Injection and expert-loading for Mixture-of-LoRA-Experts on GPT-2.

The single-adapter `inject_lora_into_gpt2` replaces each `c_attn` with a
`LoRAQKVConv1D`. This module replaces each `c_attn` with a `MoLEQKVConv1D`
and copies pretrained adapter weights from N existing checkpoints into the
N expert slots, leaving them frozen so only the per-layer router is trained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lora_gpt2.inject import freeze_base_model
from lora_gpt2.mole_layers import MoLEQKVConv1D
from lora_gpt2.utils import count_parameters


@dataclass(frozen=True)
class MoLEInjectionReport:
    replaced_modules: int
    num_experts: int
    total_parameters: int
    trainable_parameters: int
    trainable_names: list[str]


def mark_only_mole_gate_as_trainable(model: torch.nn.Module) -> None:
    """Freeze everything except per-layer router parameters.

    The router params are the only ones whose names contain
    `mole_gate_` (see `MoLEQKVConv1D._init_gate`). Experts share the
    `lora_` namespace with single-adapter LoRA but are pinned via
    `requires_grad=False` at construction; this helper preserves that.
    """
    for name, param in model.named_parameters():
        param.requires_grad = "mole_gate_" in name


def mole_gate_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return only router parameters (the trainable subset of MoLE)."""
    state = model.state_dict()
    return {key: value for key, value in state.items() if "mole_gate_" in key}


def _target_booleans(config: dict[str, Any] | None) -> tuple[bool, bool, bool]:
    if not config:
        return (True, False, True)
    target = config.get("lora", {}).get("target_modules", {}).get("attention_c_attn", {})
    return (
        bool(target.get("query", True)),
        bool(target.get("key", False)),
        bool(target.get("value", True)),
    )


def inject_mole_into_gpt2(
    model: torch.nn.Module,
    num_experts: int,
    rank: int | None = None,
    alpha: float | None = None,
    dropout: float | None = None,
    enable_lora: tuple[bool, bool, bool] | None = None,
    gate_init: str | None = None,
    routing_mode: str | None = None,
    config: dict[str, Any] | None = None,
) -> MoLEInjectionReport:
    """Replace each GPT-2 attention `c_attn` with a `MoLEQKVConv1D`.

    Mirrors `inject_lora_into_gpt2` but with N expert slots per layer.
    All experts and base parameters end up frozen; only the router is
    trainable, which matches the MoLE training recipe.
    """
    lora_config = config.get("lora", {}) if config else {}
    mole_config = config.get("mole", {}) if config else {}
    rank = int(rank if rank is not None else lora_config.get("rank", 4))
    alpha = float(alpha if alpha is not None else lora_config.get("alpha", 32))
    dropout = float(dropout if dropout is not None else lora_config.get("dropout", 0.0))
    enable_lora = enable_lora if enable_lora is not None else _target_booleans(config)
    routing_mode = routing_mode if routing_mode is not None else mole_config.get("routing_mode", "soft")
    # Default top-1 to a non-degenerate gate init: zero-init weights make
    # argmax always pick expert 0 (PyTorch ties → first index), so the
    # gate has nothing to learn from. Random init breaks ties.
    if gate_init is None:
        gate_init = mole_config.get("gate_init") or ("random" if routing_mode == "top1" else "uniform")

    freeze_base_model(model)

    try:
        blocks = model.transformer.h
    except AttributeError as exc:
        raise TypeError("Expected a Hugging Face GPT-2 style model with `transformer.h`.") from exc

    replaced = 0
    for block in blocks:
        attention = block.attn
        if isinstance(attention.c_attn, MoLEQKVConv1D):
            continue
        attention.c_attn = MoLEQKVConv1D(
            base_layer=attention.c_attn,
            rank=rank,
            alpha=alpha,
            num_experts=num_experts,
            dropout=dropout,
            enable_lora=enable_lora,
            gate_init=gate_init,
            routing_mode=routing_mode,
        )
        replaced += 1

    mark_only_mole_gate_as_trainable(model)
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    return MoLEInjectionReport(
        replaced_modules=replaced,
        num_experts=int(num_experts),
        total_parameters=count_parameters(model),
        trainable_parameters=count_parameters(model, trainable_only=True),
        trainable_names=trainable_names,
    )


# Adapter checkpoint keys look like:
#   transformer.h.{layer}.attn.c_attn.lora_A.{slice}
#   transformer.h.{layer}.attn.c_attn.lora_B.{slice}
# We need the layer index + AB type + slice name to route a tensor into
# the right MoLE expert slot, so a small regex parser keeps this readable.
_ADAPTER_KEY_RE = re.compile(
    r"transformer\.h\.(?P<layer>\d+)\.attn\.c_attn\.lora_(?P<ab>[AB])\.(?P<slice>\w+)$"
)


def _parse_adapter_keys(state_dict: dict[str, torch.Tensor]) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
    """Group an adapter checkpoint by layer index, then by AB, then by slice."""
    grouped: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for key, value in state_dict.items():
        match = _ADAPTER_KEY_RE.search(key)
        if not match:
            continue
        layer = int(match.group("layer"))
        ab = match.group("ab")
        slice_name = match.group("slice")
        grouped.setdefault(layer, {"A": {}, "B": {}})[ab][slice_name] = value
    return grouped


def load_lora_experts_into_mole(
    model: torch.nn.Module,
    expert_paths: list[str | Path],
    map_location: str | torch.device = "cpu",
) -> list[int]:
    """Populate every MoLE module with weights from N adapter checkpoints.

    `expert_paths[i]` is loaded into expert slot `i` of every MoLE block,
    so the index ordering you pass here is the index ordering the trained
    router will reference. We return the list of MoLE-block layer indices
    that were populated, useful for sanity checks in tests/scripts.
    """
    mole_blocks: list[tuple[int, MoLEQKVConv1D]] = []
    for layer_idx, block in enumerate(model.transformer.h):
        if isinstance(block.attn.c_attn, MoLEQKVConv1D):
            mole_blocks.append((layer_idx, block.attn.c_attn))
    if not mole_blocks:
        raise RuntimeError("No MoLEQKVConv1D modules found; did you call inject_mole_into_gpt2?")

    num_experts_in_model = mole_blocks[0][1].num_experts
    if num_experts_in_model != len(expert_paths):
        raise ValueError(
            f"Model has {num_experts_in_model} expert slots per layer but "
            f"{len(expert_paths)} expert checkpoints were provided."
        )

    for expert_idx, path in enumerate(expert_paths):
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        state = payload.get("adapter_state_dict", payload)
        grouped = _parse_adapter_keys(state)
        if not grouped:
            raise ValueError(f"No matching adapter keys parsed from {path}.")
        for layer_idx, mole in mole_blocks:
            if layer_idx not in grouped:
                raise KeyError(f"Adapter at {path} missing layer {layer_idx}.")
            mole.load_expert(
                expert_idx=expert_idx,
                lora_A_by_slice=grouped[layer_idx]["A"],
                lora_B_by_slice=grouped[layer_idx]["B"],
            )

    return [layer_idx for layer_idx, _ in mole_blocks]


def force_expert_weights(model: torch.nn.Module, expert_idx: int) -> None:
    """Pin the gate to a single expert (eval-time ablation tool).

    Sets each router's bias so softmax outputs ~1.0 on `expert_idx` and ~0
    elsewhere; useful for checking that any single expert reproduces its
    standalone adapter's behavior, and for ablating routing entirely.
    """
    for module in model.modules():
        if isinstance(module, MoLEQKVConv1D):
            with torch.no_grad():
                module.mole_gate_proj.weight.zero_()
                bias = module.mole_gate_proj.bias
                bias.zero_()
                # 50.0 is large enough to make softmax effectively one-hot
                # in fp32/fp16 without causing NaNs.
                bias[expert_idx] = 50.0
