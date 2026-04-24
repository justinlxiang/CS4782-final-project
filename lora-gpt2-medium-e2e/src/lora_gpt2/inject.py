"""Utilities for inserting LoRA modules into GPT-2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lora_gpt2.lora_layers import LoRAQKVConv1D
from lora_gpt2.utils import count_parameters


@dataclass(frozen=True)
class InjectionReport:
    """Summary of a GPT-2 LoRA injection pass."""

    replaced_modules: int
    total_parameters: int
    trainable_parameters: int
    trainable_names: list[str]


def freeze_base_model(model: torch.nn.Module) -> None:
    """Freeze every parameter in a model."""
    for param in model.parameters():
        param.requires_grad = False


def mark_only_lora_as_trainable(model: torch.nn.Module, bias: str = "none") -> None:
    """Freeze every non-LoRA parameter, matching the official LoRA convention."""
    if bias not in {"none", "all", "lora_only"}:
        raise ValueError("bias must be one of: 'none', 'all', 'lora_only'.")

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    if bias == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
    elif bias == "lora_only":
        for module in model.modules():
            if isinstance(module, LoRAQKVConv1D) and hasattr(module.base_layer, "bias"):
                bias_param = module.base_layer.bias
                if bias_param is not None:
                    bias_param.requires_grad = True


def lora_state_dict(model: torch.nn.Module, bias: str = "none") -> dict[str, torch.Tensor]:
    """Return a state dict containing only adapter parameters by default."""
    if bias not in {"none", "all"}:
        raise ValueError("Only 'none' and 'all' bias modes are supported here.")
    state = model.state_dict()
    if bias == "all":
        return {key: value for key, value in state.items() if "lora_" in key or "bias" in key}
    return {key: value for key, value in state.items() if "lora_" in key}


def _target_booleans(config: dict[str, Any] | None) -> tuple[bool, bool, bool]:
    if not config:
        return (True, False, True)
    target = config.get("lora", {}).get("target_modules", {}).get("attention_c_attn", {})
    return (
        bool(target.get("query", True)),
        bool(target.get("key", False)),
        bool(target.get("value", True)),
    )


def inject_lora_into_gpt2(
    model: torch.nn.Module,
    rank: int | None = None,
    alpha: float | None = None,
    dropout: float | None = None,
    enable_lora: tuple[bool, bool, bool] | None = None,
    merge_weights: bool | None = None,
    config: dict[str, Any] | None = None,
) -> InjectionReport:
    """Replace each GPT-2 attention `c_attn` with a Q/K/V-aware LoRA wrapper."""
    lora_config = config.get("lora", {}) if config else {}
    rank = int(rank if rank is not None else lora_config.get("rank", 4))
    alpha = float(alpha if alpha is not None else lora_config.get("alpha", 32))
    dropout = float(dropout if dropout is not None else lora_config.get("dropout", 0.1))
    merge_weights = bool(
        merge_weights if merge_weights is not None else lora_config.get("merge_for_eval", False)
    )
    enable_lora = enable_lora if enable_lora is not None else _target_booleans(config)

    freeze_base_model(model)

    try:
        blocks = model.transformer.h
    except AttributeError as exc:
        raise TypeError("Expected a Hugging Face GPT-2 style model with `transformer.h`.") from exc

    replaced = 0
    for block in blocks:
        attention = block.attn
        if isinstance(attention.c_attn, LoRAQKVConv1D):
            continue
        attention.c_attn = LoRAQKVConv1D(
            base_layer=attention.c_attn,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            enable_lora=enable_lora,
            merge_weights=merge_weights,
        )
        replaced += 1

    mark_only_lora_as_trainable(model, bias="none")
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    return InjectionReport(
        replaced_modules=replaced,
        total_parameters=count_parameters(model),
        trainable_parameters=count_parameters(model, trainable_only=True),
        trainable_names=trainable_names,
    )


def expected_qv_lora_parameters(
    hidden_size: int,
    num_layers: int,
    rank: int,
    enable_lora: tuple[bool, bool, bool] = (True, False, True),
) -> int:
    """Compute expected LoRA params for GPT-2 fused QKV slices."""
    enabled_count = sum(1 for enabled in enable_lora if enabled)
    return int(num_layers * enabled_count * (rank * hidden_size + hidden_size * rank))
