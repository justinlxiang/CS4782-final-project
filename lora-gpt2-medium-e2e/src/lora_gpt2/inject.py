"""Utilities for inserting LoRA modules into GPT-2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


_SLICE_ALIASES = {
    "query": "query",
    "q": "query",
    "key": "key",
    "k": "key",
    "value": "value",
    "v": "value",
}


def _layer_pattern(pattern: Any, layer_index: int) -> Mapping[str, Any] | None:
    """Return the configured slice pattern for one layer, if present."""
    missing = object()
    if pattern is None:
        return None
    if isinstance(pattern, Sequence) and not isinstance(pattern, (str, bytes, bytearray)):
        if layer_index >= len(pattern):
            return None
        layer_pattern = pattern[layer_index]
    elif isinstance(pattern, Mapping):
        layer_pattern = pattern.get(layer_index, missing)
        if layer_pattern is missing:
            layer_pattern = pattern.get(str(layer_index), missing)
        if layer_pattern is missing:
            layer_pattern = pattern.get(f"layer_{layer_index}", missing)
        if layer_pattern is missing:
            return None
    else:
        raise TypeError("LoRA rank/alpha patterns must be mappings or sequences.")

    if layer_pattern is None:
        return None
    if not isinstance(layer_pattern, Mapping):
        raise TypeError("Each LoRA layer pattern must be a mapping of slice names to values.")
    if "attention_c_attn" in layer_pattern:
        layer_pattern = layer_pattern["attention_c_attn"]
    if not isinstance(layer_pattern, Mapping):
        raise TypeError("The attention_c_attn LoRA layer pattern must be a mapping.")
    return layer_pattern


def _normalize_slice_pattern(pattern: Mapping[str, Any], value_type: type) -> dict[str, Any]:
    """Normalize q/k/v aliases to query/key/value keys."""
    normalized: dict[str, Any] = {}
    for raw_name, raw_value in pattern.items():
        name = _SLICE_ALIASES.get(str(raw_name).lower())
        if name is None:
            raise ValueError(f"Unknown LoRA slice name in rank pattern: {raw_name!r}.")
        normalized[name] = value_type(raw_value)
    return normalized


def _layer_rank_and_alpha_patterns(
    lora_config: dict[str, Any],
    layer_index: int,
) -> tuple[dict[str, int] | None, dict[str, float] | None]:
    """Return per-slice ranks and alphas for a layer, deriving alpha/r when needed."""
    rank_pattern = _layer_pattern(lora_config.get("rank_pattern"), layer_index)
    if rank_pattern is None:
        return None, None

    scalar_rank = int(lora_config.get("rank", 4))
    scalar_alpha = float(lora_config.get("alpha", 32))
    if scalar_rank <= 0:
        raise ValueError("Scalar lora.rank must be positive when deriving rank-pattern alpha.")

    normalized_ranks = _normalize_slice_pattern(rank_pattern, int)
    ranks = {
        name: int(normalized_ranks.get(name, 0))
        for name in LoRAQKVConv1D.slice_names
    }

    explicit_alpha_pattern = _layer_pattern(lora_config.get("alpha_pattern"), layer_index)
    explicit_alphas = (
        _normalize_slice_pattern(explicit_alpha_pattern, float)
        if explicit_alpha_pattern is not None
        else {}
    )
    base_scale = scalar_alpha / scalar_rank
    alphas = {
        name: float(
            explicit_alphas.get(name, base_scale * rank_value if rank_value > 0 else 0.0)
        )
        for name, rank_value in ranks.items()
    }
    return ranks, alphas


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
    for layer_index, block in enumerate(blocks):
        attention = block.attn
        if isinstance(attention.c_attn, LoRAQKVConv1D):
            continue
        rank_pattern, alpha_pattern = _layer_rank_and_alpha_patterns(lora_config, layer_index)
        attention.c_attn = LoRAQKVConv1D(
            base_layer=attention.c_attn,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            enable_lora=enable_lora,
            merge_weights=merge_weights,
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern,
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


def expected_gpt2_lora_parameters(
    config: dict[str, Any],
    hidden_size: int,
    num_layers: int,
) -> int:
    """Compute expected trainable LoRA params from scalar or rank-pattern config."""
    lora_config = config.get("lora", {})
    scalar_rank = int(lora_config.get("rank", 4))
    enable_lora = _target_booleans(config)
    if lora_config.get("rank_pattern") is None:
        return expected_qv_lora_parameters(hidden_size, num_layers, scalar_rank, enable_lora)

    total_rank_units = 0
    for layer_index in range(num_layers):
        rank_pattern, _alpha_pattern = _layer_rank_and_alpha_patterns(lora_config, layer_index)
        if rank_pattern is None:
            ranks = {name: scalar_rank for name in LoRAQKVConv1D.slice_names}
        else:
            ranks = rank_pattern
        for name, enabled in zip(LoRAQKVConv1D.slice_names, enable_lora):
            if enabled:
                total_rank_units += int(ranks.get(name, 0))
    return int(total_rank_units * (hidden_size + hidden_size))
