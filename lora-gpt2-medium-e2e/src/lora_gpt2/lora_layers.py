"""From-scratch LoRA layers used by the GPT-2 replication."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Wrap an `nn.Linear` layer with a trainable low-rank update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ) -> None:
        super().__init__()
        if rank < 0:
            raise ValueError("LoRA rank must be non-negative.")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank if self.rank > 0 else 0.0
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merge_weights = merge_weights
        self.merged = False

        for param in self.base_layer.parameters():
            param.requires_grad = False

        if self.rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.rank, base_layer.in_features))
            self.lora_B = nn.Parameter(torch.empty(base_layer.out_features, self.rank))
            self.reset_lora_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def reset_lora_parameters(self) -> None:
        """Initialize A randomly and B to zero so the initial delta is zero."""
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def lora_delta_weight(self) -> torch.Tensor:
        """Return the dense LoRA delta in `nn.Linear` weight layout."""
        if self.rank == 0:
            return torch.zeros_like(self.base_layer.weight)
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge(self) -> None:
        """Fold the LoRA delta into the frozen base weight."""
        if self.rank == 0 or self.merged:
            return
        with torch.no_grad():
            self.base_layer.weight += self.lora_delta_weight()
        self.merged = True

    def unmerge(self) -> None:
        """Remove a previously merged LoRA delta from the base weight."""
        if self.rank == 0 or not self.merged:
            return
        with torch.no_grad():
            self.base_layer.weight -= self.lora_delta_weight()
        self.merged = False

    def train(self, mode: bool = True) -> "LoRALinear":
        super().train(mode)
        if self.merge_weights:
            if mode:
                self.unmerge()
            else:
                self.merge()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.rank == 0 or self.merged:
            return base_out
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B) * self.scaling
        return base_out + lora_out


class LoRAQKVConv1D(nn.Module):
    """LoRA wrapper for Hugging Face GPT-2's fused `attn.c_attn` projection.

    GPT-2 stores Q, K, and V in one Conv1D-style projection whose weight layout
    is `(in_features, 3 * hidden_size)`. This wrapper leaves the frozen base
    projection intact and adds low-rank updates only to enabled output slices.
    """

    slice_names = ("query", "key", "value")

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        enable_lora: Iterable[bool] = (True, False, True),
        merge_weights: bool = False,
        rank_pattern: Mapping[str, int] | None = None,
        alpha_pattern: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(base_layer, "weight"):
            raise TypeError("Expected a GPT-2 Conv1D-like module with a weight.")
        if rank < 0:
            raise ValueError("LoRA rank must be non-negative.")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.rank_by_slice = {
            name: int(rank_pattern[name]) if rank_pattern and name in rank_pattern else self.rank
            for name in self.slice_names
        }
        for name, slice_rank in self.rank_by_slice.items():
            if slice_rank < 0:
                raise ValueError(f"LoRA rank for {name!r} must be non-negative.")
        self.alpha_by_slice = {
            name: (
                float(alpha_pattern[name])
                if alpha_pattern and name in alpha_pattern
                else self.alpha
            )
            for name in self.slice_names
        }
        self.scaling_by_slice = {
            name: self.alpha_by_slice[name] / slice_rank if slice_rank > 0 else 0.0
            for name, slice_rank in self.rank_by_slice.items()
        }
        self.scaling = self.alpha / self.rank if self.rank > 0 else 0.0
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merge_weights = merge_weights
        self.merged = False

        weight = self.base_layer.weight
        if weight.ndim != 2 or weight.shape[1] % 3 != 0:
            raise ValueError("Expected Conv1D weight shape `(in_features, 3 * hidden_size)`.")
        self.in_features = int(weight.shape[0])
        self.out_features = int(weight.shape[1])
        self.slice_size = self.out_features // 3

        enable = tuple(bool(value) for value in enable_lora)
        if len(enable) != 3:
            raise ValueError("`enable_lora` must contain query/key/value booleans.")
        self.enable_lora = enable
        self.enabled_slices = [
            (idx, name)
            for idx, (name, enabled) in enumerate(zip(self.slice_names, self.enable_lora))
            if enabled and self.rank_by_slice[name] > 0
        ]

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for _, name in self.enabled_slices:
            slice_rank = self.rank_by_slice[name]
            self.lora_A[name] = nn.Parameter(torch.empty(slice_rank, self.in_features))
            self.lora_B[name] = nn.Parameter(torch.empty(self.slice_size, slice_rank))
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        """Initialize enabled low-rank matrices."""
        for name in self.lora_A:
            nn.init.kaiming_uniform_(self.lora_A[name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[name])

    def _slice_delta(self, x: torch.Tensor, name: str) -> torch.Tensor:
        update = F.linear(x, self.lora_A[name])
        update = F.linear(update, self.lora_B[name]) * self.scaling_by_slice[name]
        return update

    def lora_delta_weight(self) -> torch.Tensor:
        """Return the dense LoRA delta in GPT-2 Conv1D weight layout."""
        delta = torch.zeros_like(self.base_layer.weight)
        if not self.enabled_slices:
            return delta
        for idx, name in self.enabled_slices:
            start = idx * self.slice_size
            end = start + self.slice_size
            slice_delta = (self.lora_B[name] @ self.lora_A[name]).T * self.scaling_by_slice[name]
            delta[:, start:end] = slice_delta
        return delta

    def merge(self) -> None:
        """Fold LoRA deltas into the base Conv1D weight."""
        if not self.enabled_slices or self.merged:
            return
        with torch.no_grad():
            self.base_layer.weight += self.lora_delta_weight()
        self.merged = True

    def unmerge(self) -> None:
        """Remove previously merged LoRA deltas."""
        if not self.enabled_slices or not self.merged:
            return
        with torch.no_grad():
            self.base_layer.weight -= self.lora_delta_weight()
        self.merged = False

    def train(self, mode: bool = True) -> "LoRAQKVConv1D":
        super().train(mode)
        if self.merge_weights:
            if mode:
                self.unmerge()
            else:
                self.merge()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.merged or not self.enabled_slices:
            return base_out

        lora_input = self.lora_dropout(x)
        chunks = list(base_out.split(self.slice_size, dim=-1))
        for idx, name in self.enabled_slices:
            chunks[idx] = chunks[idx] + self._slice_delta(lora_input, name)
        return torch.cat(chunks, dim=-1)


class AdaLoRAQKVConv1D(nn.Module):
    """AdaLoRA-Lite wrapper for GPT-2's fused `attn.c_attn` projection.

    This keeps the normal LoRA A/B factorization but adds one trainable scale
    and one non-trainable mask per rank component:

    `delta = B diag(s * mask) A`.
    """

    slice_names = LoRAQKVConv1D.slice_names

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        enable_lora: Iterable[bool] = (True, False, True),
        merge_weights: bool = False,
        layer_index: int | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(base_layer, "weight"):
            raise TypeError("Expected a GPT-2 Conv1D-like module with a weight.")
        if rank < 0:
            raise ValueError("AdaLoRA-Lite rank must be non-negative.")
        if merge_weights:
            raise ValueError("AdaLoRA-Lite does not support merge_weights.")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank if self.rank > 0 else 0.0
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merge_weights = False
        self.merged = False
        self.layer_index = layer_index

        weight = self.base_layer.weight
        if weight.ndim != 2 or weight.shape[1] % 3 != 0:
            raise ValueError("Expected Conv1D weight shape `(in_features, 3 * hidden_size)`.")
        self.in_features = int(weight.shape[0])
        self.out_features = int(weight.shape[1])
        self.slice_size = self.out_features // 3

        enable = tuple(bool(value) for value in enable_lora)
        if len(enable) != 3:
            raise ValueError("`enable_lora` must contain query/key/value booleans.")
        self.enable_lora = enable
        self.enabled_slices = [
            (idx, name)
            for idx, (name, enabled) in enumerate(zip(self.slice_names, self.enable_lora))
            if enabled and self.rank > 0
        ]

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.lora_s = nn.ParameterDict()
        for _, name in self.enabled_slices:
            self.lora_A[name] = nn.Parameter(torch.empty(self.rank, self.in_features))
            self.lora_B[name] = nn.Parameter(torch.empty(self.slice_size, self.rank))
            self.lora_s[name] = nn.Parameter(torch.ones(self.rank))
            self.register_buffer(f"lora_mask_{name}", torch.ones(self.rank))
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        """Initialize enabled low-rank matrices and adaptive scales."""
        for name in self.lora_A:
            nn.init.kaiming_uniform_(self.lora_A[name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[name])
            nn.init.ones_(self.lora_s[name])

    def lora_mask(self, name: str) -> torch.Tensor:
        """Return the non-trainable mask buffer for a slice."""
        return getattr(self, f"lora_mask_{name}")

    def set_lora_mask(self, name: str, mask: torch.Tensor) -> None:
        """Replace one slice mask with a 0/1 tensor."""
        if name not in self.lora_A:
            raise KeyError(f"Slice {name!r} is not enabled for AdaLoRA-Lite.")
        if mask.shape != (self.rank,):
            raise ValueError(f"Expected mask shape {(self.rank,)}, got {tuple(mask.shape)}.")
        current_mask = self.lora_mask(name)
        current_mask.copy_(mask.to(device=current_mask.device, dtype=current_mask.dtype))

    def active_rank(self, name: str) -> int:
        """Return the number of currently active rank components for a slice."""
        if name not in self.lora_A:
            return 0
        return int(self.lora_mask(name).detach().sum().item())

    def active_rank_by_slice(self) -> dict[str, int]:
        """Return active rank counts for query/key/value slices."""
        return {name: self.active_rank(name) for name in self.slice_names}

    def _slice_delta(self, x: torch.Tensor, name: str) -> torch.Tensor:
        update = F.linear(x, self.lora_A[name])
        gate = self.lora_s[name] * self.lora_mask(name).to(dtype=self.lora_s[name].dtype)
        update = update * gate
        update = F.linear(update, self.lora_B[name]) * self.scaling
        return update

    def lora_delta_weight(self) -> torch.Tensor:
        """Return the dense masked delta in GPT-2 Conv1D weight layout."""
        delta = torch.zeros_like(self.base_layer.weight)
        if not self.enabled_slices:
            return delta
        for idx, name in self.enabled_slices:
            start = idx * self.slice_size
            end = start + self.slice_size
            gate = self.lora_s[name] * self.lora_mask(name).to(dtype=self.lora_s[name].dtype)
            slice_delta = (self.lora_B[name] * gate.unsqueeze(0)) @ self.lora_A[name]
            delta[:, start:end] = slice_delta.T * self.scaling
        return delta

    def merge(self) -> None:
        """AdaLoRA-Lite intentionally keeps masks explicit during evaluation."""
        raise RuntimeError("AdaLoRA-Lite does not support merging into the base weights.")

    def unmerge(self) -> None:
        """AdaLoRA-Lite never merges weights, so unmerge is a no-op."""
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if not self.enabled_slices:
            return base_out

        lora_input = self.lora_dropout(x)
        chunks = list(base_out.split(self.slice_size, dim=-1))
        for idx, name in self.enabled_slices:
            chunks[idx] = chunks[idx] + self._slice_delta(lora_input, name)
        return torch.cat(chunks, dim=-1)
