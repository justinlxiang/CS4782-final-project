"""Mixture-of-LoRA-Experts layers (Wu et al., ICLR 2024 — MoLE).

This module wraps GPT-2's fused `attn.c_attn` projection with N pretrained
LoRA experts plus a small per-layer router. Each expert is identical in shape
to a single `LoRAQKVConv1D` adapter — A and B matrices per enabled q/v slice
at the same rank — so we can load existing adapter checkpoints as-is and
freeze them. Only the router's parameters are trainable.

Forward, per enabled slice (query / value):

    delta_slice(x) = scaling * sum_i  gate(x)_i * (B_i @ A_i)(x)

where `gate(x)` is a per-token softmax over the N experts produced by a tiny
linear layer that takes the same input x. The base GPT-2 projection is
unchanged; the MoLE delta is added to its output exactly the way a single
LoRA adapter's delta is added.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class MoLEQKVConv1D(nn.Module):
    """N-expert mixture replacement for GPT-2's `c_attn` projection.

    Mirrors the public surface of `LoRAQKVConv1D` so the rest of the
    pipeline (`inject.py`, generation, checkpointing helpers) only needs
    to know which class to instantiate.
    """

    slice_names = ("query", "key", "value")

    # Supported `routing_mode` values:
    #   "soft": dense weighted-sum over all experts (the original MoLE
    #           formulation in this file). Every token attends to every
    #           expert with a softmax-derived weight.
    #   "top1": Switch-Transformer style hard routing — pick the
    #           argmax expert per token, scale its contribution by its
    #           own softmax probability so gradient flows to the gate
    #           through the multiplier. Non-chosen experts contribute 0.
    SUPPORTED_ROUTING_MODES = ("soft", "top1")

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float,
        num_experts: int,
        dropout: float = 0.0,
        enable_lora: Iterable[bool] = (True, False, True),
        gate_init: str = "uniform",
        routing_mode: str = "soft",
    ) -> None:
        super().__init__()
        if not hasattr(base_layer, "weight"):
            raise TypeError("Expected a GPT-2 Conv1D-like module with a weight.")
        if rank <= 0:
            raise ValueError("MoLE requires positive LoRA rank.")
        if num_experts < 1:
            raise ValueError("Need at least one expert.")
        if routing_mode not in self.SUPPORTED_ROUTING_MODES:
            raise ValueError(
                f"routing_mode {routing_mode!r} not in {self.SUPPORTED_ROUTING_MODES}"
            )

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.num_experts = int(num_experts)
        self.routing_mode = routing_mode
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

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
            if enabled
        ]

        # Frozen base; experts will be marked frozen too at the end of __init__.
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Per-expert LoRA params, named like the single-adapter checkpoints
        # (`lora_A.{slice}` etc.) but with an `_expert{i}` suffix on the slice
        # key so each expert remains independently addressable.
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for _, name in self.enabled_slices:
            for expert_idx in range(self.num_experts):
                key = self._expert_key(name, expert_idx)
                self.lora_A[key] = nn.Parameter(
                    torch.empty(self.rank, self.in_features), requires_grad=False
                )
                self.lora_B[key] = nn.Parameter(
                    torch.empty(self.slice_size, self.rank), requires_grad=False
                )
        self.reset_expert_parameters()

        # Per-layer router: linear from input activation to N expert logits.
        # The name `mole_gate_` is the trainable-parameter marker that
        # `mark_only_mole_gate_as_trainable` greps for, mirroring the
        # `lora_` convention used by the single-adapter pipeline.
        self.mole_gate_proj = nn.Linear(self.in_features, self.num_experts, bias=True)
        self._init_gate(gate_init)

    @staticmethod
    def _expert_key(slice_name: str, expert_idx: int) -> str:
        return f"{slice_name}_expert{expert_idx}"

    def reset_expert_parameters(self) -> None:
        """Zero-init B, Kaiming-init A for every expert (delta starts at 0).

        After loading pretrained adapters this is overwritten; this only
        matters for the freshly-injected, unloaded case (mainly tests).
        """
        with torch.no_grad():
            for _, name in self.enabled_slices:
                for expert_idx in range(self.num_experts):
                    key = self._expert_key(name, expert_idx)
                    nn.init.kaiming_uniform_(self.lora_A[key], a=math.sqrt(5))
                    nn.init.zeros_(self.lora_B[key])

    def _init_gate(self, mode: str) -> None:
        with torch.no_grad():
            if mode == "uniform":
                # Zero weights + zero bias => softmax produces 1/N for every
                # token regardless of input. Initial behavior is the mean of
                # the N experts, a neutral starting point for the router.
                nn.init.zeros_(self.mole_gate_proj.weight)
                nn.init.zeros_(self.mole_gate_proj.bias)
            elif mode == "random":
                nn.init.kaiming_uniform_(self.mole_gate_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.mole_gate_proj.bias)
            else:
                raise ValueError(f"Unknown gate_init: {mode!r}.")

    # ---- Loading helpers -------------------------------------------------

    @torch.no_grad()
    def load_expert(self, expert_idx: int, lora_A_by_slice: dict, lora_B_by_slice: dict) -> None:
        """Copy a single expert's per-slice (A, B) tensors into this module.

        `lora_A_by_slice` and `lora_B_by_slice` are dicts keyed by slice name
        ("query" / "value") with tensor values. Shapes must match the
        configured rank, in_features, and slice_size; otherwise we raise so
        a mismatched checkpoint fails loudly instead of silently miscopying.
        """
        if expert_idx < 0 or expert_idx >= self.num_experts:
            raise IndexError(f"expert_idx {expert_idx} out of range [0, {self.num_experts}).")
        for _, name in self.enabled_slices:
            if name not in lora_A_by_slice or name not in lora_B_by_slice:
                raise KeyError(f"Expert {expert_idx} missing slice {name!r}.")
            a_src = lora_A_by_slice[name]
            b_src = lora_B_by_slice[name]
            key = self._expert_key(name, expert_idx)
            target_a = self.lora_A[key]
            target_b = self.lora_B[key]
            if a_src.shape != target_a.shape:
                raise ValueError(
                    f"Expert {expert_idx} slice {name!r} A shape {tuple(a_src.shape)} "
                    f"!= expected {tuple(target_a.shape)}"
                )
            if b_src.shape != target_b.shape:
                raise ValueError(
                    f"Expert {expert_idx} slice {name!r} B shape {tuple(b_src.shape)} "
                    f"!= expected {tuple(target_b.shape)}"
                )
            target_a.copy_(a_src)
            target_b.copy_(b_src)

    # ---- Forward ---------------------------------------------------------

    def gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-token softmax weights over the N experts.

        Shape: `(*x.shape[:-1], num_experts)`. Used by the forward pass and
        also exposed so callers can log/inspect routing behavior at eval.
        """
        logits = self.mole_gate_proj(x)
        return F.softmax(logits, dim=-1)

    def _slice_mixture_delta(self, x: torch.Tensor, name: str, gates: torch.Tensor) -> torch.Tensor:
        """Compute the routed LoRA delta for one q/v slice.

        `gates` has shape (*, num_experts) and is the softmax over experts.
        Both routing modes still call every expert under the hood — at
        rank 4 the per-expert cost is negligible — and differ only in how
        the per-token outputs are combined:

        - soft:  Σ_i  gates_i * B_i(A_i(x))             (weighted sum)
        - top1:  gates[k] * B_k(A_k(x))   where k = argmax(gates)
                 Argmax is non-differentiable but we don't backprop
                 through it; gradient reaches the gate through the
                 surviving softmax probability multiplier (Switch-T style).
        """
        contributions = []
        for expert_idx in range(self.num_experts):
            key = self._expert_key(name, expert_idx)
            inner = F.linear(x, self.lora_A[key])           # (*, rank)
            outer = F.linear(inner, self.lora_B[key])       # (*, slice_size)
            contributions.append(outer)
        stacked = torch.stack(contributions, dim=-2)        # (*, num_experts, slice_size)

        if self.routing_mode == "soft":
            weights = gates.unsqueeze(-1)                   # (*, num_experts, 1)
            mixed = (stacked * weights).sum(dim=-2)
        elif self.routing_mode == "top1":
            top1_idx = gates.argmax(dim=-1, keepdim=True)   # (*, 1)
            top1_prob = gates.gather(-1, top1_idx)          # (*, 1) — keeps graph
            chosen = stacked.gather(
                dim=-2,
                index=top1_idx.unsqueeze(-1).expand(*top1_idx.shape, stacked.size(-1)),
            ).squeeze(-2)                                   # (*, slice_size)
            mixed = top1_prob * chosen
        else:
            # Defensive: should be unreachable given __init__ validation.
            raise RuntimeError(f"unknown routing_mode {self.routing_mode!r}")

        return mixed * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if not self.enabled_slices:
            return base_out

        lora_input = self.lora_dropout(x)
        gates = self.gate_weights(x)  # gate sees the un-dropped activation
        chunks = list(base_out.split(self.slice_size, dim=-1))
        for idx, name in self.enabled_slices:
            chunks[idx] = chunks[idx] + self._slice_mixture_delta(lora_input, name, gates)
        return torch.cat(chunks, dim=-1)
