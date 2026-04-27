"""Tests for the MoLE layer + injection.

Key invariants verified here without downloading GPT-2:

1. With the gate forced to a single expert, MoLE forward equals the
   single-adapter LoRA forward populated with that expert's weights.
   This is the *load correctness* test — if expert tensors land in the
   wrong slot or the gating math is off by a factor, this catches it.
2. With the gate at uniform (default init), MoLE forward equals the
   mean of the per-expert single-adapter forwards. This is the
   *mixing math* test.
3. After injection + `mark_only_mole_gate_as_trainable`, only router
   parameters have `requires_grad=True`.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from lora_gpt2.inject import inject_lora_into_gpt2
from lora_gpt2.lora_layers import LoRAQKVConv1D
from lora_gpt2.mole_inject import (
    force_expert_weights,
    inject_mole_into_gpt2,
    load_lora_experts_into_mole,
    mark_only_mole_gate_as_trainable,
)
from lora_gpt2.mole_layers import MoLEQKVConv1D


class _FakeConv1D(nn.Module):
    """Stand-in for HF GPT-2 Conv1D with the same weight layout."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class _FakeBlock(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()

        class _Attn(nn.Module):
            def __init__(self, c_attn: nn.Module) -> None:
                super().__init__()
                self.c_attn = c_attn

        self.attn = _Attn(_FakeConv1D(in_features, 3 * in_features))


class _FakeGPT2(nn.Module):
    def __init__(self, in_features: int = 32, num_layers: int = 2) -> None:
        super().__init__()

        class _Transformer(nn.Module):
            def __init__(self, blocks: nn.ModuleList) -> None:
                super().__init__()
                self.h = blocks

        self.transformer = _Transformer(
            nn.ModuleList([_FakeBlock(in_features) for _ in range(num_layers)])
        )


def _save_adapter(model: nn.Module, path: Path) -> None:
    """Mimic the real checkpointing format used by the project."""
    state = {
        name: param.detach().clone()
        for name, param in model.state_dict().items()
        if "lora_" in name
    }
    torch.save({"adapter_state_dict": state, "config": {}, "training_state": {}}, path)


def _make_lora_clone(
    in_features: int,
    num_layers: int,
    seed: int,
    shared_base: nn.Module | None = None,
) -> nn.Module:
    """Make a fresh LoRA-injected fake model with deterministic random LoRA weights.

    If `shared_base` is provided, copy its base Conv1D weights into the new
    model before injection so every clone has identical frozen-base output —
    isolating downstream comparisons to the LoRA delta path.
    """
    torch.manual_seed(seed)
    model = _FakeGPT2(in_features=in_features, num_layers=num_layers)
    if shared_base is not None:
        with torch.no_grad():
            for src_block, dst_block in zip(
                shared_base.transformer.h, model.transformer.h
            ):
                dst_block.attn.c_attn.weight.copy_(src_block.transformer_block_weight)
                dst_block.attn.c_attn.bias.copy_(src_block.transformer_block_bias)
    inject_lora_into_gpt2(model, rank=4, alpha=8, dropout=0.0, enable_lora=(True, False, True))
    # Inject randomizes A but zeros B; perturb B so the delta is non-trivial.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn_like(param) * 0.1)
    return model


def _build_shared_base(in_features: int, num_layers: int) -> _FakeGPT2:
    """Create a base GPT-2 stand-in whose c_attn weights every expert reuses."""
    torch.manual_seed(7)
    base = _FakeGPT2(in_features=in_features, num_layers=num_layers)
    # Stash the unwrapped Conv1D weights at a stable attribute so cloners
    # can grab them after each LoRA injection has wrapped c_attn.
    for block in base.transformer.h:
        block.transformer_block_weight = block.attn.c_attn.weight.detach().clone()
        block.transformer_block_bias = block.attn.c_attn.bias.detach().clone()
    return base


def test_force_single_expert_matches_single_adapter(tmp_path: Path) -> None:
    in_features = 32
    num_layers = 2
    num_experts = 3

    # All experts and the MoLE model share one base so output deltas are
    # the only source of difference — that isolates the test to the LoRA
    # mixing path, which is what we want to verify.
    shared = _build_shared_base(in_features, num_layers)

    expert_paths: list[Path] = []
    expert_models = []
    for idx in range(num_experts):
        m = _make_lora_clone(in_features, num_layers, seed=100 + idx, shared_base=shared)
        path = tmp_path / f"expert_{idx}.pt"
        _save_adapter(m, path)
        expert_paths.append(path)
        expert_models.append(m)

    mole_model = _FakeGPT2(in_features=in_features, num_layers=num_layers)
    with torch.no_grad():
        for src_block, dst_block in zip(shared.transformer.h, mole_model.transformer.h):
            dst_block.attn.c_attn.weight.copy_(src_block.transformer_block_weight)
            dst_block.attn.c_attn.bias.copy_(src_block.transformer_block_bias)

    inject_mole_into_gpt2(
        mole_model,
        num_experts=num_experts,
        rank=4,
        alpha=8,
        dropout=0.0,
        enable_lora=(True, False, True),
    )
    load_lora_experts_into_mole(mole_model, expert_paths)

    torch.manual_seed(0)
    x = torch.randn(2, 5, in_features)

    for k in range(num_experts):
        force_expert_weights(mole_model, expert_idx=k)
        mole_out = mole_model.transformer.h[0].attn.c_attn(x)
        single_out = expert_models[k].transformer.h[0].attn.c_attn(x)
        assert torch.allclose(mole_out, single_out, atol=1e-5), (
            f"MoLE forced to expert {k} did not match single adapter {k}"
        )


def test_uniform_gate_equals_mean_of_experts(tmp_path: Path) -> None:
    in_features = 32
    num_layers = 1
    num_experts = 3

    shared = _build_shared_base(in_features, num_layers)

    expert_paths: list[Path] = []
    expert_models = []
    for idx in range(num_experts):
        m = _make_lora_clone(in_features, num_layers, seed=200 + idx, shared_base=shared)
        path = tmp_path / f"expert_{idx}.pt"
        _save_adapter(m, path)
        expert_paths.append(path)
        expert_models.append(m)

    mole_model = _FakeGPT2(in_features=in_features, num_layers=num_layers)
    with torch.no_grad():
        for src_block, dst_block in zip(shared.transformer.h, mole_model.transformer.h):
            dst_block.attn.c_attn.weight.copy_(src_block.transformer_block_weight)
            dst_block.attn.c_attn.bias.copy_(src_block.transformer_block_bias)

    inject_mole_into_gpt2(
        mole_model,
        num_experts=num_experts,
        rank=4,
        alpha=8,
        dropout=0.0,
        enable_lora=(True, False, True),
        gate_init="uniform",
    )
    load_lora_experts_into_mole(mole_model, expert_paths)

    torch.manual_seed(1)
    x = torch.randn(1, 4, in_features)

    mole_out = mole_model.transformer.h[0].attn.c_attn(x)
    base = mole_model.transformer.h[0].attn.c_attn.base_layer(x)

    expert_outs = [m.transformer.h[0].attn.c_attn(x) for m in expert_models]
    expert_deltas = [out - base for out in expert_outs]
    mean_delta = sum(expert_deltas) / num_experts

    expected = base + mean_delta
    assert torch.allclose(mole_out, expected, atol=1e-5)


def test_only_gate_is_trainable() -> None:
    model = _FakeGPT2(in_features=16, num_layers=2)
    inject_mole_into_gpt2(
        model,
        num_experts=2,
        rank=4,
        alpha=8,
        dropout=0.0,
        enable_lora=(True, False, True),
    )

    # Re-assert the trainable filter (already done by inject) and inspect.
    mark_only_mole_gate_as_trainable(model)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable, "Expected at least one trainable router parameter."
    for name in trainable:
        assert "mole_gate_" in name, f"Unexpected trainable parameter: {name}"


def test_load_expert_shape_mismatch_raises() -> None:
    model = _FakeGPT2(in_features=32, num_layers=1)
    inject_mole_into_gpt2(
        model,
        num_experts=2,
        rank=4,
        alpha=8,
        dropout=0.0,
        enable_lora=(True, False, True),
    )
    mole = model.transformer.h[0].attn.c_attn
    bad_a = {"query": torch.zeros(2, 32), "value": torch.zeros(4, 32)}  # rank wrong
    bad_b = {"query": torch.zeros(32, 4), "value": torch.zeros(32, 4)}
    with pytest.raises(ValueError, match="A shape"):
        mole.load_expert(0, bad_a, bad_b)
