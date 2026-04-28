from __future__ import annotations

import torch
from torch import nn

from lora_gpt2.inject import (
    expected_qv_adalora_lite_parameters,
    expected_gpt2_lora_parameters,
    expected_qv_lora_parameters,
    inject_lora_into_gpt2,
)
from lora_gpt2.lora_layers import AdaLoRAQKVConv1D, LoRAQKVConv1D


class TinyConv1D(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return out.view(*x.shape[:-1], self.weight.shape[1])


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_attn = TinyConv1D(8, 24)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = TinyAttention()


class TinyTransformer(nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.h = nn.ModuleList(TinyBlock() for _ in range(layers))


class TinyGPT2(nn.Module):
    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        self.transformer = TinyTransformer(layers)


def test_inject_replaces_all_c_attn_modules_and_freezes_base() -> None:
    model = TinyGPT2(layers=2)
    report = inject_lora_into_gpt2(model, rank=2, alpha=4, dropout=0.0)

    assert report.replaced_modules == 2
    assert all(isinstance(block.attn.c_attn, LoRAQKVConv1D) for block in model.transformer.h)
    assert report.trainable_parameters == expected_qv_lora_parameters(8, 2, 2)
    assert report.trainable_parameters < report.total_parameters
    assert report.trainable_names
    assert all("lora_" in name for name in report.trainable_names)


def test_inject_rank_pattern_matches_expected_parameter_budget() -> None:
    config = {
        "lora": {
            "rank": 4,
            "alpha": 32,
            "dropout": 0.0,
            "target_modules": {
                "attention_c_attn": {
                    "query": True,
                    "key": False,
                    "value": True,
                }
            },
            "rank_pattern": [
                {"query": 1, "value": 3},
                {"query": 3, "value": 1},
            ],
        }
    }
    model = TinyGPT2(layers=2)
    report = inject_lora_into_gpt2(model, config=config)

    assert report.trainable_parameters == expected_gpt2_lora_parameters(
        config,
        hidden_size=8,
        num_layers=2,
    )
    assert report.trainable_parameters == expected_qv_lora_parameters(8, 2, 2)
    assert model.transformer.h[0].attn.c_attn.scaling_by_slice["query"] == 8
    assert model.transformer.h[0].attn.c_attn.scaling_by_slice["value"] == 8
    assert model.transformer.h[1].attn.c_attn.scaling_by_slice["query"] == 8
    assert model.transformer.h[1].attn.c_attn.scaling_by_slice["value"] == 8


def test_rank_pattern_can_skip_zero_rank_slice() -> None:
    config = {
        "lora": {
            "rank": 4,
            "alpha": 32,
            "dropout": 0.0,
            "rank_pattern": [
                {"query": 0, "value": 4},
            ],
        }
    }
    model = TinyGPT2(layers=1)
    report = inject_lora_into_gpt2(model, config=config)
    wrapper = model.transformer.h[0].attn.c_attn

    assert report.trainable_parameters == 64
    assert "query" not in wrapper.lora_A
    assert "value" in wrapper.lora_A


def test_inject_adalora_lite_uses_adaptive_qkv_wrapper() -> None:
    config = {
        "lora": {
            "method": "adalora_lite",
            "rank": 4,
            "alpha": 32,
            "dropout": 0.0,
            "merge_for_eval": False,
            "target_modules": {
                "attention_c_attn": {
                    "query": True,
                    "key": False,
                    "value": True,
                }
            },
        }
    }
    model = TinyGPT2(layers=2)
    report = inject_lora_into_gpt2(model, config=config)

    assert report.replaced_modules == 2
    assert all(isinstance(block.attn.c_attn, AdaLoRAQKVConv1D) for block in model.transformer.h)
    assert report.trainable_parameters == expected_qv_adalora_lite_parameters(8, 2, 4)
    assert all("lora_" in name for name in report.trainable_names)
    assert any("lora_s" in name for name in report.trainable_names)
    assert model.transformer.h[0].attn.c_attn.active_rank("query") == 4
