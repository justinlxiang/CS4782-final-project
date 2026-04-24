from __future__ import annotations

import torch
from torch import nn

from lora_gpt2.inject import expected_qv_lora_parameters, inject_lora_into_gpt2
from lora_gpt2.lora_layers import LoRAQKVConv1D


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
