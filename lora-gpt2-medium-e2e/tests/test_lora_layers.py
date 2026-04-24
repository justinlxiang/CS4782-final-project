from __future__ import annotations

import torch
from torch import nn

from lora_gpt2.lora_layers import LoRALinear, LoRAQKVConv1D


class TinyConv1D(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.weight.shape[1],)
        out = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return out.view(size_out)


def test_qkv_lora_initial_delta_is_zero() -> None:
    torch.manual_seed(0)
    base = TinyConv1D(8, 24)
    wrapper = LoRAQKVConv1D(base, rank=2, alpha=4, dropout=0.0)
    x = torch.randn(2, 3, 8)

    assert torch.allclose(wrapper(x), base(x), atol=1e-6)


def test_qkv_lora_leaves_disabled_key_slice_unchanged() -> None:
    torch.manual_seed(0)
    base = TinyConv1D(8, 24)
    wrapper = LoRAQKVConv1D(base, rank=2, alpha=4, dropout=0.0)
    with torch.no_grad():
        wrapper.lora_B["query"].fill_(0.1)
        wrapper.lora_B["value"].fill_(0.1)
    x = torch.randn(2, 3, 8)

    base_q, base_k, base_v = base(x).split(8, dim=-1)
    q, k, v = wrapper(x).split(8, dim=-1)

    assert not torch.allclose(q, base_q)
    assert torch.allclose(k, base_k, atol=1e-6)
    assert not torch.allclose(v, base_v)


def test_lora_linear_merge_matches_unmerged_eval() -> None:
    torch.manual_seed(0)
    base = nn.Linear(5, 7)
    wrapper = LoRALinear(base, rank=2, alpha=4, dropout=0.0, merge_weights=True)
    with torch.no_grad():
        wrapper.lora_B.normal_(mean=0.0, std=0.1)
    x = torch.randn(4, 5)

    unmerged = wrapper(x)
    wrapper.eval()
    merged = wrapper(x)

    assert wrapper.merged
    assert torch.allclose(unmerged, merged, atol=1e-6)
