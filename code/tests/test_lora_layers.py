from __future__ import annotations

import torch
from torch import nn

from lora_gpt2.lora_layers import AdaLoRAQKVConv1D, LoRALinear, LoRAQKVConv1D


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


def test_qkv_lora_supports_per_slice_rank_and_alpha() -> None:
    torch.manual_seed(0)
    base = TinyConv1D(8, 24)
    wrapper = LoRAQKVConv1D(
        base,
        rank=4,
        alpha=32,
        dropout=0.0,
        rank_pattern={"query": 2, "value": 6},
        alpha_pattern={"query": 16, "value": 48},
    )

    assert wrapper.lora_A["query"].shape == (2, 8)
    assert wrapper.lora_B["query"].shape == (8, 2)
    assert wrapper.lora_A["value"].shape == (6, 8)
    assert wrapper.lora_B["value"].shape == (8, 6)
    assert wrapper.scaling_by_slice["query"] == 8
    assert wrapper.scaling_by_slice["value"] == 8


def test_qkv_lora_rank_pattern_can_skip_enabled_slice() -> None:
    torch.manual_seed(0)
    base = TinyConv1D(8, 24)
    wrapper = LoRAQKVConv1D(
        base,
        rank=4,
        alpha=32,
        dropout=0.0,
        rank_pattern={"query": 0, "value": 3},
        alpha_pattern={"query": 0, "value": 24},
    )
    with torch.no_grad():
        wrapper.lora_B["value"].fill_(0.1)
    x = torch.randn(2, 3, 8)

    base_q, base_k, base_v = base(x).split(8, dim=-1)
    q, k, v = wrapper(x).split(8, dim=-1)

    assert "query" not in wrapper.lora_A
    assert torch.allclose(q, base_q, atol=1e-6)
    assert torch.allclose(k, base_k, atol=1e-6)
    assert not torch.allclose(v, base_v)


def test_qkv_lora_supports_unequal_slice_ranks_and_scaling() -> None:
    base = TinyConv1D(8, 24)
    wrapper = LoRAQKVConv1D(
        base,
        rank=4,
        alpha=32,
        dropout=0.0,
        rank_pattern={"query": 1, "key": 0, "value": 3},
        alpha_pattern={"query": 8, "key": 0, "value": 24},
    )

    assert wrapper.lora_A["query"].shape == (1, 8)
    assert wrapper.lora_B["query"].shape == (8, 1)
    assert "key" not in wrapper.lora_A
    assert wrapper.lora_A["value"].shape == (3, 8)
    assert wrapper.lora_B["value"].shape == (8, 3)
    assert wrapper.scaling_by_slice["query"] == 8
    assert wrapper.scaling_by_slice["value"] == 8


def test_qkv_lora_merge_matches_unmerged_eval_with_rank_pattern() -> None:
    torch.manual_seed(0)
    base = TinyConv1D(8, 24)
    wrapper = LoRAQKVConv1D(
        base,
        rank=4,
        alpha=32,
        dropout=0.0,
        merge_weights=True,
        rank_pattern={"query": 1, "key": 0, "value": 3},
        alpha_pattern={"query": 8, "key": 0, "value": 24},
    )
    with torch.no_grad():
        wrapper.lora_B["query"].normal_(mean=0.0, std=0.1)
        wrapper.lora_B["value"].normal_(mean=0.0, std=0.1)
    x = torch.randn(2, 3, 8)

    unmerged = wrapper(x)
    wrapper.eval()
    merged = wrapper(x)

    assert wrapper.merged
    assert torch.allclose(unmerged, merged, atol=1e-6)


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


def test_adalora_lite_qkv_initial_delta_is_zero() -> None:
    torch.manual_seed(0)
    base = TinyConv1D(8, 24)
    wrapper = AdaLoRAQKVConv1D(base, rank=4, alpha=32, dropout=0.0)
    x = torch.randn(2, 3, 8)

    assert torch.allclose(wrapper(x), base(x), atol=1e-6)


def test_adalora_lite_mask_removes_rank_components() -> None:
    torch.manual_seed(0)
    base = TinyConv1D(8, 24)
    wrapper = AdaLoRAQKVConv1D(base, rank=2, alpha=4, dropout=0.0)
    x = torch.randn(2, 3, 8)
    with torch.no_grad():
        wrapper.lora_B["query"][:, 0].fill_(0.25)
        wrapper.lora_B["query"][:, 1].fill_(0.50)
        wrapper.lora_s["query"].copy_(torch.tensor([1.0, 2.0]))

    full_q = wrapper(x).split(8, dim=-1)[0]
    wrapper.set_lora_mask("query", torch.tensor([1.0, 0.0]))
    one_component_q = wrapper(x).split(8, dim=-1)[0]
    wrapper.set_lora_mask("query", torch.tensor([0.0, 0.0]))
    zero_component_q = wrapper(x).split(8, dim=-1)[0]
    base_q = base(x).split(8, dim=-1)[0]

    assert not torch.allclose(full_q, one_component_q)
    assert not torch.allclose(one_component_q, zero_component_q)
    assert torch.allclose(zero_component_q, base_q, atol=1e-6)
