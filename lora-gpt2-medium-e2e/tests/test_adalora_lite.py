from __future__ import annotations

import torch
from torch import nn

from lora_gpt2.adalora_lite import AdaLoRALiteRankAllocator
from lora_gpt2.inject import inject_lora_into_gpt2


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


def _adalora_model() -> TinyGPT2:
    config = {
        "lora": {
            "method": "adalora_lite",
            "rank": 2,
            "alpha": 4,
            "dropout": 0.0,
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
    inject_lora_into_gpt2(model, config=config)
    return model


def test_allocator_keeps_global_top_k_components() -> None:
    model = _adalora_model()
    first = model.transformer.h[0].attn.c_attn
    second = model.transformer.h[1].attn.c_attn
    first.lora_s["query"].grad = torch.tensor([10.0, 9.0])
    first.lora_s["value"].grad = torch.tensor([1.0, 2.0])
    second.lora_s["query"].grad = torch.tensor([3.0, 4.0])
    second.lora_s["value"].grad = torch.tensor([8.0, 7.0])
    allocator = AdaLoRALiteRankAllocator(
        target_rank_units=3,
        mask_interval=1,
        init_warmup_steps=0,
        beta=0.0,
    )

    allocator.update_scores(model)
    summary = allocator.update_masks(model, global_step=1)

    assert summary is not None
    assert summary["active_rank_units"] == 3
    assert first.active_rank("query") == 2
    assert second.active_rank("value") == 1
    assert first.active_rank("value") == 0
    assert second.active_rank("query") == 0


def test_allocator_respects_warmup_before_masking() -> None:
    model = _adalora_model()
    first = model.transformer.h[0].attn.c_attn
    first.lora_s["query"].grad = torch.tensor([10.0, 9.0])
    allocator = AdaLoRALiteRankAllocator(
        target_rank_units=1,
        mask_interval=1,
        init_warmup_steps=5,
        beta=0.0,
    )

    allocator.update_scores(model)
    summary = allocator.update_masks(model, global_step=4)

    assert summary is None
    assert allocator.active_rank_units(model) == 8


def test_allocator_uses_deterministic_tie_breaking() -> None:
    model = _adalora_model()
    for _name, module in model.named_modules():
        if hasattr(module, "lora_s"):
            for slice_name in module.lora_s:
                module.lora_s[slice_name].grad = torch.zeros_like(module.lora_s[slice_name])
    allocator = AdaLoRALiteRankAllocator(
        target_rank_units=2,
        mask_interval=1,
        init_warmup_steps=0,
        beta=0.0,
    )

    allocator.update_scores(model)
    allocator.update_masks(model, global_step=1)

    first = model.transformer.h[0].attn.c_attn
    assert torch.allclose(first.lora_mask("query"), torch.tensor([1.0, 1.0]))
    assert allocator.active_rank_units(model) == 2


def test_allocator_state_dict_restores_ema_scores() -> None:
    model = _adalora_model()
    first = model.transformer.h[0].attn.c_attn
    first.lora_s["query"].grad = torch.tensor([10.0, 9.0])
    allocator = AdaLoRALiteRankAllocator(
        target_rank_units=1,
        mask_interval=1,
        init_warmup_steps=0,
        beta=0.0,
    )
    allocator.update_scores(model)

    restored = AdaLoRALiteRankAllocator(
        target_rank_units=1,
        mask_interval=1,
        init_warmup_steps=0,
        beta=0.0,
    )
    restored.load_state_dict(allocator.state_dict())

    assert restored.state_dict()["ema_scores"] == allocator.state_dict()["ema_scores"]
