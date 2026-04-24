from __future__ import annotations

import torch
from torch import nn

from lora_gpt2.checkpointing import load_adapter_checkpoint, save_adapter_checkpoint
from lora_gpt2.inject import lora_state_dict
from lora_gpt2.lora_layers import LoRAQKVConv1D


class TinyConv1D(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return out.view(*x.shape[:-1], self.weight.shape[1])


def test_lora_state_dict_excludes_base_weights() -> None:
    wrapper = LoRAQKVConv1D(TinyConv1D(8, 24), rank=2, alpha=4, dropout=0.0)
    state = lora_state_dict(wrapper)

    assert state
    assert all("lora_" in key for key in state)
    assert all("base_layer" not in key for key in state)


def test_adapter_checkpoint_round_trip(tmp_path) -> None:
    source = LoRAQKVConv1D(TinyConv1D(8, 24), rank=2, alpha=4, dropout=0.0)
    target = LoRAQKVConv1D(TinyConv1D(8, 24), rank=2, alpha=4, dropout=0.0)
    with torch.no_grad():
        source.lora_B["query"].fill_(0.5)

    checkpoint = tmp_path / "adapter.pt"
    save_adapter_checkpoint(checkpoint, source, config={"test": True})
    _missing, unexpected = load_adapter_checkpoint(checkpoint, target, strict=False)

    assert not unexpected
    assert torch.allclose(target.lora_B["query"], source.lora_B["query"])
