#!/usr/bin/env python
"""Non-training dry checks for the LoRA/data/checkpoint code."""

from __future__ import annotations

import argparse
import tempfile
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.checkpointing import load_adapter_checkpoint, save_adapter_checkpoint
from lora_gpt2.config import load_config
from lora_gpt2.data import DataCollatorForCompletionOnlyLM, E2ERecord, E2ETokenizedDataset
from lora_gpt2.lora_layers import LoRAQKVConv1D


class TinyConv1D(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).view(*x.shape[:-1], -1)


class ToyTokenizer:
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) % 97 + 1 for char in text]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    base = TinyConv1D(8, 24)
    wrapper = LoRAQKVConv1D(base, rank=2, alpha=4, dropout=0.0)
    x = torch.randn(2, 3, 8)
    assert torch.allclose(wrapper(x), base(x), atol=1e-6)

    dataset = E2ETokenizedDataset(
        [E2ERecord("name[Blue Spice]", "Blue Spice is a restaurant.")],
        tokenizer=ToyTokenizer(),
        prompt_template=config["data"]["prompt_template"],
        max_length=64,
    )
    batch = DataCollatorForCompletionOnlyLM(pad_token_id=0)([dataset[0]])
    assert batch["labels"].shape == batch["input_ids"].shape

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir) / "adapter.pt"
        save_adapter_checkpoint(checkpoint, wrapper, config=config)
        missing, unexpected = load_adapter_checkpoint(checkpoint, wrapper, strict=False)
        assert not unexpected
        assert isinstance(missing, list)

    print("dry smoke checks passed")


if __name__ == "__main__":
    main()
