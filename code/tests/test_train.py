from __future__ import annotations

import torch

from lora_gpt2.train import shifted_label_smoothed_loss


def test_shifted_label_smoothed_loss_ignores_prompt_positions() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor(
        [
            [-100, -100, 3, 4],
            [-100, 2, -100, 5],
        ]
    )

    loss = shifted_label_smoothed_loss(logits, labels, label_smoothing=0.1)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
