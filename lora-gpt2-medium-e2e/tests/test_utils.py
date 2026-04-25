from __future__ import annotations

import pytest
import torch

from lora_gpt2.utils import get_device


def test_get_device_accepts_cpu() -> None:
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_rejects_unavailable_mps(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(ValueError, match="MPS was requested"):
        get_device("mps")
