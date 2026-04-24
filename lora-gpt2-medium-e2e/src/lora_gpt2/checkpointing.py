"""Adapter-only checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lora_gpt2.inject import lora_state_dict
from lora_gpt2.utils import ensure_dir


def save_adapter_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save only LoRA adapter weights and lightweight metadata."""
    checkpoint_path = Path(path)
    ensure_dir(checkpoint_path.parent)
    payload = {
        "adapter_state_dict": lora_state_dict(model),
        "config": config or {},
        "extra": extra or {},
    }
    torch.save(payload, checkpoint_path)


def load_adapter_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load adapter weights into an already-created LoRA-injected model."""
    payload = torch.load(Path(path), map_location="cpu")
    state = payload.get("adapter_state_dict", payload)
    result = model.load_state_dict(state, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)
