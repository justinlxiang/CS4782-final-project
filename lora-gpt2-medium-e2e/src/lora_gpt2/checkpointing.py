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


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: dict[str, Any],
    training_state: dict[str, Any],
) -> None:
    """Save adapter weights plus optimizer/scheduler state for exact resume."""
    checkpoint_path = Path(path)
    ensure_dir(checkpoint_path.parent)
    payload = {
        "adapter_state_dict": lora_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "training_state": training_state,
    }
    torch.save(payload, checkpoint_path)


def load_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Load a full training checkpoint and return its training state."""
    payload = torch.load(Path(path), map_location="cpu")
    adapter_state = payload.get("adapter_state_dict", payload)
    model.load_state_dict(adapter_state, strict=strict)

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return dict(payload.get("training_state", {}))
