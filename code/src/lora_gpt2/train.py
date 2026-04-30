"""Training utilities for the LoRA replication.

The CLI supports a dry-run mode; this module also contains real training
functions for later use once training is approved.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm


def create_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Create AdamW over trainable parameters only."""
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for optimizer.")
    training = config["training"]
    return torch.optim.AdamW(
        trainable_params,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, float(training.get("adam_beta2", 0.999))),
        eps=float(training.get("adam_epsilon", 1e-6)),
    )


def create_linear_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
):
    """Create a linear warmup/decay scheduler."""
    from transformers import get_linear_schedule_with_warmup

    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


def shifted_label_smoothed_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Causal LM loss with optional label smoothing and ignored labels."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    if label_smoothing <= 0:
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index,
        )

    vocab_size = shift_logits.size(-1)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    valid_mask = shift_labels.ne(ignore_index)
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)

    nll = -log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    smooth = -log_probs.mean(dim=-1)
    loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    loss = loss.masked_select(valid_mask)
    if loss.numel() == 0:
        return shift_logits.sum() * 0.0
    return loss.mean()


def forward_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute causal LM loss with the project's completion-only labels."""
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
    )
    return shifted_label_smoothed_loss(outputs.logits, batch["labels"], label_smoothing)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config: dict[str, Any],
    progress: bool = True,
    progress_desc: str = "train",
) -> float:
    """Run one training epoch. Do not call this until training is approved.

    With ``progress=True`` (default) we wrap the dataloader in a tqdm bar
    that displays a running mean loss. The single-task `scripts/train.py`
    drives its own loop so this tqdm is for callers that use this helper
    directly (currently `scripts/train_mole.py` and any future scripts).
    """
    model.train()
    total_loss = 0.0
    steps = 0
    grad_accum = int(config["training"].get("gradient_accumulation_steps", 1))
    label_smoothing = float(config["training"].get("label_smoothing", 0.0))
    max_grad_norm = float(config["training"].get("max_grad_norm", 0.0))

    optimizer.zero_grad(set_to_none=True)
    if progress:
        try:
            total_steps = len(dataloader)
        except TypeError:
            total_steps = None
        iterator = tqdm(dataloader, total=total_steps, desc=progress_desc, unit="step")
    else:
        iterator = dataloader
    for step, batch in enumerate(iterator, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        loss = forward_loss(model, batch, label_smoothing=label_smoothing)
        (loss / grad_accum).backward()

        if step % grad_accum == 0:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    max_grad_norm,
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu())
        steps += 1
        if progress and isinstance(iterator, tqdm):
            iterator.set_postfix(loss=f"{total_loss / steps:.4f}")

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    label_smoothing: float = 0.0,
    progress: bool = True,
    progress_desc: str = "eval",
) -> float:
    """Evaluate average validation loss without updating parameters."""
    model.eval()
    total_loss = 0.0
    steps = 0
    if progress:
        try:
            total_steps = len(dataloader)
        except TypeError:
            total_steps = None
        iterator = tqdm(dataloader, total=total_steps, desc=progress_desc, unit="step", leave=False)
    else:
        iterator = dataloader
    for batch in iterator:
        batch = {key: value.to(device) for key, value in batch.items()}
        loss = forward_loss(model, batch, label_smoothing=label_smoothing)
        total_loss += float(loss.detach().cpu())
        steps += 1
        if progress and isinstance(iterator, tqdm):
            iterator.set_postfix(loss=f"{total_loss / steps:.4f}")
    return total_loss / max(steps, 1)
