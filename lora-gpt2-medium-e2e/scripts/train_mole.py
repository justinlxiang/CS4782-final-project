#!/usr/bin/env python
"""Train the MoLE router on top of N frozen LoRA experts.

Usage (after the per-task experts are trained and processed-data dirs exist):

    python scripts/train_mole.py --config configs/mole_e2e_dart_webnlg.yaml --train --device cuda

Without `--train` the script does a dry run: loads GPT-2, injects the MoLE
modules, copies in the expert weights, and prints router parameter counts
without taking any optimizer steps. That dry run is the cheapest way to
verify expert checkpoints load against the configured rank/alpha/targets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from lora_gpt2.config import load_config, resolve_path
from lora_gpt2.data import DataCollatorForCompletionOnlyLM
from lora_gpt2.modeling import load_base_model, load_tokenizer
from lora_gpt2.mole_inject import (
    inject_mole_into_gpt2,
    load_lora_experts_into_mole,
    mole_gate_state_dict,
)
from lora_gpt2.train import (
    create_linear_scheduler,
    create_optimizer,
    evaluate_loss,
    train_one_epoch,
)
from lora_gpt2.utils import count_parameters, ensure_dir, read_jsonl


class _PreTokenizedJSONLDataset(Dataset):
    """Tiny adapter around `read_jsonl` for already-tokenized example files.

    The processed JSONL produced by `prepare_e2e.py` carries `input_ids`,
    `labels`, and `prompt_length` per row; the collator only needs the
    first two plus an attention mask, which we synthesize here.
    """

    def __init__(self, path: Path) -> None:
        self.records = list(read_jsonl(path))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        record = self.records[idx]
        input_ids = list(record["input_ids"])
        return {
            "input_ids": input_ids,
            "labels": list(record["labels"]),
            "attention_mask": [1] * len(input_ids),
        }


def _build_mixed_dataset(config: dict, split: str) -> Dataset:
    """Concatenate per-task processed splits into one dataset.

    The router learns from data alone — there is no task ID label — so the
    mixed dataset is simply the union of all per-task examples shuffled
    together by the dataloader. If a task's split file is missing we skip
    that task with a warning rather than failing the whole run.
    """
    file_key = "train_file" if split == "train" else "valid_file"
    parts: list[Dataset] = []
    for task in config["data"]["task_splits"]:
        processed_dir = resolve_path(config, task["processed_dir"])
        path = processed_dir / task[file_key]
        if not path.exists():
            print(f"warn: missing {path} for task {task['name']!r}; skipping.")
            continue
        parts.append(_PreTokenizedJSONLDataset(path))
        print(f"loaded {len(parts[-1])} {split} examples from {path}")
    if not parts:
        raise FileNotFoundError(f"No {split} files found for any task in config.")
    return ConcatDataset(parts)


def _resolve_expert_paths(config: dict) -> list[Path]:
    return [resolve_path(config, path) for path in config["mole"]["expert_paths"]]


def _save_router(path: Path, model: torch.nn.Module, config: dict) -> None:
    """Persist only the router state. Experts are reproducible from disk."""
    ensure_dir(path.parent)
    payload = {
        "router_state_dict": mole_gate_state_dict(model),
        "config": config,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", action="store_true", help="Run real training.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(resolve_path(config, config["project"]["output_dir"]))

    print("loading base model + tokenizer...")
    tokenizer = load_tokenizer(config)
    model = load_base_model(config)

    expert_paths = _resolve_expert_paths(config)
    num_experts = len(expert_paths)
    expert_names = config["mole"].get("expert_names", [str(i) for i in range(num_experts)])
    print(f"injecting MoLE with {num_experts} experts: {expert_names}")
    # gate_init / routing_mode are read from config["mole"] inside the
    # injection helper so the same code path works for both modes.
    report = inject_mole_into_gpt2(
        model,
        num_experts=num_experts,
        config=config,
    )
    routing_mode = config["mole"].get("routing_mode", "soft")
    print(f"replaced {report.replaced_modules} c_attn modules "
          f"(routing_mode={routing_mode}); "
          f"router params: {report.trainable_parameters:,} / "
          f"total: {report.total_parameters:,}")

    print("loading expert checkpoints...")
    populated_layers = load_lora_experts_into_mole(model, expert_paths)
    print(f"populated {len(populated_layers)} MoLE layers from {num_experts} adapter files")

    if not args.train:
        print("dry-run complete; pass --train to optimize the router.")
        return

    device = torch.device(args.device)
    model.to(device)

    print("building mixed train/valid datasets...")
    train_dataset = _build_mixed_dataset(config, split="train")
    valid_dataset = _build_mixed_dataset(config, split="valid")
    collator = DataCollatorForCompletionOnlyLM(pad_token_id=int(tokenizer.pad_token_id))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config["training"].get("validation_batch_size", 4)),
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = create_optimizer(model, config)
    epochs = int(config["training"]["epochs"])
    grad_accum = int(config["training"].get("gradient_accumulation_steps", 1))
    total_steps = max(1, (len(train_loader) * epochs) // grad_accum)
    scheduler = create_linear_scheduler(
        optimizer,
        num_training_steps=total_steps,
        warmup_steps=int(config["training"].get("warmup_steps", 0)),
    )

    metrics_path = output_dir / "metrics.jsonl"
    metrics_handle = metrics_path.open("a", encoding="utf-8")

    for epoch in range(1, epochs + 1):
        print(f"epoch {epoch}/{epochs} - training router...")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, config
        )
        val_loss = evaluate_loss(
            model, valid_loader, device,
            label_smoothing=float(config["training"].get("label_smoothing", 0.0)),
        )
        record = {
            "type": "validation",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        print(record)
        metrics_handle.write(json.dumps(record) + "\n")
        metrics_handle.flush()

        ckpt_path = output_dir / "checkpoints" / f"router_epoch_{epoch}.pt"
        _save_router(ckpt_path, model, config)
        print(f"saved {ckpt_path}")

    final_path = output_dir / "checkpoints" / "router_final.pt"
    _save_router(final_path, model, config)
    metrics_handle.close()
    print(f"done. final router: {final_path}")
    print(f"router params: {count_parameters(model, trainable_only=True):,}")


if __name__ == "__main__":
    main()
