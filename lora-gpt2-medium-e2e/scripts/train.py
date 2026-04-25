#!/usr/bin/env python
"""Training entrypoint. Use `--dry-run` before approved training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.checkpointing import load_training_checkpoint, save_training_checkpoint
from lora_gpt2.config import load_config, resolve_path
from lora_gpt2.data import DataCollatorForCompletionOnlyLM, ProcessedE2EDataset
from lora_gpt2.modeling import load_lora_model
from lora_gpt2.train import create_linear_scheduler, create_optimizer, forward_loss
from lora_gpt2.utils import ensure_dir, get_device, parameter_report, seed_everything


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def build_dataloader(
    config: dict,
    tokenizer,
    split: str,
    batch_size: int,
    max_examples: int | None,
    shuffle: bool,
    seed: int | None = None,
) -> tuple[DataLoader, Path, int]:
    split_path = resolve_path(
        config,
        Path(config["data"]["processed_dir"]) / f"{split}.jsonl",
    )
    dataset = ProcessedE2EDataset(split_path, max_examples=max_examples)
    collator = DataCollatorForCompletionOnlyLM(pad_token_id=tokenizer.pad_token_id)
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=generator,
    )
    return dataloader, split_path, len(dataset)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors after loading a checkpoint."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Build model/optimizer and exit.")
    parser.add_argument(
        "--smoke-train",
        action="store_true",
        help="Run a tiny approved training smoke test with optimizer steps.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run full training with adapter checkpointing. Use only when ready.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--dry-run-split", default="train")
    parser.add_argument("--dry-run-max-examples", type=int, default=2)
    parser.add_argument("--dry-run-batch-size", type=int, default=1)
    parser.add_argument("--smoke-max-steps", type=int, default=2)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Optional hard cap for training verification runs.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume --train from a full training checkpoint saved by this script.",
    )
    parser.add_argument(
        "--dry-run-forward-pass",
        action="store_true",
        help="Run one no-grad forward loss on a tiny batch. This does not train.",
    )
    args = parser.parse_args()

    selected_modes = sum(bool(value) for value in (args.dry_run, args.smoke_train, args.train))
    if selected_modes != 1:
        raise SystemExit(
            "Choose exactly one mode: --dry-run, --smoke-train, or --train."
        )
    if args.smoke_train and not (1 <= args.smoke_max_steps <= 10):
        raise SystemExit("--smoke-max-steps must be between 1 and 10.")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise SystemExit("--max-train-steps must be positive when provided.")
    if args.resume_checkpoint and not args.train:
        raise SystemExit("--resume-checkpoint is only supported with --train.")

    config = load_config(args.config)
    seed_everything(int(config["project"]["seed"]))
    device = get_device(args.device)
    model, tokenizer, report = load_lora_model(config)
    train_batch_size = int(config["training"]["batch_size"])
    train_split_path = resolve_path(config, Path(config["data"]["processed_dir"]) / "train.jsonl")
    train_dataset_size = len(ProcessedE2EDataset(train_split_path))
    steps_per_epoch = (train_dataset_size + train_batch_size - 1) // train_batch_size
    configured_steps = steps_per_epoch * int(config["training"]["epochs"])
    total_training_steps = (
        min(configured_steps, args.max_train_steps)
        if args.max_train_steps is not None
        else configured_steps
    )
    optimizer = create_optimizer(model, config)
    scheduler = create_linear_scheduler(
        optimizer,
        num_training_steps=max(1, args.smoke_max_steps if args.smoke_train else total_training_steps),
        warmup_steps=0 if args.smoke_train or args.dry_run else int(config["training"]["warmup_steps"]),
    )
    params = parameter_report(model)

    max_examples = max(args.dry_run_max_examples, args.smoke_max_steps) if args.smoke_train else args.dry_run_max_examples
    dataloader, split_path, dataset_size = build_dataloader(
        config=config,
        tokenizer=tokenizer,
        split=args.dry_run_split,
        batch_size=args.dry_run_batch_size,
        max_examples=max_examples,
        shuffle=False,
    )
    first_batch = next(iter(dataloader))

    print(f"dry_run={args.dry_run}")
    print(f"smoke_train={args.smoke_train}")
    print(f"train={args.train}")
    print(f"device={device}")
    print(f"replaced_modules={report.replaced_modules}")
    print(f"trainable_parameters={params['trainable']}")
    print(f"optimizer_param_groups={len(optimizer.param_groups)}")
    print(f"scheduler={scheduler.__class__.__name__}")
    print(f"dry_run_dataset={split_path}")
    print(f"dry_run_examples={dataset_size}")
    print(f"dry_run_batch_input_shape={tuple(first_batch['input_ids'].shape)}")
    print(f"dry_run_batch_label_shape={tuple(first_batch['labels'].shape)}")

    if args.dry_run_forward_pass:
        model.to(device)
        first_batch = {key: value.to(device) for key, value in first_batch.items()}
        with torch.no_grad():
            loss = forward_loss(
                model,
                first_batch,
                label_smoothing=float(config["training"].get("label_smoothing", 0.0)),
            )
        print(f"dry_run_forward_loss={float(loss.detach().cpu()):.4f}")
        print("dry_run_forward_pass=true")

    if args.smoke_train:
        model.to(device)
        model.train()
        label_smoothing = float(config["training"].get("label_smoothing", 0.0))
        max_grad_norm = float(config["training"].get("max_grad_norm", 0.0))
        losses: list[float] = []
        for step, batch in enumerate(dataloader, start=1):
            if step > args.smoke_max_steps:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = forward_loss(model, batch, label_smoothing=label_smoothing)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    max_grad_norm,
                )
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            print(f"smoke_step={step} loss={loss_value:.4f}")
        print(f"smoke_train_steps={len(losses)}")
        print("smoke_train_complete=true")

    if args.train:
        output_dir = ensure_dir(resolve_path(config, config["project"]["output_dir"]))
        checkpoints_dir = ensure_dir(output_dir / "checkpoints")
        metrics_path = output_dir / "metrics.jsonl"
        if metrics_path.exists() and not args.resume_checkpoint:
            metrics_path.unlink()
        write_json(output_dir / "config.json", config)
        write_json(output_dir / "parameter_report.json", params)
        resume_state: dict = {}
        if args.resume_checkpoint:
            resume_path = resolve_path(config, args.resume_checkpoint)
            resume_state = load_training_checkpoint(
                resume_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                strict=False,
            )
            print(f"resumed_checkpoint={resume_path}")

        print(f"train_dataset={train_split_path}")
        print(f"train_examples={train_dataset_size}")
        print(f"train_batch_size={train_batch_size}")
        print(f"configured_training_steps={configured_steps}")
        print(f"effective_training_steps={total_training_steps}")
        print(f"output_dir={output_dir}")

        model.to(device)
        if args.resume_checkpoint:
            move_optimizer_state(optimizer, device)
        model.train()
        label_smoothing = float(config["training"].get("label_smoothing", 0.0))
        max_grad_norm = float(config["training"].get("max_grad_norm", 0.0))
        save_every = int(config["training"].get("save_every_steps", 1000))
        global_step = int(resume_state.get("global_step", 0))
        start_epoch = int(resume_state.get("epoch", 1))
        skip_batches = int(resume_state.get("batch_in_epoch", 0))
        current_epoch = start_epoch
        current_batch_in_epoch = skip_batches
        remaining_steps = max(total_training_steps - global_step, 0)

        with tqdm(
            total=remaining_steps,
            initial=0,
            desc="training",
            unit="step",
            dynamic_ncols=True,
        ) as progress:
            for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
                current_epoch = epoch
                train_loader, full_train_path, full_train_size = build_dataloader(
                    config=config,
                    tokenizer=tokenizer,
                    split="train",
                    batch_size=train_batch_size,
                    max_examples=None,
                    shuffle=True,
                    seed=int(config["project"]["seed"]) + epoch,
                )
                for batch_in_epoch, batch in enumerate(train_loader, start=1):
                    if epoch == start_epoch and batch_in_epoch <= skip_batches:
                        continue
                    current_batch_in_epoch = batch_in_epoch
                    global_step += 1
                    batch = {key: value.to(device) for key, value in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    loss = forward_loss(model, batch, label_smoothing=label_smoothing)
                    loss.backward()
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [param for param in model.parameters() if param.requires_grad],
                            max_grad_norm,
                        )
                    optimizer.step()
                    scheduler.step()

                    loss_value = float(loss.detach().cpu())
                    record = {
                        "epoch": epoch,
                        "batch_in_epoch": batch_in_epoch,
                        "step": global_step,
                        "loss": loss_value,
                    }
                    append_jsonl(metrics_path, record)
                    progress.set_postfix(
                        epoch=epoch,
                        loss=f"{loss_value:.4f}",
                        step=global_step,
                    )
                    progress.update(1)

                    if save_every > 0 and global_step % save_every == 0:
                        checkpoint_path = checkpoints_dir / f"adapter_step_{global_step}.pt"
                        save_training_checkpoint(
                            checkpoint_path,
                            model,
                            optimizer,
                            scheduler,
                            config=config,
                            training_state={
                                "epoch": epoch,
                                "batch_in_epoch": batch_in_epoch,
                                "global_step": global_step,
                                "loss": loss_value,
                            },
                        )
                        tqdm.write(f"saved_checkpoint={checkpoint_path}")

                    if args.max_train_steps is not None and global_step >= args.max_train_steps:
                        break
                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    break

        final_checkpoint = checkpoints_dir / "adapter_final.pt"
        save_training_checkpoint(
            final_checkpoint,
            model,
            optimizer,
            scheduler,
            config=config,
            training_state={
                "step": global_step,
                "global_step": global_step,
                "type": "final",
                "epoch": current_epoch,
                "batch_in_epoch": current_batch_in_epoch,
            },
        )
        print(f"saved_final_checkpoint={final_checkpoint}")
        print(f"train_steps={global_step}")
        print("train_complete=true")


if __name__ == "__main__":
    main()
