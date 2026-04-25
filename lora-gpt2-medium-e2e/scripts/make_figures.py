#!/usr/bin/env python
"""Create report figures from a completed LoRA E2E run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.config import load_config, resolve_path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def rolling_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    averaged = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        averaged.append(running / min(index + 1, window))
    return averaged


def save_training_loss_figure(records: list[dict[str, Any]], output_path: Path) -> dict[str, float]:
    import matplotlib.pyplot as plt

    steps = [int(record["step"]) for record in records]
    losses = [float(record["loss"]) for record in records]
    smoothed = rolling_average(losses, window=200)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, losses, color="#8fb3ff", alpha=0.22, linewidth=0.8, label="step loss")
    ax.plot(steps, smoothed, color="#1f5fbf", linewidth=2.0, label="200-step average")
    ax.set_title("GPT-2 Medium LoRA Training Loss on E2E")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Completion-only loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "num_steps": float(len(records)),
    }


def save_epoch_loss_figure(records: list[dict[str, Any]], output_path: Path) -> dict[str, float]:
    import matplotlib.pyplot as plt

    by_epoch: dict[int, list[float]] = {}
    for record in records:
        by_epoch.setdefault(int(record["epoch"]), []).append(float(record["loss"]))
    epochs = sorted(by_epoch)
    epoch_losses = [sum(by_epoch[epoch]) / len(by_epoch[epoch]) for epoch in epochs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, epoch_losses, marker="o", color="#2d7f5e", linewidth=2.0)
    ax.set_title("Average Training Loss by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean loss")
    ax.set_xticks(epochs)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {f"epoch_{epoch}_mean_loss": loss for epoch, loss in zip(epochs, epoch_losses)}


def save_parameter_figure(parameter_report: dict[str, Any], output_path: Path) -> dict[str, float]:
    import matplotlib.pyplot as plt

    full_finetune_params = float(parameter_report["total"])
    lora_params = float(parameter_report["trainable"])
    labels = ["Full fine-tuning", "LoRA adapters"]
    values = [full_finetune_params, lora_params]
    colors = ["#9b9b9b", "#c4554d"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_yscale("log")
    ax.set_title("Trainable Parameter Count")
    ax.set_ylabel("Parameters (log scale)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value / 1_000_000:.2f}M",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "full_finetune_trainable_params": full_finetune_params,
        "lora_trainable_params": lora_params,
        "trainable_fraction": float(parameter_report["trainable_fraction"]),
    }


def save_metric_figure(metrics: dict[str, Any], output_path: Path) -> dict[str, float]:
    import matplotlib.pyplot as plt

    numeric_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if key != "num_examples" and isinstance(value, (int, float))
    }
    if not numeric_metrics:
        return {}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(numeric_metrics)
    values = [numeric_metrics[name] for name in names]
    ax.bar(names, values, color="#6e65b7")
    ax.set_title("Generated E2E Metrics")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return numeric_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--metrics-file", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = resolve_path(config, args.run_dir or config["project"]["output_dir"])
    figures_dir = resolve_path(config, args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = resolve_path(config, config["evaluation"]["predictions_file"])
    metrics_file = (
        resolve_path(config, args.metrics_file)
        if args.metrics_file
        else predictions_file.with_suffix(".metrics.json")
    )

    summary: dict[str, Any] = {"figures_dir": str(figures_dir), "run_dir": str(run_dir)}

    training_metrics = run_dir / "metrics.jsonl"
    if training_metrics.exists():
        records = read_jsonl(training_metrics)
        if records:
            summary["training_loss"] = save_training_loss_figure(
                records,
                figures_dir / "training_loss.png",
            )
            summary["epoch_loss"] = save_epoch_loss_figure(
                records,
                figures_dir / "epoch_loss.png",
            )

    parameter_report = run_dir / "parameter_report.json"
    if parameter_report.exists():
        summary["parameters"] = save_parameter_figure(
            read_json(parameter_report),
            figures_dir / "trainable_parameters.png",
        )

    if metrics_file.exists():
        summary["metrics"] = save_metric_figure(
            read_json(metrics_file),
            figures_dir / "evaluation_metrics.png",
        )

    summary_path = figures_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"wrote figures to {figures_dir}")
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
