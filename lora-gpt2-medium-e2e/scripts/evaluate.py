#!/usr/bin/env python
"""Evaluate generated E2E predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.config import load_config, resolve_path
from lora_gpt2.evaluate import (
    corpus_bleu,
    read_prediction_texts,
    run_official_e2e_evaluator,
    write_metric_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--official", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    predictions_file = resolve_path(config, config["evaluation"]["predictions_file"])
    references_file = resolve_path(config, config["evaluation"]["references_file"])
    predictions = read_prediction_texts(predictions_file, field="prediction")
    references = read_prediction_texts(references_file, field="completion")
    metrics = {"bleu": corpus_bleu(predictions, references)}

    if args.official:
        completed = run_official_e2e_evaluator(
            resolve_path(config, config["evaluation"]["official_e2e_script_dir"]),
            references_file,
            predictions_file,
        )
        metrics["official_stdout"] = completed.stdout

    output_path = predictions_file.with_suffix(".metrics.json")
    write_metric_summary(output_path, metrics)
    print(f"wrote metrics to {output_path}")


if __name__ == "__main__":
    main()
