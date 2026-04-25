#!/usr/bin/env python
"""Evaluate generated E2E predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.config import load_config, resolve_path
from lora_gpt2.data import load_e2e_records, raw_split_path
from lora_gpt2.evaluate import (
    corpus_bleu,
    corpus_multi_reference_bleu,
    corpus_multi_reference_rouge_l,
    corpus_rouge_l,
    group_e2e_predictions_and_references,
    read_prediction_records,
    read_prediction_texts,
    run_official_e2e_evaluator,
    write_official_e2e_files,
    write_text_lines,
    write_metric_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--split", default=None)
    parser.add_argument("--predictions-file", default=None)
    parser.add_argument("--references-file", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    split = args.split or config["generation"].get("split", "test")
    predictions_file = resolve_path(
        config,
        args.predictions_file or config["evaluation"]["predictions_file"],
    )
    references_file = resolve_path(
        config,
        args.references_file or config["evaluation"]["references_file"],
    )
    if not references_file.exists():
        references = [record.completion for record in load_e2e_records(raw_split_path(config, split))]
        write_text_lines(references_file, references)
        print(f"wrote {len(references)} references to {references_file}")

    raw_records = load_e2e_records(raw_split_path(config, split))
    prediction_records = read_prediction_records(predictions_file)
    predictions = read_prediction_texts(predictions_file, field="prediction")
    references = read_prediction_texts(references_file, field="completion")
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise SystemExit("--max-examples must be positive when provided.")
        predictions = predictions[: args.max_examples]
        references = references[: args.max_examples]
        raw_records = raw_records[: args.max_examples]
        prediction_records = prediction_records[: args.max_examples]
    if len(predictions) != len(references):
        raise SystemExit(
            f"Prediction/reference count mismatch: {len(predictions)} predictions, "
            f"{len(references)} references."
        )

    grouped_predictions, grouped_references, grouped_contexts = (
        group_e2e_predictions_and_references(raw_records, prediction_records)
    )
    official_references_file = predictions_file.with_suffix(".e2e_refs.txt")
    official_predictions_file = predictions_file.with_suffix(".e2e_preds.txt")
    write_official_e2e_files(
        official_references_file,
        official_predictions_file,
        grouped_predictions,
        grouped_references,
    )

    metrics = {
        "num_examples": len(predictions),
        "num_unique_mrs": len(grouped_contexts),
        "bleu": corpus_multi_reference_bleu(grouped_predictions, grouped_references),
        "rouge_l": corpus_multi_reference_rouge_l(grouped_predictions, grouped_references),
        "line_bleu": corpus_bleu(predictions, references),
        "line_rouge_l": corpus_rouge_l(predictions, references),
        "official_references_file": str(official_references_file),
        "official_predictions_file": str(official_predictions_file),
    }

    if args.official:
        completed = run_official_e2e_evaluator(
            resolve_path(config, config["evaluation"]["official_e2e_script_dir"]),
            official_references_file,
            official_predictions_file,
        )
        metrics["official_stdout"] = completed.stdout

    output_path = predictions_file.with_suffix(".metrics.json")
    write_metric_summary(output_path, metrics)
    print(f"wrote metrics to {output_path}")


if __name__ == "__main__":
    main()
