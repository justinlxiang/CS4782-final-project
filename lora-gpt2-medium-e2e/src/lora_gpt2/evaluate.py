"""Metric helpers for E2E outputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from lora_gpt2.data import E2ERecord


def corpus_bleu(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute a sacreBLEU corpus score for quick sanity checks."""
    import sacrebleu

    score = sacrebleu.corpus_bleu(list(predictions), [list(references)], force=True)
    return float(score.score)


def corpus_multi_reference_bleu(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> float:
    """Compute corpus BLEU with variable-count references per prediction."""
    import sacrebleu

    if not predictions:
        return 0.0
    max_refs = max(len(item) for item in references)
    reference_streams = []
    for ref_index in range(max_refs):
        stream = []
        for ref_group in references:
            stream.append(ref_group[ref_index] if ref_index < len(ref_group) else ref_group[0])
        reference_streams.append(stream)
    score = sacrebleu.corpus_bleu(list(predictions), reference_streams, force=True)
    return float(score.score)


def corpus_rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute average ROUGE-L F1 as a percentage."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(reference, prediction)["rougeL"].fmeasure
        for prediction, reference in zip(predictions, references)
    ]
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores) * 100.0)


def corpus_multi_reference_rouge_l(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> float:
    """Compute average best-reference ROUGE-L F1 as a percentage."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for prediction, ref_group in zip(predictions, references):
        scores.append(
            max(
                scorer.score(reference, prediction)["rougeL"].fmeasure
                for reference in ref_group
            )
        )
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores) * 100.0)


def read_lines(path: str | Path) -> list[str]:
    """Read a text file as stripped lines."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def read_prediction_texts(path: str | Path, field: str = "prediction") -> list[str]:
    """Read plain-text lines or JSONL prediction records."""
    lines = read_lines(path)
    if not lines:
        return []
    if lines[0].lstrip().startswith("{"):
        return [str(json.loads(line).get(field, "")) for line in lines if line.strip()]
    return lines


def read_prediction_records(path: str | Path) -> list[dict[str, str]]:
    """Read JSONL predictions, preserving ids and prompts when present."""
    lines = read_lines(path)
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if line.lstrip().startswith("{"):
            item = json.loads(line)
            records.append(
                {
                    "id": str(item.get("id", index)),
                    "prompt": str(item.get("prompt", "")),
                    "prediction": str(item.get("prediction", "")),
                }
            )
        else:
            records.append({"id": str(index), "prompt": "", "prediction": line})
    return records


def group_e2e_predictions_and_references(
    records: Sequence[E2ERecord],
    predictions: Sequence[dict[str, str]],
) -> tuple[list[str], list[list[str]], list[str]]:
    """Group E2E references by unique MR, matching the official decode step."""
    if len(records) != len(predictions):
        raise ValueError(
            f"Expected one prediction per raw E2E row, got {len(predictions)} "
            f"predictions for {len(records)} records."
        )

    order: list[str] = []
    references_by_context: dict[str, list[str]] = {}
    predictions_by_context: dict[str, str] = {}
    for record, prediction_record in zip(records, predictions):
        context = record.context
        if context not in references_by_context:
            order.append(context)
            references_by_context[context] = []
            predictions_by_context[context] = prediction_record["prediction"]
        references_by_context[context].append(record.completion)

    grouped_predictions = [predictions_by_context[context] for context in order]
    grouped_references = [references_by_context[context] for context in order]
    return grouped_predictions, grouped_references, order


def write_text_lines(path: str | Path, lines: Sequence[str]) -> None:
    """Write one text record per line."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(str(line).replace("\n", " ").strip() + "\n")


def write_official_e2e_files(
    references_path: str | Path,
    predictions_path: str | Path,
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> None:
    """Write official E2E evaluator input files."""
    ref_path = Path(references_path)
    pred_path = Path(predictions_path)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with ref_path.open("w", encoding="utf-8") as ref_handle, pred_path.open(
        "w",
        encoding="utf-8",
    ) as pred_handle:
        for prediction, ref_group in zip(predictions, references):
            for reference in ref_group:
                ref_handle.write(str(reference).replace("\n", " ").strip() + "\n")
            ref_handle.write("\n")
            pred_handle.write(str(prediction).replace("\n", " ").strip() + "\n")


def write_metric_summary(path: str | Path, metrics: dict[str, Any]) -> None:
    """Write metrics to JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def run_official_e2e_evaluator(
    script_dir: str | Path,
    references_file: str | Path,
    predictions_file: str | Path,
    script_name: str = "measure_scores.py",
) -> subprocess.CompletedProcess[str]:
    """Run the external E2E-style evaluator if it is installed locally.

    The DART evaluator ships the same `measure_scores.py` interface (refs file
    with blank-line-separated groups, predictions file with one line per group)
    so the same runner works for either dataset by pointing `script_dir` at the
    matching vendored evaluator. Override `script_name` only if a fork renames
    the entry point.
    """
    script = Path(script_dir) / script_name
    if not script.exists():
        raise FileNotFoundError(f"Official evaluator script not found at {script}.")
    return subprocess.run(
        [sys.executable, str(script), str(references_file), str(predictions_file), "-p"],
        check=True,
        capture_output=True,
        text=True,
    )


def run_official_dart_evaluator(
    script_dir: str | Path,
    references_file: str | Path,
    predictions_file: str | Path,
) -> subprocess.CompletedProcess[str]:
    """Run the external DART evaluator (BLEU / METEOR / TER).

    Thin wrapper around `run_official_e2e_evaluator` that documents intent at
    the call site; the underlying invocation is identical because the DART
    evaluator reuses the E2E `measure_scores.py` interface.
    """
    return run_official_e2e_evaluator(script_dir, references_file, predictions_file)
