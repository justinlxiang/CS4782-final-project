"""Metric helpers for E2E outputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def corpus_bleu(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute a sacreBLEU corpus score for quick sanity checks."""
    import sacrebleu

    score = sacrebleu.corpus_bleu(list(predictions), [list(references)])
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


def write_text_lines(path: str | Path, lines: Sequence[str]) -> None:
    """Write one text record per line."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(str(line).replace("\n", " ").strip() + "\n")


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
) -> subprocess.CompletedProcess[str]:
    """Run the external E2E evaluator if it is installed locally."""
    script = Path(script_dir) / "measure_scores.py"
    if not script.exists():
        raise FileNotFoundError(f"Official E2E evaluator not found at {script}.")
    return subprocess.run(
        [sys.executable, str(script), str(references_file), str(predictions_file), "-p"],
        check=True,
        capture_output=True,
        text=True,
    )
