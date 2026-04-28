#!/usr/bin/env python
"""Compute metrics for one (task, predictions) cell of the matrix.

Two metric stacks are emitted side by side:

1. *Uniform stack* (always run): sacrebleu corpus BLEU + NLTK
   meteor_score + pyter3 TER. Identical math across all 15 cells —
   apples-to-apples within each column for method comparison.

2. *Official stack* (when --official is passed): runs
   `tuetschek/e2e-metrics measure_scores.py` on the same
   predictions. Emits BLEU + NIST + METEOR + ROUGE_L + CIDEr.
   This is the scorer the LoRA paper used for E2E (Hu et al. 2021
   Table 6) and the same one Justin's `paper_aligned_official_beam`
   runs use. Yale-LILY/dart bundles a copy at
   `evaluation/e2e-metrics/measure_scores.py`, so the same binary
   produces "paper-style NLG eval" numbers for DART/WebNLG too.

Both reference sets are reconstructed from the raw split in
first-occurrence dedup order — exactly the order
`eval_matrix_generate.py` writes predictions in — so predictions
and references align row-for-row without an explicit
references-file artifact on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.config import load_config
from lora_gpt2.data import load_e2e_records, raw_split_path
from lora_gpt2.evaluate import (
    corpus_multi_reference_bleu,
    read_prediction_records,
    run_official_e2e_evaluator,
    write_official_e2e_files,
)
from lora_gpt2.utils import ensure_dir


def _grouped_references(task_config: dict, split: str) -> tuple[list[str], list[list[str]]]:
    """Return (unique_contexts, references_grouped) in first-occurrence order.

    `references_grouped[i]` is the list of all reference completions seen
    in the raw data for `unique_contexts[i]`. Order matches what
    `eval_matrix_generate.py::_unique_contexts_in_order` produces, so
    predictions and references align by index without an explicit
    join-on-context step.
    """
    raw_records = load_e2e_records(raw_split_path(task_config, split))
    seen: dict[str, int] = {}
    contexts: list[str] = []
    references: list[list[str]] = []
    for record in raw_records:
        if record.context not in seen:
            seen[record.context] = len(contexts)
            contexts.append(record.context)
            references.append([])
        references[seen[record.context]].append(record.completion)
    return contexts, references


def corpus_meteor(predictions: list[str], references_grouped: list[list[str]]) -> float:
    """Mean per-sentence METEOR. Multi-ref: NLTK takes the max over refs.

    Returned as a 0-100 percentage to match the BLEU scale used elsewhere.
    """
    import nltk
    from nltk.translate.meteor_score import meteor_score

    # Lazy resource downloads — each call is a no-op if already cached.
    for resource in ("wordnet", "punkt", "punkt_tab", "omw-1.4"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    scores: list[float] = []
    for prediction, references in zip(predictions, references_grouped):
        if not prediction.strip() or not references:
            scores.append(0.0)
            continue
        ref_tokens = [r.split() for r in references]
        pred_tokens = prediction.split()
        scores.append(float(meteor_score(ref_tokens, pred_tokens)))
    if not scores:
        return 0.0
    return 100.0 * sum(scores) / len(scores)


def corpus_ter(predictions: list[str], references_grouped: list[list[str]]) -> float:
    """Mean per-sentence TER. Multi-ref: take the min (best) TER over refs.

    Reported in the standard TER convention as a 0-1 ratio (lower is better).
    Sentences with empty predictions get TER=1.0 — they edit-distance to the
    full reference, which is the standard sane default.
    """
    import pyter

    scores: list[float] = []
    for prediction, references in zip(predictions, references_grouped):
        if not prediction.strip():
            scores.append(1.0)
            continue
        if not references:
            scores.append(0.0)
            continue
        per_ref = []
        for reference in references:
            ref_tokens = reference.split()
            if not ref_tokens:
                continue
            per_ref.append(float(pyter.ter(prediction.split(), ref_tokens)))
        if not per_ref:
            scores.append(1.0)
        else:
            scores.append(min(per_ref))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _parse_e2e_metrics_stdout(stdout: str) -> dict[str, float]:
    """Pick BLEU/NIST/METEOR/ROUGE_L/CIDEr out of measure_scores.py stdout.

    The script prints a `SCORES:` block followed by `KEY: value` lines;
    we just regex-grep each metric and ignore everything else. Returns
    BLEU as a 0-100 percentage to match the rest of the codebase
    (sacrebleu/uniform stack also outputs 0-100); the script itself
    prints BLEU as a 0-1 fraction, so we multiply.
    """
    keys = ("BLEU", "NIST", "METEOR", "ROUGE_L", "CIDEr")
    scale_to_percent = {"BLEU", "METEOR", "ROUGE_L"}
    parsed: dict[str, float] = {}
    for line in stdout.splitlines():
        line = line.strip()
        for key in keys:
            prefix = f"{key}:"
            if line.startswith(prefix):
                try:
                    value = float(line.split(":", 1)[1].strip())
                except ValueError:
                    continue
                if key in scale_to_percent:
                    value *= 100.0
                parsed[key.lower()] = value
                break
    return parsed


def run_official_metrics(
    predictions: list[str],
    references_grouped: list[list[str]],
    e2e_metrics_dir: Path,
    work_dir: Path,
) -> dict[str, float]:
    """Score with tuetschek/e2e-metrics; return its parsed metrics dict.

    `e2e_metrics_dir` is the local clone of the evaluator (the notebook
    clones it once into `external/e2e-metrics`). `work_dir` is where we
    write the blank-line-separated refs file and the per-line preds
    file the script consumes — kept alongside the matrix's per-cell
    output so the inputs are inspectable post-hoc.
    """
    refs_file = work_dir / "official_refs.txt"
    preds_file = work_dir / "official_preds.txt"
    write_official_e2e_files(refs_file, preds_file, predictions, references_grouped)
    completed = run_official_e2e_evaluator(e2e_metrics_dir, refs_file, preds_file)
    metrics = _parse_e2e_metrics_stdout(completed.stdout)
    if not metrics:
        # Surface the raw output so a user can debug a parse failure.
        raise RuntimeError(
            f"Failed to parse any metrics from measure_scores.py stdout:\n{completed.stdout}"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", required=True,
                        help="Per-task config (used only for raw data path + split lookup).")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--predictions-file", required=True,
                        help="JSONL produced by eval_matrix_generate.py "
                             "(one row per unique MR, in canonical order).")
    parser.add_argument("--output-file", required=True,
                        help="Where to write the metric JSON summary.")
    parser.add_argument("--variant", default=None,
                        help="Optional label echoed into the output JSON.")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="If set, truncate the reference set to the first N unique "
                             "MRs to align with a generator run that used --max-examples.")
    parser.add_argument("--official", action="store_true",
                        help="Also run tuetschek/e2e-metrics on the predictions for "
                             "paper-comparable BLEU/NIST/METEOR/ROUGE_L/CIDEr.")
    parser.add_argument("--official-script-dir", default="external/e2e-metrics",
                        help="Path to the cloned tuetschek/e2e-metrics repo.")
    args = parser.parse_args()

    task_config = load_config(args.task_config)
    contexts, references_grouped = _grouped_references(task_config, args.split)

    if args.max_examples is not None:
        # Generator emits predictions in the same first-occurrence dedup
        # order, truncated to the same count, so slicing the reference
        # set to len = max_examples keeps row-for-row alignment.
        contexts = contexts[: args.max_examples]
        references_grouped = references_grouped[: args.max_examples]

    prediction_records = read_prediction_records(args.predictions_file)
    if len(prediction_records) != len(contexts):
        raise SystemExit(
            f"Prediction/reference count mismatch: "
            f"{len(prediction_records)} predictions vs {len(contexts)} unique contexts. "
            f"Likely cause: predictions file was generated with --max-examples N but the "
            f"evaluator wasn't told that — pass --max-examples N here too."
        )

    predictions = [record["prediction"] for record in prediction_records]

    print(f"computing BLEU ({len(predictions)} hyps × multi-ref)...")
    bleu = corpus_multi_reference_bleu(predictions, references_grouped)
    print(f"computing METEOR...")
    meteor = corpus_meteor(predictions, references_grouped)
    print(f"computing TER...")
    ter = corpus_ter(predictions, references_grouped)

    summary = {
        "task_config": str(args.task_config),
        "split": args.split,
        "predictions_file": str(args.predictions_file),
        "variant": args.variant,
        "num_unique_contexts": len(contexts),
        "metrics": {
            "bleu": bleu,
            "meteor": meteor,
            "ter": ter,
        },
    }

    if args.official:
        # The e2e-metrics scorer needs to be present on disk; we don't
        # auto-clone it from this script (the notebook handles install
        # in one place so all 15 cells share it).
        e2e_metrics_dir = Path(args.official_script_dir)
        if not (e2e_metrics_dir / "measure_scores.py").exists():
            raise SystemExit(
                f"--official requested but {e2e_metrics_dir}/measure_scores.py not found. "
                f"Install with `git clone https://github.com/tuetschek/e2e-metrics "
                f"{e2e_metrics_dir}`."
            )
        print("computing official e2e-metrics (BLEU/NIST/METEOR/ROUGE_L/CIDEr)...")
        official = run_official_metrics(
            predictions,
            references_grouped,
            e2e_metrics_dir=e2e_metrics_dir,
            work_dir=Path(args.predictions_file).parent,
        )
        summary["official_metrics"] = official

    output_path = Path(args.output_file)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("metrics", "official_metrics") if k in summary},
                     indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
