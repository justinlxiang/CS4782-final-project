#!/usr/bin/env python
"""Compute BLEU + METEOR + TER for one (task, predictions) cell of the matrix.

Uses one consistent metric toolchain across all three tasks:

  BLEU    sacrebleu corpus BLEU (multi-reference, force=True)
  METEOR  NLTK meteor_score (handles multi-reference natively, takes max)
  TER     pyter3 (sentence-level; multi-reference => take min over refs)

We deliberately do NOT use each task's bespoke official scorer (multi-bleu.perl
for DART, GenerationEval for WebNLG, e2e-metrics for E2E). The user goal is
*method comparison within a benchmark*, not paper-replication; identical
scorers across all 15 cells gives apples-to-apples within each column. The
trade-off is that absolute numbers will differ slightly from paper-published
values (BLEU implementations disagree by ~0.5 absolute), but the relative
ranking of the 5 model variants on each task is what we care about.

References are reconstructed from the task config's raw split, grouped by
context in first-occurrence order — the same canonical order
`eval_matrix_generate.py` uses, so predictions and references align row-for-
row without a separate references-file artifact on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.config import load_config
from lora_gpt2.data import load_e2e_records, raw_split_path
from lora_gpt2.evaluate import corpus_multi_reference_bleu, read_prediction_records
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
    args = parser.parse_args()

    task_config = load_config(args.task_config)
    contexts, references_grouped = _grouped_references(task_config, args.split)

    prediction_records = read_prediction_records(args.predictions_file)
    if len(prediction_records) != len(contexts):
        raise SystemExit(
            f"Prediction/reference count mismatch: "
            f"{len(prediction_records)} predictions vs {len(contexts)} unique contexts. "
            f"Likely cause: predictions file was generated with a different split or "
            f"with --max-examples. Re-run eval_matrix_generate.py to align."
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

    output_path = Path(args.output_file)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
