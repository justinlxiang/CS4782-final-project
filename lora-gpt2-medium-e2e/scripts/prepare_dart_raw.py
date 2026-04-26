#!/usr/bin/env python
"""Convert raw DART JSON splits into the JSONL format consumed by prepare_e2e.py.

DART (https://github.com/Yale-LILY/dart) ships per-split JSON files where each
example carries a list of (subject, predicate, object) triples plus a list of
human-written annotations. We linearize the triples into a single GPT-2 prompt
context and emit one (context, completion) JSONL row per (example, annotation)
pair, matching the multi-reference shape that the existing E2E pipeline already
groups during evaluation.

We also write a multi-reference text file for the test split, formatted as
blank-line-separated reference groups, which is what the official E2E/DART
Perl evaluators expect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.config import load_config, resolve_path
from lora_gpt2.utils import ensure_dir, write_jsonl


# The triple linearization mirrors the conventions used by the official LoRA
# NLG preprocessing for DART: each triple is rendered with explicit head /
# relation / tail markers so GPT-2 can learn segment boundaries from BPE alone.
HEAD_TOKEN = "<H>"
RELATION_TOKEN = "<R>"
TAIL_TOKEN = "<T>"


def linearize_tripleset(tripleset: list[list[str]]) -> str:
    parts: list[str] = []
    for triple in tripleset:
        if len(triple) != 3:
            raise ValueError(f"Expected (s, p, o) triple, got {triple!r}.")
        subject, relation, obj = (str(x).strip() for x in triple)
        parts.append(f"{HEAD_TOKEN} {subject} {RELATION_TOKEN} {relation} {TAIL_TOKEN} {obj}")
    return " ".join(parts)


def annotation_texts(example: dict) -> list[str]:
    annotations = example.get("annotations") or []
    texts: list[str] = []
    for entry in annotations:
        text = entry.get("text") if isinstance(entry, dict) else None
        if text:
            texts.append(str(text).strip())
    return texts


def convert_split(json_path: Path) -> list[dict[str, str]]:
    """Expand each DART example into (context, completion) JSONL rows."""
    with json_path.open("r", encoding="utf-8") as handle:
        examples = json.load(handle)

    rows: list[dict[str, str]] = []
    for example in examples:
        tripleset = example.get("tripleset") or []
        if not tripleset:
            continue
        context = linearize_tripleset(tripleset)
        refs = annotation_texts(example)
        if not refs:
            continue
        for ref in refs:
            rows.append({"context": context, "completion": ref})
    return rows


def write_multi_reference_file(path: Path, rows: list[dict[str, str]]) -> None:
    """Write blank-line-separated reference groups (one block per unique context).

    The official E2E and DART Perl evaluators both consume this layout: every
    line is a reference, an empty line ends a group, and prediction line N
    corresponds to group N. Grouping here matches what the generation pipeline
    does at decode time (one prediction per unique meaning representation).
    """
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for row in rows:
        context = row["context"]
        if context not in grouped:
            grouped[context] = []
            order.append(context)
        grouped[context].append(row["completion"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for context in order:
            for ref in grouped[context]:
                handle.write(ref.replace("\n", " ").strip() + "\n")
            handle.write("\n")


def resolve_input(raw_dir: Path, split: str, override: str | None) -> Path:
    if override is not None:
        return Path(override)
    # Accept the canonical Yale-LILY naming and a short fallback so partners
    # can drop in either filename without renaming.
    candidates = [
        raw_dir / f"dart-v1.1.1-full-{split}.json",
        raw_dir / f"{split}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No DART JSON found for split={split!r} under {raw_dir}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


# DART's official splits are train/dev/test; this repo's pipeline uses the
# E2E convention of train/valid/test, so we map dev -> valid here.
SPLIT_TO_DART = {"train": "train", "valid": "dev", "test": "test"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Repo-side split names; mapped to DART's train/dev/test.",
    )
    parser.add_argument("--train-json", default=None)
    parser.add_argument("--valid-json", default=None)
    parser.add_argument("--test-json", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    raw_dir = ensure_dir(resolve_path(config, data_config["raw_dir"]))
    processed_dir = ensure_dir(resolve_path(config, data_config["processed_dir"]))

    overrides = {
        "train": args.train_json,
        "valid": args.valid_json,
        "test": args.test_json,
    }

    for split in args.splits:
        if split not in SPLIT_TO_DART:
            raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_TO_DART)}.")
        dart_split = SPLIT_TO_DART[split]
        json_path = resolve_input(raw_dir, dart_split, overrides[split])
        rows = convert_split(json_path)

        out_path = raw_dir / f"{split}.jsonl"
        write_jsonl(out_path, rows)
        print(f"wrote {len(rows)} rows to {out_path} (from {json_path.name})")

        if split == "test":
            ref_path = Path(resolve_path(
                config,
                config.get("evaluation", {}).get(
                    "references_file",
                    str(processed_dir / "references_test.txt"),
                ),
            ))
            write_multi_reference_file(ref_path, rows)
            unique_groups = len({row["context"] for row in rows})
            print(f"wrote {unique_groups} reference groups to {ref_path}")


if __name__ == "__main__":
    main()
