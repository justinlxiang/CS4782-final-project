#!/usr/bin/env python
"""Convert raw WebNLG JSON splits into the JSONL format consumed by prepare_e2e.py.

Microsoft's LoRA repo hosts the WebNLG 2017 Challenge data in JSON form at
examples/NLG/data/webnlg_challenge_2017/{train,dev,test}.json. Each split is
a dict with an "entries" list, where each entry is itself a single-key dict
mapping a numeric id to a record holding:

  - modifiedtripleset: list of {subject, property, object} dicts
  - lexicalisations:  list of {lex, comment, lang, xml_id} dicts

We linearize the triples into a single GPT-2 prompt context using the same
`<H> s <R> p <T> o` markers used for DART (the official LoRA NLG preprocessing
convention), and emit one (context, completion) JSONL row per
(record, lexicalisation) pair, filtering to lexicalisations marked
`comment: "good"` per the WebNLG challenge convention.

For the test split we also write a multi-reference text file in the
blank-line-separated format the official WebNLG / DART / E2E Perl evaluators
all consume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.config import load_config, resolve_path
from lora_gpt2.utils import ensure_dir, write_jsonl


HEAD_TOKEN = "<H>"
RELATION_TOKEN = "<R>"
TAIL_TOKEN = "<T>"


def linearize_tripleset(tripleset: list[dict]) -> str:
    parts: list[str] = []
    for triple in tripleset:
        subject = str(triple.get("subject", "")).strip()
        relation = str(triple.get("property", "")).strip()
        obj = str(triple.get("object", "")).strip()
        if not (subject and relation and obj):
            raise ValueError(f"Incomplete WebNLG triple: {triple!r}")
        parts.append(f"{HEAD_TOKEN} {subject} {RELATION_TOKEN} {relation} {TAIL_TOKEN} {obj}")
    return " ".join(parts)


def good_lexicalisations(record: dict) -> list[str]:
    """Return the human-written verbalizations marked `comment: "good"`.

    Some WebNLG entries also contain `comment: "bad"` lexicalisations that
    were rejected during data collection; the original challenge convention
    excludes them from training and evaluation.
    """
    texts: list[str] = []
    for entry in record.get("lexicalisations") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("comment", "good") != "good":
            continue
        text = entry.get("lex")
        if text:
            texts.append(str(text).strip())
    return texts


def convert_split(json_path: Path) -> list[dict[str, str]]:
    """Expand each WebNLG record into (context, completion) JSONL rows."""
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows: list[dict[str, str]] = []
    for entry in payload.get("entries", []):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"Unexpected WebNLG entry shape: {entry!r}")
        # Each entry is wrapped in a single-key dict like {"42": {...}}; we
        # only care about the inner record, not the numeric id.
        record = next(iter(entry.values()))
        tripleset = record.get("modifiedtripleset") or []
        if not tripleset:
            continue
        context = linearize_tripleset(tripleset)
        refs = good_lexicalisations(record)
        if not refs:
            continue
        for ref in refs:
            rows.append({"context": context, "completion": ref})
    return rows


def write_multi_reference_file(path: Path, rows: list[dict[str, str]]) -> None:
    """Write blank-line-separated reference groups (one block per unique context)."""
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


def resolve_input(raw_dir: Path, webnlg_split: str, override: str | None) -> Path:
    if override is not None:
        return Path(override)
    candidate = raw_dir / f"{webnlg_split}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"No WebNLG JSON found for split={webnlg_split!r} under {raw_dir}. "
        f"Expected: {candidate}"
    )


# Repo splits (E2E convention) → WebNLG official names.
SPLIT_TO_WEBNLG = {"train": "train", "valid": "dev", "test": "test"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Repo-side split names; mapped to WebNLG's train/dev/test.",
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
        if split not in SPLIT_TO_WEBNLG:
            raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_TO_WEBNLG)}.")
        webnlg_split = SPLIT_TO_WEBNLG[split]
        json_path = resolve_input(raw_dir, webnlg_split, overrides[split])
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
