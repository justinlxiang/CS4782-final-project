#!/usr/bin/env python
"""Prepare/tokenize E2E splits for GPT-2 conditional LM training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.config import load_config, resolve_path
from lora_gpt2.data import E2ETokenizedDataset, load_e2e_records, raw_split_path, save_tokenized_split
from lora_gpt2.modeling import load_tokenizer
from lora_gpt2.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = load_tokenizer(config)
    data_config = config["data"]
    processed_dir = ensure_dir(resolve_path(config, data_config["processed_dir"]))

    for split in args.splits:
        source = raw_split_path(config, split)
        records = load_e2e_records(source)
        dataset = E2ETokenizedDataset(
            records,
            tokenizer=tokenizer,
            prompt_template=data_config["prompt_template"],
            max_length=int(data_config["max_length"]),
            append_bos_to_context=bool(data_config.get("append_bos_to_context", False)),
            append_eos_to_target=bool(data_config.get("append_eos_to_target", True)),
            mask_prompt_labels=bool(data_config.get("mask_prompt_labels", True)),
            add_leading_space_to_target=bool(data_config.get("add_leading_space_to_target", True)),
        )
        output_path = processed_dir / f"{split}.jsonl"
        save_tokenized_split(output_path, dataset.examples)
        print(f"wrote {len(dataset)} examples to {output_path}")


if __name__ == "__main__":
    main()
