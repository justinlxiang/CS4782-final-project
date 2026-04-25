#!/usr/bin/env python
"""Generate E2E predictions from a trained adapter checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.checkpointing import load_adapter_checkpoint
from lora_gpt2.config import load_config, resolve_path
from lora_gpt2.data import format_prompt, load_e2e_records, raw_split_path
from lora_gpt2.generate import generate_continuations
from lora_gpt2.modeling import load_lora_model
from lora_gpt2.utils import get_device, write_jsonl


def batched(items: list[str], batch_size: int):
    """Yield fixed-size slices from a list."""
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    split = args.split or config["generation"].get("split", "test")
    batch_size = args.batch_size or int(config["generation"].get("batch_size", 1))
    if batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    model, tokenizer, _report = load_lora_model(config)
    load_adapter_checkpoint(args.adapter, model, strict=False)
    device = get_device()
    model.to(device)

    records = load_e2e_records(raw_split_path(config, split))
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise SystemExit("--max-examples must be positive when provided.")
        records = records[: args.max_examples]
    prompts = [format_prompt(record.context, config["data"]["prompt_template"]) for record in records]
    predictions = []
    for start, batch_prompts in tqdm(
        batched(prompts, batch_size),
        total=(len(prompts) + batch_size - 1) // batch_size,
        desc=f"generating {split}",
        unit="batch",
    ):
        generations = generate_continuations(model, tokenizer, batch_prompts, config, device)
        predictions.extend(
            {
                "id": str(start + offset),
                "prompt": prompt,
                "prediction": generation,
            }
            for offset, (prompt, generation) in enumerate(zip(batch_prompts, generations))
        )

    output_path = resolve_path(
        config,
        args.output_file or config["evaluation"]["predictions_file"],
    )
    write_jsonl(output_path, predictions)
    print(f"wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
