#!/usr/bin/env python
"""Generate E2E predictions from a trained adapter checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.checkpointing import load_adapter_checkpoint
from lora_gpt2.config import load_config, resolve_path
from lora_gpt2.data import format_prompt, load_e2e_records, raw_split_path
from lora_gpt2.generate import format_predictions, generate_continuations
from lora_gpt2.modeling import load_lora_model
from lora_gpt2.utils import get_device, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    split = args.split or config["generation"].get("split", "test")
    model, tokenizer, _report = load_lora_model(config)
    load_adapter_checkpoint(args.adapter, model, strict=False)
    device = get_device()
    model.to(device)

    records = load_e2e_records(raw_split_path(config, split))
    prompts = [format_prompt(record.context, config["data"]["prompt_template"]) for record in records]
    generations = generate_continuations(model, tokenizer, prompts, config, device)
    predictions = format_predictions(prompts, generations)

    output_path = resolve_path(config, config["evaluation"]["predictions_file"])
    write_jsonl(output_path, predictions)
    print(f"wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
