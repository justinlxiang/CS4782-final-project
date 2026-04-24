#!/usr/bin/env python
"""Report expected or actual LoRA parameter counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.config import load_config
from lora_gpt2.inject import expected_qv_lora_parameters, inject_lora_into_gpt2
from lora_gpt2.modeling import load_base_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--load-model", action="store_true", help="Actually load GPT-2 and inject LoRA.")
    args = parser.parse_args()

    config = load_config(args.config)
    rank = int(config["lora"]["rank"])
    hidden_size = 1024
    num_layers = 24
    expected = expected_qv_lora_parameters(hidden_size, num_layers, rank)
    print(f"expected_qv_lora_parameters={expected}")

    if args.load_model:
        model = load_base_model(config)
        report = inject_lora_into_gpt2(model, config=config)
        print(f"actual_trainable_parameters={report.trainable_parameters}")
        print(f"replaced_modules={report.replaced_modules}")


if __name__ == "__main__":
    main()
