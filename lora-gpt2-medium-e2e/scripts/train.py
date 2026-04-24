#!/usr/bin/env python
"""Training entrypoint. Use `--dry-run` before approved training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.config import load_config
from lora_gpt2.modeling import load_lora_model
from lora_gpt2.train import create_optimizer
from lora_gpt2.utils import parameter_report, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Build model/optimizer and exit.")
    args = parser.parse_args()

    if not args.dry_run:
        raise SystemExit(
            "Training execution is intentionally blocked for now. "
            "Use --dry-run, or remove this guard only after explicit approval."
        )

    config = load_config(args.config)
    seed_everything(int(config["project"]["seed"]))
    model, _tokenizer, report = load_lora_model(config)
    optimizer = create_optimizer(model, config)
    params = parameter_report(model)

    print(f"dry_run=true")
    print(f"replaced_modules={report.replaced_modules}")
    print(f"trainable_parameters={params['trainable']}")
    print(f"optimizer_param_groups={len(optimizer.param_groups)}")


if __name__ == "__main__":
    main()
