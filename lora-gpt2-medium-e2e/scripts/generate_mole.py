#!/usr/bin/env python
"""Generate predictions through a MoLE-routed model.

Mirrors `scripts/generate.py` but constructs the model with N frozen LoRA
experts plus a learned router instead of a single LoRA adapter. Each task
config (E2E / DART / WebNLG) is read separately to recover the right raw
data path and prompt template per task; the MoLE-config gives us the shared
expert layout and the trained router weights.

Usage:

    python scripts/generate_mole.py \\
        --mole-config configs/mole_e2e_dart_webnlg.yaml \\
        --task-config configs/dart_gpt2_medium_lora.yaml \\
        --router outputs/runs/mole_e2e_dart_webnlg/checkpoints/router_final.pt \\
        --split test

If `--force-expert NAME` is passed, the router is bypassed by pinning the
gate to that expert's slot (using `mole_config["mole"]["expert_names"]` for
name → index lookup). This is the simplest way to verify that a routed
generation through expert k matches the standalone single-adapter run, and
to ablate routing as a baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.config import load_config, resolve_path
from lora_gpt2.data import format_prompt, load_e2e_records, raw_split_path
from lora_gpt2.generate import generate_continuations
from lora_gpt2.modeling import load_base_model, load_tokenizer
from lora_gpt2.mole_inject import (
    force_expert_weights,
    inject_mole_into_gpt2,
    load_lora_experts_into_mole,
)
from lora_gpt2.utils import get_device, write_jsonl


def batched(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _load_router(path: Path, model: torch.nn.Module) -> tuple[list[str], list[str]]:
    """Load only the router state into a freshly-injected MoLE model."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("router_state_dict", payload)
    result = model.load_state_dict(state, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mole-config", required=True,
        help="Config defining the expert layout (e.g., configs/mole_e2e_dart_webnlg.yaml).",
    )
    parser.add_argument(
        "--task-config", required=True,
        help="Per-task config providing data paths, prompt template, and decoding settings.",
    )
    parser.add_argument(
        "--router", default=None,
        help="Trained router checkpoint. Omit only when --force-expert is set.",
    )
    parser.add_argument("--force-expert", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    if args.router is None and args.force_expert is None:
        raise SystemExit("Provide either --router (a trained gate) or --force-expert.")

    mole_config = load_config(args.mole_config)
    task_config = load_config(args.task_config)
    split = args.split or task_config["generation"].get("split", "test")
    batch_size = args.batch_size or int(task_config["generation"].get("batch_size", 1))
    if batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    print("loading base model + tokenizer...")
    model = load_base_model(mole_config)
    tokenizer = load_tokenizer(mole_config)

    expert_paths = [resolve_path(mole_config, p) for p in mole_config["mole"]["expert_paths"]]
    expert_names = mole_config["mole"].get(
        "expert_names", [f"expert{i}" for i in range(len(expert_paths))]
    )
    print(f"injecting MoLE with {len(expert_paths)} experts: {expert_names}")
    inject_mole_into_gpt2(
        model,
        num_experts=len(expert_paths),
        gate_init=mole_config["mole"].get("gate_init", "uniform"),
        config=mole_config,
    )
    load_lora_experts_into_mole(model, expert_paths)

    if args.force_expert is not None:
        if args.force_expert not in expert_names:
            raise SystemExit(
                f"--force-expert {args.force_expert!r} not in {expert_names}"
            )
        idx = expert_names.index(args.force_expert)
        print(f"forcing gate to expert {idx} ({args.force_expert})")
        force_expert_weights(model, expert_idx=idx)
    else:
        print(f"loading router from {args.router}")
        missing, unexpected = _load_router(Path(args.router), model)
        if unexpected:
            print(f"warn: unexpected router keys: {unexpected}")
        # Missing non-router keys are expected (experts/base aren't in this file).
        non_router_unloaded = [k for k in missing if "mole_gate_" in k]
        if non_router_unloaded:
            raise SystemExit(f"router checkpoint missing keys: {non_router_unloaded}")

    device = get_device()
    model.to(device)
    model.eval()

    records = load_e2e_records(raw_split_path(task_config, split))
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise SystemExit("--max-examples must be positive when provided.")
        records = records[: args.max_examples]

    prompts = [
        format_prompt(record.context, task_config["data"]["prompt_template"])
        for record in records
    ]
    predictions = []
    for start, batch_prompts in tqdm(
        batched(prompts, batch_size),
        total=(len(prompts) + batch_size - 1) // batch_size,
        desc=f"generating {split}",
        unit="batch",
    ):
        # Decoding settings come from the task config, not the MoLE config,
        # because beam/length-penalty/no-repeat-ngram are task-specific.
        generations = generate_continuations(model, tokenizer, batch_prompts, task_config, device)
        predictions.extend(
            {
                "id": str(start + offset),
                "prompt": prompt,
                "prediction": generation,
            }
            for offset, (prompt, generation) in enumerate(zip(batch_prompts, generations))
        )

    output_path = resolve_path(
        task_config,
        args.output_file or task_config["evaluation"]["predictions_file"],
    )
    write_jsonl(output_path, predictions)
    print(f"wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
