#!/usr/bin/env python
"""Generate predictions for one (variant, task, split) cell of the eval matrix.

Variants supported:
  base                             pretrained GPT-2 medium, no adapters
  e2e | dart | webnlg              base + that task's single-task LoRA
  mole                             base + 3 frozen LoRAs + trained MoLE router

For each call, predictions are deduplicated to one prediction per unique
meaning representation (context). This both speeds generation (4-12x
fewer prompts) and matches what each task's official evaluator expects:
one hypothesis per group, with the references bundled as a multi-ref
group per entry. Per-row predictions of the same MR are identical
under deterministic decoding, so dedup is lossless.

Decoding settings come from the *task's* generation config block, so
when a non-matching LoRA is being evaluated on a different task it
still uses that task's beam/length-penalty/no-repeat-ngram. Decoding
stays consistent within a column of the matrix; only the model varies.

Output is a JSONL file with `{id, prompt, prediction}` per unique MR,
in the same deterministic order across variants so the side-by-side
comparison aligns row-for-row.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.checkpointing import load_adapter_checkpoint
from lora_gpt2.config import load_config, resolve_path
from lora_gpt2.data import format_prompt, load_e2e_records, raw_split_path
from lora_gpt2.generate import generate_continuations
from lora_gpt2.inject import inject_lora_into_gpt2
from lora_gpt2.modeling import load_base_model, load_tokenizer
from lora_gpt2.mole_inject import (
    inject_mole_into_gpt2,
    load_lora_experts_into_mole,
)
from lora_gpt2.utils import write_jsonl

# Used both at construction time (which adapter to load) and as part
# of the canonical output path so cells of the matrix never collide.
SINGLE_TASK_VARIANTS = ("e2e", "dart", "webnlg")
SUPPORTED_VARIANTS = ("base",) + SINGLE_TASK_VARIANTS + ("mole",)


def _build_model(
    variant: str,
    task_config: dict,
    *,
    adapter_paths: dict[str, Path],
    mole_config: dict | None,
    mole_router_path: Path | None,
):
    """Construct a model+tokenizer for the requested variant.

    The base model and tokenizer are always loaded from `task_config` so
    that pad-token handling and tokenizer choice are consistent with the
    decoding pipeline regardless of variant.
    """
    tokenizer = load_tokenizer(task_config)
    model = load_base_model(task_config)

    if variant == "base":
        return model, tokenizer

    if variant in SINGLE_TASK_VARIANTS:
        # All three single-task LoRAs trained with rank=4, alpha=32, q+v
        # targets, so injecting under the task_config (which already has
        # those values) produces a structurally compatible adapter slot.
        inject_lora_into_gpt2(model, config=task_config)
        adapter_path = adapter_paths[variant]
        load_adapter_checkpoint(adapter_path, model, strict=False)
        return model, tokenizer

    if variant == "mole":
        if mole_config is None or mole_router_path is None:
            raise ValueError("mole variant requires --mole-config and --mole-router")
        inject_mole_into_gpt2(
            model,
            num_experts=len(mole_config["mole"]["expert_paths"]),
            config=mole_config,
        )
        expert_paths = [resolve_path(mole_config, p) for p in mole_config["mole"]["expert_paths"]]
        load_lora_experts_into_mole(model, expert_paths)
        # Router state lives by itself under "router_state_dict" (see
        # scripts/train_mole.py). load_state_dict with strict=False lets
        # the router weights drop into mole_gate_* params without the
        # base/expert keys being present.
        payload = torch.load(mole_router_path, map_location="cpu", weights_only=False)
        state = payload.get("router_state_dict", payload)
        model.load_state_dict(state, strict=False)
        return model, tokenizer

    raise ValueError(f"Unknown variant: {variant!r} (expected one of {SUPPORTED_VARIANTS})")


def _unique_contexts_in_order(records) -> list[str]:
    """Preserve first-occurrence order of unique meaning representations.

    First-occurrence ordering is canonical (independent of dict hashing
    and Python version) so different runs of the matrix produce
    line-aligned outputs that can be diffed cell-by-cell.
    """
    seen = set()
    ordered: list[str] = []
    for record in records:
        if record.context in seen:
            continue
        seen.add(record.context)
        ordered.append(record.context)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        required=True,
        choices=SUPPORTED_VARIANTS,
        help="Which model to evaluate.",
    )
    parser.add_argument(
        "--task-config",
        required=True,
        help="Per-task config (data paths + decoding settings used by this run).",
    )
    parser.add_argument(
        "--split",
        default="valid",
        help="Raw split name (default: valid). Must exist under config['data']['raw_dir'].",
    )
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--beam-override", type=int, default=None,
                        help="If set, override config['generation']['num_beams'] uniformly.")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Cap on unique contexts (after dedup) for quick smoke runs.")
    # E2E / DART / WebNLG single-task adapter paths. Repository convention is
    # `<task>_lora_r4_alpha32/checkpoints/adapter_final.pt` relative to the
    # project root; override here if the layout is different.
    parser.add_argument("--e2e-adapter", default="e2e_lora_r4_alpha32/checkpoints/adapter_final.pt")
    parser.add_argument("--dart-adapter", default="dart_lora_r4_alpha32/checkpoints/adapter_final.pt")
    parser.add_argument("--webnlg-adapter", default="webnlg_lora_r4_alpha32/checkpoints/adapter_final.pt")
    parser.add_argument("--mole-config", default=None,
                        help="Required when --variant=mole.")
    parser.add_argument("--mole-router", default=None,
                        help="Required when --variant=mole. Path to router_*.pt.")
    args = parser.parse_args()

    task_config = load_config(args.task_config)

    # Optional uniform beam override — cuts compute roughly linearly.
    # Applied once here so generate_continuations consumes the modified
    # config naturally (no separate code path for the override).
    if args.beam_override is not None:
        task_config["generation"]["num_beams"] = int(args.beam_override)

    adapter_paths: dict[str, Path] = {
        "e2e": Path(args.e2e_adapter),
        "dart": Path(args.dart_adapter),
        "webnlg": Path(args.webnlg_adapter),
    }
    mole_config = load_config(args.mole_config) if args.mole_config else None
    mole_router_path = Path(args.mole_router) if args.mole_router else None

    print(f"variant={args.variant}  task_config={args.task_config}  split={args.split}")
    model, tokenizer = _build_model(
        args.variant,
        task_config,
        adapter_paths=adapter_paths,
        mole_config=mole_config,
        mole_router_path=mole_router_path,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    raw_records = load_e2e_records(raw_split_path(task_config, args.split))
    contexts = _unique_contexts_in_order(raw_records)
    if args.max_examples is not None:
        contexts = contexts[: args.max_examples]
    print(f"raw rows: {len(raw_records)}  unique MRs: {len(contexts)}")

    prompts = [format_prompt(c, task_config["data"]["prompt_template"]) for c in contexts]
    batch_size = args.batch_size or int(task_config["generation"].get("batch_size", 4))
    if batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    predictions: list[dict[str, str]] = []
    # mininterval=2.0 keeps the progress bar useful when stdout/stderr are
    # piped (subprocess from a notebook) instead of attached to a TTY: tqdm
    # prints a fresh line on each update in non-TTY mode, so without
    # throttling we'd flood the cell with hundreds of lines per run.
    for batch_start in tqdm(
        range(0, len(prompts), batch_size),
        total=(len(prompts) + batch_size - 1) // batch_size,
        desc=f"{args.variant} on {Path(args.task_config).stem}/{args.split}",
        unit="batch",
        mininterval=2.0,
    ):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        batch_contexts = contexts[batch_start : batch_start + batch_size]
        gens = generate_continuations(model, tokenizer, batch_prompts, task_config, device)
        for offset, (context, gen) in enumerate(zip(batch_contexts, gens)):
            predictions.append({
                "id": str(batch_start + offset),
                "context": context,
                "prompt": batch_prompts[offset],
                "prediction": gen,
            })

    output_path = Path(args.output_file)
    write_jsonl(output_path, predictions)
    print(f"wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
