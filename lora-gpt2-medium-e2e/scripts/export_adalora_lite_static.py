#!/usr/bin/env python
# pyright: reportMissingImports=false
"""Export an AdaLoRA-Lite checkpoint as a compact static LoRA adapter."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_gpt2.config import load_config, resolve_path
from lora_gpt2.utils import ensure_dir


_A_KEY = re.compile(
    r"^(?P<prefix>.*transformer\.h\.(?P<layer>\d+)\.attn\.c_attn)"
    r"\.lora_A\.(?P<slice>query|key|value)$"
)


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if "adapter_state_dict" in payload:
        return payload
    return {"adapter_state_dict": payload, "config": {}, "extra": {}}


def _build_static_payload(
    payload: dict[str, Any],
    fallback_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    state: dict[str, torch.Tensor] = payload["adapter_state_dict"]
    config = copy.deepcopy(payload.get("config") or fallback_config)
    lora_config = config.setdefault("lora", {})
    rank = int(lora_config.get("rank", 16))
    alpha = float(lora_config.get("alpha", 32))
    adaptive_scaling = alpha / rank if rank > 0 else 0.0
    rank_pattern: list[dict[str, int]] = []
    alpha_pattern: list[dict[str, float]] = []
    static_state: dict[str, torch.Tensor] = {}
    allocation_rows: list[dict[str, Any]] = []

    layer_entries: dict[int, dict[str, tuple[str, torch.Tensor]]] = {}
    for key, tensor in state.items():
        match = _A_KEY.match(key)
        if match is None:
            continue
        layer = int(match.group("layer"))
        slice_name = match.group("slice")
        layer_entries.setdefault(layer, {})[slice_name] = (match.group("prefix"), tensor)

    if not layer_entries:
        raise ValueError("No AdaLoRA-Lite lora_A tensors found in checkpoint.")

    for layer in range(max(layer_entries) + 1):
        layer_rank_pattern = {"query": 0, "key": 0, "value": 0}
        layer_alpha_pattern = {"query": 0.0, "key": 0.0, "value": 0.0}
        for slice_name, value in layer_entries.get(layer, {}).items():
            prefix, lora_a = value
            lora_b = state[f"{prefix}.lora_B.{slice_name}"]
            lora_s = state[f"{prefix}.lora_s.{slice_name}"]
            mask = state[f"{prefix}.lora_mask_{slice_name}"]
            active = torch.nonzero(mask > 0, as_tuple=False).flatten()
            active_rank = int(active.numel())
            layer_rank_pattern[slice_name] = active_rank
            layer_alpha_pattern[slice_name] = active_rank * adaptive_scaling
            allocation_rows.append(
                {
                    "layer": layer,
                    "slice": slice_name,
                    "active_rank": active_rank,
                    "active_indices": [int(index) for index in active.tolist()],
                }
            )
            if active_rank == 0:
                continue
            static_state[f"{prefix}.lora_A.{slice_name}"] = lora_a.index_select(0, active).clone()
            static_state[f"{prefix}.lora_B.{slice_name}"] = (
                lora_b.index_select(1, active) * lora_s.index_select(0, active).unsqueeze(0)
            ).clone()
        rank_pattern.append(layer_rank_pattern)
        alpha_pattern.append(layer_alpha_pattern)

    lora_config["method"] = "lora"
    lora_config["rank_pattern"] = rank_pattern
    lora_config["alpha_pattern"] = alpha_pattern
    lora_config.pop("adalora_lite", None)
    static_payload = {
        "adapter_state_dict": static_state,
        "config": config,
        "extra": {
            "source": "adalora_lite_static_export",
            "allocation": allocation_rows,
            "source_extra": payload.get("extra", {}),
        },
    }
    return static_payload, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="AdaLoRA-Lite config used to train the adapter.")
    parser.add_argument("--adapter", required=True, help="AdaLoRA-Lite adapter checkpoint.")
    parser.add_argument("--output-adapter", required=True, help="Output compact static LoRA checkpoint.")
    parser.add_argument("--output-config", required=True, help="Output static LoRA config JSON.")
    parser.add_argument("--allocation-json", default=None, help="Optional allocation report path.")
    args = parser.parse_args()

    config = load_config(args.config)
    payload = _load_payload(resolve_path(config, args.adapter))
    static_payload, static_config = _build_static_payload(payload, fallback_config=config)

    output_adapter = resolve_path(config, args.output_adapter)
    output_config = resolve_path(config, args.output_config)
    ensure_dir(output_adapter.parent)
    ensure_dir(output_config.parent)
    torch.save(static_payload, output_adapter)
    output_config.write_text(json.dumps(static_config, indent=2, sort_keys=True), encoding="utf-8")
    if args.allocation_json:
        allocation_path = resolve_path(config, args.allocation_json)
        ensure_dir(allocation_path.parent)
        allocation_path.write_text(
            json.dumps(static_payload["extra"]["allocation"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(f"static_adapter={output_adapter}")
    print(f"static_config={output_config}")
    print(f"static_tensors={len(static_payload['adapter_state_dict'])}")


if __name__ == "__main__":
    main()
