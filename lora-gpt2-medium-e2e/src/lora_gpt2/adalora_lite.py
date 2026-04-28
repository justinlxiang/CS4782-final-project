"""AdaLoRA-Lite rank allocation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from lora_gpt2.lora_layers import AdaLoRAQKVConv1D


_SLICE_ORDER = {name: index for index, name in enumerate(AdaLoRAQKVConv1D.slice_names)}


@dataclass(frozen=True)
class RankComponent:
    """Identifier for one rank component in one adapted Q/K/V slice."""

    module_name: str
    layer_index: int
    slice_name: str
    component_index: int


def iter_adalora_lite_modules(model: torch.nn.Module) -> Iterable[tuple[str, AdaLoRAQKVConv1D]]:
    """Yield all AdaLoRA-Lite QKV modules in a model."""
    for name, module in model.named_modules():
        if isinstance(module, AdaLoRAQKVConv1D):
            yield name, module


class AdaLoRALiteRankAllocator:
    """Global top-K rank allocator for AdaLoRA-Lite masks."""

    def __init__(
        self,
        target_rank_units: int,
        mask_interval: int = 200,
        init_warmup_steps: int = 500,
        beta: float = 0.85,
        min_rank_per_slice: int = 0,
    ) -> None:
        if target_rank_units <= 0:
            raise ValueError("target_rank_units must be positive.")
        if mask_interval <= 0:
            raise ValueError("mask_interval must be positive.")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1).")
        if min_rank_per_slice < 0:
            raise ValueError("min_rank_per_slice must be non-negative.")
        self.target_rank_units = int(target_rank_units)
        self.mask_interval = int(mask_interval)
        self.init_warmup_steps = int(init_warmup_steps)
        self.beta = float(beta)
        self.min_rank_per_slice = int(min_rank_per_slice)
        self.ema_scores: dict[RankComponent, float] = {}
        self.last_mask_step: int | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AdaLoRALiteRankAllocator | None":
        """Build an allocator if the config enables AdaLoRA-Lite."""
        lora_config = config.get("lora", {})
        if str(lora_config.get("method", "lora")).lower() != "adalora_lite":
            return None
        adalora_config = lora_config.get("adalora_lite", {})
        return cls(
            target_rank_units=int(adalora_config.get("target_rank_units", 192)),
            mask_interval=int(adalora_config.get("mask_interval", 200)),
            init_warmup_steps=int(adalora_config.get("init_warmup_steps", 500)),
            beta=float(adalora_config.get("beta", 0.85)),
            min_rank_per_slice=int(adalora_config.get("min_rank_per_slice", 0)),
        )

    def components(self, model: torch.nn.Module) -> list[tuple[RankComponent, AdaLoRAQKVConv1D]]:
        """Return every enabled rank component with its owning module."""
        components: list[tuple[RankComponent, AdaLoRAQKVConv1D]] = []
        for module_name, module in iter_adalora_lite_modules(model):
            layer_index = int(module.layer_index if module.layer_index is not None else -1)
            for _slice_idx, slice_name in module.enabled_slices:
                for component_index in range(module.rank):
                    components.append(
                        (
                            RankComponent(
                                module_name=module_name,
                                layer_index=layer_index,
                                slice_name=slice_name,
                                component_index=component_index,
                            ),
                            module,
                        )
                    )
        return components

    def update_scores(self, model: torch.nn.Module) -> dict[str, Any]:
        """Update EMA importance scores from current `s.grad` values."""
        observed = 0
        for module_name, module in iter_adalora_lite_modules(model):
            layer_index = int(module.layer_index if module.layer_index is not None else -1)
            for _slice_idx, slice_name in module.enabled_slices:
                scales = module.lora_s[slice_name]
                if scales.grad is None:
                    continue
                raw_scores = (scales.detach() * scales.grad.detach()).abs().float().cpu()
                for component_index, raw_score in enumerate(raw_scores.tolist()):
                    component = RankComponent(
                        module_name=module_name,
                        layer_index=layer_index,
                        slice_name=slice_name,
                        component_index=component_index,
                    )
                    previous = self.ema_scores.get(component, float(raw_score))
                    self.ema_scores[component] = (
                        self.beta * previous + (1.0 - self.beta) * float(raw_score)
                    )
                    observed += 1
        return {"observed_components": observed, "tracked_components": len(self.ema_scores)}

    def should_update_masks(self, global_step: int) -> bool:
        """Return whether masks should be updated at this step."""
        if global_step < self.init_warmup_steps:
            return False
        return global_step % self.mask_interval == 0

    def state_dict(self) -> dict[str, Any]:
        """Return allocator state for training-checkpoint resume."""
        return {
            "target_rank_units": self.target_rank_units,
            "mask_interval": self.mask_interval,
            "init_warmup_steps": self.init_warmup_steps,
            "beta": self.beta,
            "min_rank_per_slice": self.min_rank_per_slice,
            "last_mask_step": self.last_mask_step,
            "ema_scores": [
                {
                    "module_name": component.module_name,
                    "layer_index": component.layer_index,
                    "slice_name": component.slice_name,
                    "component_index": component.component_index,
                    "score": score,
                }
                for component, score in self.ema_scores.items()
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore allocator EMA scores from a training checkpoint."""
        self.last_mask_step = state.get("last_mask_step")
        self.ema_scores = {
            RankComponent(
                module_name=str(item["module_name"]),
                layer_index=int(item["layer_index"]),
                slice_name=str(item["slice_name"]),
                component_index=int(item["component_index"]),
            ): float(item["score"])
            for item in state.get("ema_scores", [])
        }

    def update_masks(self, model: torch.nn.Module, global_step: int) -> dict[str, Any] | None:
        """Globally keep top-scoring components and zero the rest."""
        if not self.should_update_masks(global_step):
            return None

        components = self.components(model)
        if not components:
            return None
        if self.target_rank_units > len(components):
            raise ValueError(
                f"target_rank_units={self.target_rank_units} exceeds available components={len(components)}."
            )

        keep = self._select_components(components)
        masks_by_module: dict[tuple[str, str], torch.Tensor] = {}
        modules_by_name = {module_name: module for module_name, module in iter_adalora_lite_modules(model)}
        for component, _module in components:
            key = (component.module_name, component.slice_name)
            if key not in masks_by_module:
                module = modules_by_name[component.module_name]
                masks_by_module[key] = torch.zeros(
                    module.rank,
                    device=module.lora_mask(component.slice_name).device,
                    dtype=module.lora_mask(component.slice_name).dtype,
                )
            if component in keep:
                masks_by_module[key][component.component_index] = 1

        for (module_name, slice_name), mask in masks_by_module.items():
            modules_by_name[module_name].set_lora_mask(slice_name, mask)

        self.last_mask_step = int(global_step)
        return self.allocation_summary(model, global_step=global_step)

    def _select_components(
        self,
        components: list[tuple[RankComponent, AdaLoRAQKVConv1D]],
    ) -> set[RankComponent]:
        """Select components with deterministic tie-breaking."""
        sorted_components = sorted(
            (component for component, _module in components),
            key=self._sort_key,
        )

        keep: set[RankComponent] = set()
        if self.min_rank_per_slice > 0:
            by_slice: dict[tuple[str, str], list[RankComponent]] = {}
            for component in sorted_components:
                by_slice.setdefault((component.module_name, component.slice_name), []).append(component)
            required = self.min_rank_per_slice * len(by_slice)
            if required > self.target_rank_units:
                raise ValueError(
                    "min_rank_per_slice requires more components than target_rank_units."
                )
            for slice_components in by_slice.values():
                keep.update(slice_components[: self.min_rank_per_slice])

        for component in sorted_components:
            if len(keep) >= self.target_rank_units:
                break
            keep.add(component)
        return keep

    def _sort_key(self, component: RankComponent) -> tuple[float, int, int, str, int]:
        """Sort by descending score, then stable layer/slice/component identity."""
        score = self.ema_scores.get(component, 0.0)
        return (
            -float(score),
            component.layer_index,
            _SLICE_ORDER.get(component.slice_name, 99),
            component.module_name,
            component.component_index,
        )

    def active_rank_units(self, model: torch.nn.Module) -> int:
        """Return total active rank components across all AdaLoRA-Lite modules."""
        total = 0
        for _name, module in iter_adalora_lite_modules(model):
            for _slice_idx, slice_name in module.enabled_slices:
                total += module.active_rank(slice_name)
        return total

    def allocation_summary(
        self,
        model: torch.nn.Module,
        global_step: int | None = None,
    ) -> dict[str, Any]:
        """Return a serializable active-rank summary by layer and slice."""
        layers: list[dict[str, Any]] = []
        for module_name, module in iter_adalora_lite_modules(model):
            ranks = module.active_rank_by_slice()
            layers.append(
                {
                    "module": module_name,
                    "layer": module.layer_index,
                    "query": ranks["query"],
                    "key": ranks["key"],
                    "value": ranks["value"],
                    "total": sum(ranks.values()),
                }
            )
        layers.sort(key=lambda item: (-1 if item["layer"] is None else int(item["layer"])))
        return {
            "type": "adalora_lite_allocation",
            "step": global_step,
            "target_rank_units": self.target_rank_units,
            "active_rank_units": sum(int(item["total"]) for item in layers),
            "layers": layers,
        }
