"""LoRA GPT-2 Medium E2E replication package."""

from lora_gpt2.config import load_config

__all__ = [
    "LoRALinear",
    "AdaLoRAQKVConv1D",
    "LoRAQKVConv1D",
    "inject_lora_into_gpt2",
    "load_config",
    "mark_only_lora_as_trainable",
]


def __getattr__(name: str):
    """Load torch-dependent helpers only when they are requested."""
    if name in {"LoRALinear", "AdaLoRAQKVConv1D", "LoRAQKVConv1D"}:
        from lora_gpt2.lora_layers import AdaLoRAQKVConv1D, LoRALinear, LoRAQKVConv1D

        return {
            "LoRALinear": LoRALinear,
            "AdaLoRAQKVConv1D": AdaLoRAQKVConv1D,
            "LoRAQKVConv1D": LoRAQKVConv1D,
        }[name]
    if name in {"inject_lora_into_gpt2", "mark_only_lora_as_trainable"}:
        from lora_gpt2.inject import inject_lora_into_gpt2, mark_only_lora_as_trainable

        return {
            "inject_lora_into_gpt2": inject_lora_into_gpt2,
            "mark_only_lora_as_trainable": mark_only_lora_as_trainable,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
