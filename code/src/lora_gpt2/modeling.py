"""Model and tokenizer construction helpers."""

from __future__ import annotations

from typing import Any

from lora_gpt2.inject import InjectionReport, inject_lora_into_gpt2


def load_tokenizer(config: dict[str, Any]):
    """Load the configured GPT-2 tokenizer."""
    from transformers import AutoTokenizer

    tokenizer_name = config["model"].get("tokenizer_name_or_path") or config["model"]["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if config["model"].get("pad_token") == "eos_token" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(config: dict[str, Any]):
    """Load the configured causal language model."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(config["model"]["name_or_path"])


def load_lora_model(config: dict[str, Any]):
    """Load GPT-2, inject LoRA, and return `(model, tokenizer, report)`."""
    model = load_base_model(config)
    tokenizer = load_tokenizer(config)
    report: InjectionReport = inject_lora_into_gpt2(model, config=config)
    return model, tokenizer, report
