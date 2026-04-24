"""E2E NLG parsing, tokenization, and batching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from lora_gpt2.config import resolve_path
from lora_gpt2.utils import read_jsonl, write_jsonl


@dataclass(frozen=True)
class E2ERecord:
    """One E2E meaning representation and reference completion."""

    context: str
    completion: str


def parse_e2e_line(line: str) -> E2ERecord:
    """Parse an official-style `context||completion` E2E line."""
    if "||" not in line:
        raise ValueError("Expected an E2E line with `context||completion`.")
    context, completion = line.split("||", 1)
    return E2ERecord(context=context.strip(), completion=completion.strip())


def load_e2e_records(path: str | Path) -> list[E2ERecord]:
    """Load E2E records from raw `.txt` or JSONL files."""
    data_path = Path(path)
    if data_path.suffix == ".jsonl":
        records: list[E2ERecord] = []
        for item in read_jsonl(data_path):
            context = item.get("context") or item.get("mr")
            completion = item.get("completion") or item.get("target") or item.get("ref")
            if context is None or completion is None:
                raise ValueError(f"Missing context/completion fields in {data_path}.")
            records.append(E2ERecord(context=str(context), completion=str(completion)))
        return records

    records = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(parse_e2e_line(stripped))
    return records


def format_prompt(context: str, template: str) -> str:
    """Format a meaning representation into the conditional GPT-2 prompt."""
    return template.format(mr=context)


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    """Tokenize with either a Hugging Face tokenizer or a simple test double."""
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def tokenize_record(
    record: E2ERecord,
    tokenizer: Any,
    prompt_template: str,
    max_length: int,
    append_eos_to_target: bool = True,
    mask_prompt_labels: bool = True,
    add_leading_space_to_target: bool = True,
) -> dict[str, Any]:
    """Tokenize one E2E example and build completion-only labels."""
    prompt = format_prompt(record.context, prompt_template)
    target = record.completion
    if add_leading_space_to_target and target and not target.startswith(" "):
        target = " " + target

    prompt_ids = _tokenize(tokenizer, prompt)
    target_ids = _tokenize(tokenizer, target)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if append_eos_to_target and eos_token_id is not None:
        target_ids = target_ids + [int(eos_token_id)]

    input_ids = (prompt_ids + target_ids)[:max_length]
    prompt_label = -100 if mask_prompt_labels else None
    labels = (
        ([prompt_label] * len(prompt_ids) if mask_prompt_labels else prompt_ids)
        + target_ids
    )[:max_length]

    return {
        "context": record.context,
        "completion": record.completion,
        "prompt": prompt,
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_length": min(len(prompt_ids), len(input_ids)),
    }


class E2ETokenizedDataset(Dataset[dict[str, Any]]):
    """Tokenized E2E records for conditional causal language modeling."""

    def __init__(
        self,
        records: Sequence[E2ERecord],
        tokenizer: Any,
        prompt_template: str,
        max_length: int,
        append_eos_to_target: bool = True,
        mask_prompt_labels: bool = True,
        add_leading_space_to_target: bool = True,
    ) -> None:
        self.examples = [
            tokenize_record(
                record=record,
                tokenizer=tokenizer,
                prompt_template=prompt_template,
                max_length=max_length,
                append_eos_to_target=append_eos_to_target,
                mask_prompt_labels=mask_prompt_labels,
                add_leading_space_to_target=add_leading_space_to_target,
            )
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class DataCollatorForCompletionOnlyLM:
    """Pad `input_ids`, `attention_mask`, and completion-only `labels`."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(example["input_ids"]) for example in examples)
        input_ids = []
        attention_mask = []
        labels = []

        for example in examples:
            pad_len = max_len - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(example["attention_mask"] + [0] * pad_len)
            labels.append(example["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def save_tokenized_split(path: str | Path, examples: Sequence[dict[str, Any]]) -> None:
    """Save tokenized examples as JSONL for reproducibility."""
    serializable = []
    for example in examples:
        serializable.append(
            {
                key: value
                for key, value in example.items()
                if key not in {"attention_mask"} or isinstance(value, list)
            }
        )
    write_jsonl(path, serializable)


def raw_split_path(config: dict[str, Any], split: str) -> Path:
    """Return the configured raw E2E split path."""
    raw_dir = resolve_path(config, config["data"]["raw_dir"])
    for suffix in (".jsonl", ".txt"):
        candidate = raw_dir / f"{split}{suffix}"
        if candidate.exists():
            return candidate
    return raw_dir / f"{split}.txt"
