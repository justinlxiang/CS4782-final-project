from __future__ import annotations

from lora_gpt2.data import (
    DataCollatorForCompletionOnlyLM,
    E2ERecord,
    E2ETokenizedDataset,
    parse_e2e_line,
    tokenize_record,
)


class ToyTokenizer:
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) % 89 + 1 for char in text]


def test_parse_e2e_context_completion_line() -> None:
    record = parse_e2e_line("name[The Eagle]||The Eagle is a pub.")

    assert record.context == "name[The Eagle]"
    assert record.completion == "The Eagle is a pub."


def test_tokenize_masks_prompt_labels() -> None:
    tokenizer = ToyTokenizer()
    example = tokenize_record(
        E2ERecord("name[The Eagle]", "The Eagle is a pub."),
        tokenizer=tokenizer,
        prompt_template="MR: {mr}\nText:",
        max_length=128,
    )

    prompt_length = example["prompt_length"]
    assert all(label == -100 for label in example["labels"][:prompt_length])
    assert example["labels"][-1] == tokenizer.eos_token_id
    assert len(example["input_ids"]) == len(example["labels"])


def test_collator_pads_labels_with_ignore_index() -> None:
    tokenizer = ToyTokenizer()
    dataset = E2ETokenizedDataset(
        [
            E2ERecord("name[A]", "A is good."),
            E2ERecord("name[Longer Place]", "Longer Place is good."),
        ],
        tokenizer=tokenizer,
        prompt_template="MR: {mr}\nText:",
        max_length=128,
    )
    batch = DataCollatorForCompletionOnlyLM(pad_token_id=0)([dataset[0], dataset[1]])

    assert batch["input_ids"].shape == batch["labels"].shape
    assert (batch["labels"] == -100).any()
