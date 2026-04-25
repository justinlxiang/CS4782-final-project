from __future__ import annotations

from lora_gpt2.generate import encode_generation_prompts, extract_continuation, format_predictions


class ToyTokenizer:
    eos_token_id = 0
    pad_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) % 89 + 1 for char in text]


def test_extract_continuation_removes_prompt_and_stop_markers() -> None:
    prompt = "Generate:\n"
    full_text = "Generate:\nThe Eagle is a pub.<|endoftext|> extra"

    assert extract_continuation(full_text, prompt) == "The Eagle is a pub."


def test_format_predictions_uses_stable_ids() -> None:
    predictions = format_predictions(["p1", "p2"], ["g1", "g2"])

    assert predictions == [
        {"id": "0", "prompt": "p1", "prediction": "g1"},
        {"id": "1", "prompt": "p2", "prediction": "g2"},
    ]


def test_encode_generation_prompts_can_append_bos_separator() -> None:
    encoded = encode_generation_prompts(
        ToyTokenizer(),
        ["name : Blue Spice"],
        append_bos_to_context=True,
    )

    assert encoded["input_ids"][0, -1].item() == ToyTokenizer.eos_token_id
    assert encoded["attention_mask"][0, -1].item() == 1
