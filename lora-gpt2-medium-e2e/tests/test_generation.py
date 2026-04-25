from __future__ import annotations

from types import SimpleNamespace

import torch

from lora_gpt2.generate import (
    _eos_token_ids,
    encode_generation_prompts,
    extract_continuation,
    format_predictions,
    official_style_beam_search_continuations,
)


class ToyTokenizer:
    eos_token_id = 0
    pad_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) % 89 + 1 for char in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        mapping = {1: " first", 2: "<eos>"}
        return "".join(mapping.get(int(token_id), "") for token_id in token_ids)


class TinyBeamModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=4)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
    ):
        del attention_mask, position_ids, use_cache
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, self.config.vocab_size), -1000.0)
        if past_key_values is None:
            logits[:, -1, 1] = 0.0
            logits[:, -1, 2] = -1.0
        else:
            logits[:, -1, 2] = 0.0
            logits[:, -1, 1] = -1.0
        total_len = seq_len if past_key_values is None else past_key_values[0][0].size(2) + seq_len
        key = torch.zeros(batch_size, 1, total_len, 1)
        value = torch.zeros(batch_size, 1, total_len, 1)
        return SimpleNamespace(logits=logits, past_key_values=((key, value),))


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


def test_eos_token_ids_accepts_single_or_multiple_ids() -> None:
    assert _eos_token_ids(628) == [628]
    assert _eos_token_ids([50256, 628]) == [50256, 628]


def test_official_style_beam_search_stops_at_eos() -> None:
    config = {
        "data": {"append_bos_to_context": True},
        "generation": {
            "max_new_tokens": 4,
            "num_beams": 2,
            "length_penalty": 0.8,
            "no_repeat_ngram_size": 4,
            "repetition_penalty": 1.0,
            "eos_token_id": [2],
        },
    }

    outputs = official_style_beam_search_continuations(
        TinyBeamModel(),
        ToyTokenizer(),
        ["prompt"],
        config,
        torch.device("cpu"),
    )

    assert outputs == ["first"]
