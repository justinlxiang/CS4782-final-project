from __future__ import annotations

from lora_gpt2.generate import extract_continuation, format_predictions


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
