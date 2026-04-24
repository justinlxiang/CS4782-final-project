from __future__ import annotations

from lora_gpt2.evaluate import read_prediction_texts


def test_read_prediction_texts_supports_jsonl(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"prediction": "The Eagle is good."}\n', encoding="utf-8")

    assert read_prediction_texts(path) == ["The Eagle is good."]
