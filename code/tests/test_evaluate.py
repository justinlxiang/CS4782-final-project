from __future__ import annotations

from lora_gpt2.data import E2ERecord
from lora_gpt2.evaluate import (
    group_e2e_predictions_and_references,
    read_prediction_records,
    read_prediction_texts,
    write_official_e2e_files,
)


def test_read_prediction_texts_supports_jsonl(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"prediction": "The Eagle is good."}\n', encoding="utf-8")

    assert read_prediction_texts(path) == ["The Eagle is good."]


def test_group_e2e_predictions_matches_official_decode_grouping() -> None:
    records = [
        E2ERecord(context="name : Blue Spice", completion="Blue Spice is good."),
        E2ERecord(context="name : Blue Spice", completion="Blue Spice is nice."),
        E2ERecord(context="name : The Eagle", completion="The Eagle is a pub."),
    ]
    predictions = [
        {"id": "0", "prompt": "name : Blue Spice", "prediction": "Blue Spice is a cafe."},
        {"id": "1", "prompt": "name : Blue Spice", "prediction": "Blue Spice is a cafe."},
        {"id": "2", "prompt": "name : The Eagle", "prediction": "The Eagle is a pub."},
    ]

    grouped_predictions, grouped_references, grouped_contexts = (
        group_e2e_predictions_and_references(records, predictions)
    )

    assert grouped_contexts == ["name : Blue Spice", "name : The Eagle"]
    assert grouped_predictions == ["Blue Spice is a cafe.", "The Eagle is a pub."]
    assert grouped_references == [
        ["Blue Spice is good.", "Blue Spice is nice."],
        ["The Eagle is a pub."],
    ]


def test_write_official_e2e_files_uses_blank_line_reference_groups(tmp_path) -> None:
    ref_path = tmp_path / "e2e_ref.txt"
    pred_path = tmp_path / "e2e_pred.txt"

    write_official_e2e_files(
        ref_path,
        pred_path,
        predictions=["prediction one", "prediction two"],
        references=[["ref one a", "ref one b"], ["ref two"]],
    )

    assert ref_path.read_text(encoding="utf-8") == "ref one a\nref one b\n\nref two\n\n"
    assert pred_path.read_text(encoding="utf-8") == "prediction one\nprediction two\n"


def test_read_prediction_records_supports_plain_text(tmp_path) -> None:
    path = tmp_path / "predictions.txt"
    path.write_text("first\nsecond\n", encoding="utf-8")

    assert read_prediction_records(path) == [
        {"id": "0", "prompt": "", "prediction": "first"},
        {"id": "1", "prompt": "", "prediction": "second"},
    ]
