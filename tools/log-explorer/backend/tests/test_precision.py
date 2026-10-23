import io
import zipfile

import pytest
from app import _analyze, _parse_literal


def archive(name, content):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as handle:
        handle.writestr(name, content)
    return output.getvalue()


def test_csv_identifiers_and_large_integers_are_exact():
    data = _analyze(
        archive("requests.csv", "request_id,input_tokens\n000123,9007199254740993\n"),
        "input.zip",
    )
    row = data["tables"]["requests"][0]
    assert row["request_id"] == "000123"
    assert row["input_tokens"] == 9007199254740993


@pytest.mark.parametrize("text", ['{"value": 1e999}', '{"value": NaN}'])
def test_json_overflow_is_not_a_measurement(text):
    with pytest.raises(ValueError):
        _analyze(archive("steps.json", text), "input.zip")


def test_literal_complexity_guard_cannot_fall_back_to_unbounded_eval():
    with pytest.raises(ValueError, match="complexity"):
        _parse_literal("[" * 70 + "0" + "]" * 70)
    with pytest.raises(ValueError, match="Non-finite"):
        _parse_literal("{'value': 1e999}")


def test_known_config_and_records_have_semantic_descriptions():
    from app import BASE

    data = _analyze((BASE / "sample.zip").read_bytes(), "sample.zip")
    field = next(
        f for f in data["fields"] if f["path"].endswith("tensor_parallel_size")
    )
    assert "параллелизма" in field["description"]
    assert any("приёмка QC" in r["description"] for r in data["records"])
