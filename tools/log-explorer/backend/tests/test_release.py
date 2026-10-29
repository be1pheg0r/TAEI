from __future__ import annotations

import io
import json
import random
import stat
import zipfile

import app as explorer
from fastapi.testclient import TestClient

client = TestClient(explorer.app)


def make_zip(files: dict[str, str], *, symlink: str | None = None) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            if name == symlink:
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, text)
    return stream.getvalue()


def upload(blob: bytes, filename: str = "test.zip"):
    return client.post(
        "/api/analyze",
        files={"file": (filename, blob, "application/zip")},
    )


def test_valid_archive_api_preserves_large_measurements_and_identifiers():
    blob = make_zip(
        {
            "results.json": json.dumps(
                {"mode": "run", "logical_prompt_tokens": 9007199254740993}
            ),
            "requests.csv": ("request_id,input_tokens\n000123,9007199254740993\n"),
        }
    )

    response = upload(blob, "measurements.zip")

    assert response.status_code == 200
    data = response.json()
    assert data["archive"]["name"] == "measurements.zip"
    assert data["metrics"]["logical_prompt_tokens"] == 9007199254740993
    assert data["tables"]["requests"] == [
        {
            "source": "requests.csv",
            "request_id": "000123",
            "input_tokens": 9007199254740993,
        }
    ]
    assert data["summary"]["record_count"] >= 1


def test_unsafe_or_complex_archives_fail_closed_with_specific_reasons():
    control = upload(make_zip({"steps.json": '{"step_id":"safe"}'}))
    assert control.status_code == 200
    assert control.json()["records"]

    attacks = (
        (make_zip({"../escape.json": "{}"}), "Unsafe or duplicate ZIP path"),
        (
            make_zip({"link.json": "target", "target": "{}"}, symlink="link.json"),
            "Unsafe or duplicate ZIP path",
        ),
        (
            make_zip({"compressed.log": "x" * 1_000_000}),
            "compression ratio exceeds limit",
        ),
        (
            make_zip({"deep.json": '{"x":' + "[" * 70 + "0" + "]" * 70 + "}"}),
            "JSON nesting exceeds parser limit",
        ),
    )
    for blob, expected_detail in attacks:
        response = upload(blob)
        assert response.status_code == 400
        assert expected_detail in response.json()["detail"]


def test_oversized_request_is_rejected_before_spooling_or_archive_parsing(monkeypatch):
    monkeypatch.setattr(explorer, "MAX_REQUEST", 512)
    monkeypatch.setattr(
        explorer,
        "_analyze_path",
        lambda *_args: (_ for _ in ()).throw(AssertionError("archive parser ran")),
    )
    monkeypatch.setattr(
        explorer.tempfile,
        "NamedTemporaryFile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("upload was spooled")
        ),
    )

    generator = random.Random(17)
    payload = "".join(generator.choices("abcdefghijklmnopqrstuvwxyz", k=8_000))
    response = upload(make_zip({"payload.log": payload}))

    assert response.status_code == 413
    assert "Request body exceeds" in response.text
