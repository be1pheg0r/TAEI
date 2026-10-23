"""Public parser rejection paths; fixtures are intentionally synthetic."""

import io
import os
import zipfile

import pytest
import requests

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8091").rstrip("/")


def zipped(entries, compressed=False):
    output = io.BytesIO()
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(output, "w", compression=compression) as handle:
        for name, value in entries:
            handle.writestr(name, value)
    return output.getvalue()


@pytest.mark.parametrize(
    "content",
    [
        b"not a ZIP",
        zipped([("../unsafe.log", "INFO | unsafe")]),
        zipped([("/absolute.log", "INFO | unsafe")]),
        zipped([("steps.json", '{"duration": NaN}')]),
        zipped([("steps.json", '{"duration": 1e999}')]),
        zipped([("steps.json", '{"x":1,"x":2}')]),
        zipped([("large.log", "0" * 100_000)], compressed=True),
    ],
    ids=[
        "invalid-zip",
        "traversal",
        "absolute-path",
        "nan",
        "overflow",
        "duplicate-json-key",
        "compression-ratio",
    ],
)
def test_rejected_archive_is_recoverable(content):
    response = requests.post(
        BASE + "/api/analyze",
        files={"file": ("unsafe.zip", content, "application/zip")},
        timeout=30,
    )
    assert response.status_code == 400, response.text
    assert response.json().get("detail")
    assert requests.get(BASE + "/api/health", timeout=10).json()["status"] == "ok"


def test_python_expressions_are_not_executed():
    content = zipped(
        [
            (
                "run.log",
                "INFO | metric suspicious=__import__('pathlib')"
                ".Path('/tmp/not-a-real-experiment').read_text()\n",
            )
        ]
    )
    response = requests.post(
        BASE + "/api/analyze",
        files={"file": ("expression.zip", content, "application/zip")},
        timeout=30,
    )
    assert response.ok, response.text
    value = response.json()["metrics"]["suspicious"]
    assert isinstance(value, str) and "__import__" in value
