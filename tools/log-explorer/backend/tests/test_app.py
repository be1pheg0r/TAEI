from __future__ import annotations

import io
import json
import zipfile
from unittest.mock import patch

import pytest
from app import _analyze, app
from fastapi.testclient import TestClient

client = TestClient(app)


def make_zip(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return stream.getvalue()


def test_sample_round_trip_and_metadata():
    response = client.get("/api/example")
    assert response.status_code == 200
    body = response.json()
    assert body["archive"]["name"] == "sample.zip"
    assert any(
        file["name"] == "hydra.log" and file["size"] == 0
        for file in body["archive"]["files"]
    )
    assert body["summary"] == {
        "experiment": "nano_vllm_dataset_rtx_a4000_tp2",
        "status": "finished",
        "run_count": 1,
        "accepted_count": 0,
        "record_count": len(body["records"]),
        "attempt_count": 1,
        "record_preview_count": len(body["records"]),
        "record_preview_limit": 500,
        "records_are_preview": False,
        "warnings": [],
    }
    assert body["runs"][0]["run_id"]
    assert body["tables"]["runs"][0]["run_id"] == body["runs"][0]["run_id"]
    assert all(
        record["description"] and "raw_message" in record for record in body["records"]
    )
    assert all(field["description"] for field in body["fields"])
    runs = body["runs"][0]
    assert runs["effective_config"]["engine_revision"]
    assert runs["real_generate_completed"] is True


def test_structured_trace_tables_and_csv_types():
    files = {
        "nested/steps.jsonl": (
            '{"step_id":"s1","step_index":0,"contexts_before":{},"request_ids":[]}\n'
        ),
        "requests.csv": (
            "request_id,actual_output_tokens,temperature,ignore_eos,note\n"
            "r1,5,0.5,true,\n"
        ),
        "nested/telemetry.json": json.dumps([{"interval_s": 1.5, "field_status": {}}]),
        "runs.json": json.dumps([{"run_id": "r", "primary_status": "pending"}]),
    }
    result = _analyze(make_zip(files), "traces.zip")
    assert result["tables"]["steps"][0]["step_id"] == "s1"
    assert result["tables"]["steps"][0]["contexts_before"] == {}
    assert result["tables"]["requests"][0]["actual_output_tokens"] == 5
    assert result["tables"]["requests"][0]["temperature"] == 0.5
    assert result["tables"]["requests"][0]["ignore_eos"] is True
    assert result["tables"]["requests"][0]["note"] is None
    assert result["tables"]["telemetry"][0]["interval_s"] == 1.5
    assert result["runs"] == [{"run_id": "r", "primary_status": "pending"}]


def test_repeated_runs_metrics_accumulate():
    log = (
        "INFO | metric runs=[{'run_id': 'a'}, {'run_id': 'b'}]\n"
        "INFO | metric runs=[{'run_id': 'c'}]\n"
    )
    result = _analyze(make_zip({"run.log": log}), "runs.zip")
    assert [run["run_id"] for run in result["runs"]] == ["a", "b", "c"]
    assert len(result["tables"]["runs"]) == 3


def test_nonfinite_and_deep_json_rejected():
    for payload in ('{"x":NaN}', '{"x":' + "[" * 70 + "0" + "]" * 70 + "}"):
        with pytest.raises(ValueError):
            _analyze(make_zip({"steps.json": payload}), "bad.zip")


def test_zip_path_duplicates_and_bomb_rejected():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("a.json", "{}")
        archive.writestr("A.JSON", "{}")
    with pytest.raises(ValueError, match="duplicate"):
        _analyze(stream.getvalue(), "duplicate.zip")
    with pytest.raises(ValueError, match="ratio"):
        _analyze(make_zip({"large.log": "x" * (1024 * 1024)}), "bomb.zip")


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_unbounded_request_body_is_not_rejected_by_old_26mib_guard():
    with patch("starlette.requests.Request.form", side_effect=AssertionError("parsed")):
        response = client.post(
            "/api/analyze",
            content=b"not multipart",
            headers={
                "content-type": "multipart/form-data; boundary=x",
                "content-length": str(26 * 1024 * 1024 + 1),
            },
        )
    assert response.status_code != 413


def test_analyze_accepts_valid_upload():
    response = client.post(
        "/api/analyze",
        files={"file": ("ok.zip", make_zip({"a.log": "INFO | ok"}), "application/zip")},
    )
    assert response.status_code == 200
    assert response.json()["archive"]["files"][0]["name"] == "a.log"


def test_large_archive_limits_are_not_fixed_upload_or_member_caps():
    import app as module

    assert module.MAX_UPLOAD is None
    assert module.MAX_MEMBER is None
    assert module.MAX_EXPANDED is None
    assert module.MAX_FILES is None


def test_encrypted_rar_members_are_rejected(tmp_path, monkeypatch):
    import app as module
    import rarfile

    class Info:
        filename = "secret.json"
        file_size = 1
        compress_size = 1
        mode = 0
        file_redir = None

        def is_dir(self):
            return False

        def is_symlink(self):
            return False

        def needs_password(self):
            return True

    class Archive:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def infolist(self):
            return [Info()]

    monkeypatch.setattr(rarfile, "RarFile", Archive)
    archive_path = tmp_path / "encrypted.rar"
    archive_path.write_bytes(b"Rar!\x1a\x07\x00")
    with pytest.raises(ValueError, match="Encrypted RAR"):
        module._analyze_path(archive_path, "encrypted.rar")


def test_rar_magic_is_detected_and_read_from_disk(tmp_path):
    import app as module
    from libarchive.exception import ArchiveError

    archive_path = tmp_path / "probe.rar"
    archive_path.write_bytes(b"Rar!\x1a\x07\x00")
    with pytest.raises(ArchiveError):
        module._analyze_path(archive_path, "probe.rar")


def test_large_row_sets_report_exact_totals_and_bounded_previews():
    rows = "".join(json.dumps({"step_id": f"s{i}"}) + "\n" for i in range(700))
    result = _analyze(make_zip({"steps.jsonl": rows}), "many.zip")
    assert len(result["records"]) == 500
    assert result["summary"]["record_count"] == 700
    assert result["archive"]["files"][0]["records"] == 700
    assert result["tables"]["_meta"]["counts"]["steps"] == 700
    assert len(result["tables"]["steps"]) == 500
    assert result["tables"]["_meta"]["preview_limit"] == 500
    assert result["tables"]["_meta"]["is_preview"] is True
