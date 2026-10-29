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
        "metrics_are_preview": False,
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


def test_current_runtime_log_contract_parses_lifecycle_and_compact_metrics():
    log = (
        "INFO     | Starting experiment: nano_vllm_dataset_rtx_a4000_tp2\n"
        "INFO     | metrics: mode=plan, run_count=48, request_count=768, \n"
        "logical_prompt_tokens=1774040, requested_output_tokens=90240, +2 more\n"
        "INFO     | Finished experiment: nano_vllm_dataset_rtx_a4000_tp2 · "
        "results: /run/results.json\n"
    )
    result = _analyze(make_zip({"run.log": log}), "runtime-logs.zip")
    assert result["summary"]["experiment"] == "nano_vllm_dataset_rtx_a4000_tp2"
    assert result["summary"]["status"] == "finished"
    assert result["metrics"] == {
        "mode": "plan",
        "run_count": 48,
        "request_count": 768,
        "logical_prompt_tokens": 1774040,
        "requested_output_tokens": 90240,
    }
    assert result["summary"]["metrics_are_preview"] is True
    assert "Краткая сводка" in next(
        record["description"]
        for record in result["records"]
        if record["kind"] == "metric-summary"
    )
    assert result["runs"] == []  # Compact summaries do not contain attempt records.


def test_results_json_is_authoritative_when_included_in_archive():
    result = _analyze(
        make_zip(
            {
                "run.log": "INFO     | metrics: mode=plan, run_count=2, +1 more\n",
                "results.json": json.dumps(
                    {"mode": "plan", "run_count": 2, "request_count": 12}
                ),
            }
        ),
        "with-results.zip",
    )
    assert result["metrics"] == {"mode": "plan", "run_count": 2, "request_count": 12}
    assert result["summary"]["metrics_are_preview"] is False


def test_compact_metrics_preview_and_scalar_types_are_not_guessed():
    result = _analyze(
        make_zip(
            {
                "run.log": (
                    "INFO | metrics: mode=run, enabled=True, disabled=False, "
                    "optional=None, attempt_count=48\n"
                )
            }
        ),
        "compact.zip",
    )
    assert result["summary"]["metrics_are_preview"] is True
    assert result["metrics"] == {
        "mode": "run",
        "enabled": True,
        "disabled": False,
        "optional": None,
        "attempt_count": 48,
    }


def test_results_after_record_preview_still_supply_full_runs_and_metrics():
    runs = [
        {
            "run_id": f"R{i:02}",
            "attempt_id": f"attempt-{i}",
            "primary_status": "accepted" if i == 0 else "not_evaluated",
            "qc": {"status": "disabled", "valid": None},
        }
        for i in range(48)
    ]
    result = _analyze(
        make_zip(
            {
                "steps.jsonl": "".join(
                    json.dumps({"step_id": f"s{i}"}) + "\n" for i in range(501)
                ),
                "results.json": json.dumps(
                    {
                        "mode": "run",
                        "attempt_count": 48,
                        "accepted_primary_count": 1,
                        "runs": runs,
                    }
                ),
            }
        ),
        "late-results.zip",
    )
    assert result["summary"]["record_preview_count"] == 500
    assert result["summary"]["record_count"] == 502
    assert result["metrics"]["accepted_primary_count"] == 1
    assert len(result["runs"]) == 48
    assert result["summary"]["attempt_count"] == 48
    assert result["summary"]["accepted_count"] == 1
    assert len(result["tables"]["runs"]) == 48
    assert result["tables"]["_meta"]["counts"]["runs"] == 48
    assert result["tables"]["_meta"]["preview_rows"]["runs"] == 48
    assert result["tables"]["runs"][0]["source"] == "results.json"


def test_compact_metrics_after_record_preview_are_still_aggregated():
    result = _analyze(
        make_zip(
            {
                "steps.jsonl": "".join(
                    json.dumps({"step_id": f"s{i}"}) + "\n" for i in range(501)
                ),
                "run.log": "INFO | metrics: mode=run, attempt_count=48\n",
            }
        ),
        "late-compact.zip",
    )
    assert result["summary"]["record_preview_count"] == 500
    assert result["metrics"] == {"mode": "run", "attempt_count": 48}
    assert result["summary"]["metrics_are_preview"] is True


def test_runtime_lifecycle_after_record_preview_still_sets_final_status():
    log = "INFO | Starting experiment: long_run\n"
    log += "INFO | working\n" * 501
    log += "INFO | metrics: attempt_count=48\n"
    log += "INFO | Finished experiment: long_run · results: /run/results.json\n"
    result = _analyze(make_zip({"run.log": log}), "long-runtime.zip")
    assert result["summary"]["record_preview_count"] == 500
    assert result["summary"]["experiment"] == "long_run"
    assert result["summary"]["status"] == "finished"


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
