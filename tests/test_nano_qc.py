"""Проверяет fail-closed аудит артефактов и связей трасс nano-vLLM.

Структурные фикстуры не считаются свидетельством GPU-инференса. Тесты
проверяют хэши и безопасность путей манифеста, схему и калибровочные барьеры,
совпадение запросов/результатов/токенов, уникальность идентификаторов,
монотонность времени, сохранение объема работы, рост контекста запросов и
завершение фиксированных волн. Поврежденные данные должны безопасно отклоняться.
"""

import hashlib
import json

from taei.nano_dataset.qc import audit_run_directory
from taei.nano_dataset.workload import build_workload


def _write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _valid_attempt(root):
    """A truthful structural fixture; not evidence of real GPU inference."""
    workload = build_workload("S", 42)
    ids = {
        "hardware_id": "H1",
        "environment_id": "E1",
        "model_id": "M1",
        "workload_id": workload["workload_id"],
    }
    for name, identifier in (
        ("hardware", "H1"),
        ("environment", "E1"),
        ("model", "M1"),
    ):
        _write_json(
            root / f"{name}.json", {f"{name}_id": identifier, "field_status": {}}
        )
    _write_json(root / "workload.json", workload)
    requests = []
    for request in workload["requests"]:
        requests.append(
            {
                **request,
                "output_token_ids": [11] * request["requested_output_tokens"],
                "actual_output_tokens": request["requested_output_tokens"],
                "output_token_count": request["requested_output_tokens"],
                "status": "completed",
                "run_id": "R01",
                "attempt_id": "A1",
                **ids,
                "config_id": "CFG1",
                "field_status": {
                    name: {"status": "not_applicable", "reason": "unique prefix"}
                    for name in ("prefix_family_id", "prefix_group_id")
                },
            }
        )
    (root / "requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in requests), encoding="utf-8"
    )
    steps, links, tokens, sched = [], [], [], []

    def stamp(ns):
        return {
            "timestamp_utc": "2026-10-23T10:00:00+00:00",
            "monotonic_ns": ns,
            "clock_id": "synthetic-test-clock",
            "process_id": 99,
        }

    for request in requests:
        sched.append(
            {
                **stamp(100),
                "event": "enqueue",
                "event_type": "admit",
                "request_id": request["request_id"],
            }
        )
    for wave in range(4):
        members = requests[wave * 8 : (wave + 1) * 8]
        member_ids = [request["request_id"] for request in members]
        for token_index in range(128):
            index = len(steps)
            sid = str(index + 1)
            start = 10000 + index * 1000
            prefill = token_index == 0
            scheduled_each = 128 if prefill else 1
            steps.append(
                {
                    **stamp(start + 900),
                    "step_id": sid,
                    "step_index": index,
                    "phase": "prefill" if prefill else "decode",
                    "request_ids": member_ids,
                    "step_host_start_ns": start,
                    "step_host_end_ns": start + 900,
                    "actual_batch_size": 8,
                    "produced_tokens": 8,
                    "scheduled_tokens": scheduled_each * 8,
                    "fresh_prefill_tokens": 1024 if prefill else 0,
                    "recompute_tokens": 0,
                    "prefix_reused_tokens": 0,
                    "wave_id": str(wave),
                }
            )
            for sequence_index, request in enumerate(members):
                links.append(
                    {
                        **stamp(start + 800),
                        "step_id": sid,
                        "request_id": request["request_id"],
                        "relation": "scheduled",
                        "sequence_index": sequence_index,
                        "context_tokens_before": 128 + token_index,
                        "context_tokens_after": 129 + token_index,
                        "scheduled_tokens": scheduled_each,
                        "produced_tokens": 1,
                        "fresh_prefill_tokens": 128 if prefill else 0,
                        "recompute_tokens": 0,
                        "prefix_reused_tokens": 0,
                    }
                )
                tokens.append(
                    {
                        **stamp(start + 700),
                        "step_id": sid,
                        "request_id": request["request_id"],
                        "token_index": token_index,
                        "token_id": 11,
                        "token_type": "sampled_output",
                    }
                )
            sched.append(
                {
                    **stamp(start + 100),
                    "step_id": sid,
                    "event": "schedule",
                    "event_type": "schedule",
                    "admitted_request_ids": member_ids if prefill else [],
                    "request_ids": member_ids,
                    "wave_id": str(wave),
                }
            )
    for filename, rows in (
        ("steps.jsonl", steps),
        ("step_requests.jsonl", links),
        ("token_events.jsonl", tokens),
        ("scheduler_events.jsonl", sched),
        (
            "cache_events.jsonl",
            [{**stamp(1000), "operation": "allocate", "prefix_cache_mode": "on"}],
        ),
        ("telemetry.jsonl", []),
    ):
        (root / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    _write_json(
        root / "runs.json",
        {
            "run_id": "R01",
            "attempt_id": "A1",
            "status": "success",
            "primary_status": "accepted",
            "config_id": "CFG1",
            "condition_id": "C01",
            "pair_id": "P1",
            "repeat_id": "01",
            "run_order": 1,
            "prefix_cache_mode": "on",
            "scheduler_mode": "fixed_batch",
            **ids,
            "metrics": {"request_count": 32, "output_tokens": 4096},
            "timing_policy": {"primary_analysis_gate": "pending_calibration"},
        },
    )
    files = {}
    for p in root.iterdir():
        if p.name != "manifest.json":
            raw = p.read_bytes()
            files[p.name] = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "1.0.0",
            "collector_version": "test",
            "run_id": "R01",
            "attempt_id": "A1",
            **ids,
            "config_id": "CFG1",
            "files": files,
            "coverage": {
                "primary_complete": True,
                "missing_hooks": [],
                "capabilities": {"real_nano_vllm": True},
            },
            "hash_policy": "SHA-256 exact bytes",
        },
    )


def _refresh_hashes(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        raw = (root / name).read_bytes()
        record.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    _write_json(root / "manifest.json", manifest)


def test_audit_detects_hash_tamper_and_orphan_request(tmp_path):
    (tmp_path / "requests.jsonl").write_text(json.dumps({"request_id": "Q1"}) + "\n")
    (tmp_path / "steps.jsonl").write_text(
        json.dumps({"step_id": "S1", "request_ids": ["Q2"]}) + "\n"
    )
    (tmp_path / "step_requests.jsonl").write_text("\n")
    (tmp_path / "cache_events.jsonl").write_text("\n")
    (tmp_path / "scheduler_events.jsonl").write_text("\n")
    (tmp_path / "token_events.jsonl").write_text("\n")
    (tmp_path / "telemetry.jsonl").write_text("\n")
    expected = hashlib.sha256((tmp_path / "requests.jsonl").read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": {"requests.jsonl": {"sha256": expected}}})
    )
    report = audit_run_directory(tmp_path)
    assert any("unknown request Q2" in error for error in report["errors"])
    (tmp_path / "requests.jsonl").write_text("tampered\n")
    report = audit_run_directory(tmp_path)
    assert any("hash mismatch" in error for error in report["errors"])


def test_missing_primary_hooks_cannot_be_accepted(tmp_path):
    report = audit_run_directory(tmp_path)
    assert report["primary_status"] == "rejected"
    assert report["accepted"] is False


def test_success_and_nonempty_hooks_do_not_bypass_pending_calibration(tmp_path):
    _valid_attempt(tmp_path)
    report = audit_run_directory(tmp_path)
    assert report["accepted"] is False
    assert report["primary_status"] == "rejected"
    assert any("calibration" in error.lower() for error in report["errors"])


def test_producer_shaped_trace_joins_emit_contract_and_reaches_only_calibration_gate(
    tmp_path,
):
    """Producer event names/fields survive persistence and QC joins."""
    _valid_attempt(tmp_path)
    token_path = tmp_path / "token_events.jsonl"
    tokens = [json.loads(line) for line in token_path.read_text().splitlines()]
    for row in tokens:
        row["token_type"] = "sampled_output"
        row["token_origin"] = "sampled_output"
    token_path.write_text("".join(json.dumps(row) + "\n" for row in tokens))
    sched_path = tmp_path / "scheduler_events.jsonl"
    schedule = [json.loads(line) for line in sched_path.read_text().splitlines()]
    for row in schedule:
        if row.get("event_type") == "schedule":
            row.pop("event", None)
            row["step_id"] = str(row["step_id"])
    sched_path.write_text("".join(json.dumps(row) + "\n" for row in schedule))
    _refresh_hashes(tmp_path)

    report = audit_run_directory(tmp_path)
    assert report["accepted"] is False
    assert report["errors"] == [
        "overhead calibration is pending or unverified; primary acceptance is gated",
    ]


def test_schedule_decision_requires_known_step_and_exact_request_join(tmp_path):
    _valid_attempt(tmp_path)
    path = tmp_path / "scheduler_events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    event = next(row for row in rows if row.get("event_type") == "schedule")
    event["step_id"] = "missing-step"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_hashes(tmp_path)
    report = audit_run_directory(tmp_path)
    assert any("unknown or missing step_id" in error for error in report["errors"])
    assert any("no matching schedule decision" in error for error in report["errors"])


def test_prefix_off_capacity_check_is_distinct_from_real_lookup(tmp_path):
    _valid_attempt(tmp_path)
    run_path = tmp_path / "runs.json"
    run = json.loads(run_path.read_text())
    run["prefix_cache_mode"] = "off"
    _write_json(run_path, run)
    path = tmp_path / "cache_events.jsonl"
    rows = [
        {
            "operation": "can_allocate",
            "event_type": "capacity_check",
            "prefix_cache_mode": "off",
        }
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_hashes(tmp_path)
    report = audit_run_directory(tmp_path)
    assert not any("prefix lookup/hash" in error for error in report["errors"])

    rows[0]["event_type"] = "lookup"
    path.write_text(json.dumps(rows[0]) + "\n")
    _refresh_hashes(tmp_path)
    report = audit_run_directory(tmp_path)
    assert any("prefix lookup/hash" in error for error in report["errors"])


def test_coherent_synthetic_fixed_wave_attempt_is_accepted(tmp_path):
    """Synthetic structure fixture only; it is not GPU execution evidence."""
    _valid_attempt(tmp_path)
    run = json.loads((tmp_path / "runs.json").read_text())
    run["timing_policy"]["primary_analysis_gate"] = "calibrated"
    run["scheduler_mode"] = "fixed_batch"
    run["prefix_cache_mode"] = "on"
    _write_json(tmp_path / "runs.json", run)
    _refresh_hashes(tmp_path)

    report = audit_run_directory(tmp_path)
    assert report["accepted"] is True, report["errors"]


def test_malformed_unhashable_ids_fail_closed_without_raising(tmp_path):
    _valid_attempt(tmp_path)
    steps_path = tmp_path / "steps.jsonl"
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    steps[0]["step_id"] = ["not-hashable"]
    steps[0]["request_ids"] = [["not-hashable"]]
    steps[0]["clock_id"] = {"not": "hashable"}
    steps_path.write_text("".join(json.dumps(row) + "\n" for row in steps))
    _refresh_hashes(tmp_path)

    report = audit_run_directory(tmp_path)
    assert report["accepted"] is False
    assert any("step_id" in error or "clock_id" in error for error in report["errors"])


def test_manifest_path_traversal_is_rejected(tmp_path):
    _valid_attempt(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"]["../outside.json"] = {"sha256": "0" * 64, "bytes": 0}
    _write_json(tmp_path / "manifest.json", manifest)
    report = audit_run_directory(tmp_path)
    assert not report["accepted"]
    assert any(
        "unsafe" in error.lower() or "path" in error.lower()
        for error in report["errors"]
    )


def test_successful_attempt_requires_exact_request_output_and_token_events(tmp_path):
    _valid_attempt(tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl").read_text().splitlines()
    ]
    rows[0]["output_token_ids"] = []
    (tmp_path / "requests.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    report = audit_run_directory(tmp_path)
    assert not report["accepted"]
    assert any(
        "output" in e.lower() or "hash mismatch" in e.lower() for e in report["errors"]
    )


def test_exact_hash_tamper_is_detected(tmp_path):
    _valid_attempt(tmp_path)
    with (tmp_path / "steps.jsonl").open("a") as stream:
        stream.write("{}\n")
    report = audit_run_directory(tmp_path)
    assert any("hash mismatch: steps.jsonl" in error for error in report["errors"])


def test_duplicate_request_and_monotonic_inversion_are_rejected(tmp_path):
    _valid_attempt(tmp_path)
    req_path = tmp_path / "requests.jsonl"
    lines = req_path.read_text().splitlines()
    req_path.write_text("\n".join(lines + [lines[0]]) + "\n")
    steps_path = tmp_path / "steps.jsonl"
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    steps[1]["monotonic_ns"] = 0
    steps_path.write_text("".join(json.dumps(row) + "\n" for row in steps))
    _refresh_hashes(tmp_path)
    errors = audit_run_directory(tmp_path)["errors"]
    assert any("duplicate request_id" in error for error in errors)
    assert any("monotonic clock inversion" in error for error in errors)


def test_symlink_filemap_entry_cannot_escape_run_directory(tmp_path):
    _valid_attempt(tmp_path)
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}")
    (tmp_path / "external.json").symlink_to(outside)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"]["external.json"] = {
        "sha256": hashlib.sha256(b"{}").hexdigest(),
        "bytes": 2,
    }
    _write_json(tmp_path / "manifest.json", manifest)
    report = audit_run_directory(tmp_path)
    assert not report["accepted"]
    assert any("symlink" in error.lower() for error in report["errors"])


def _calibrated_attempt(root):
    _valid_attempt(root)
    run = json.loads((root / "runs.json").read_text())
    run["timing_policy"]["primary_analysis_gate"] = "calibrated"
    _write_json(root / "runs.json", run)
    _refresh_hashes(root)
    assert audit_run_directory(root)["accepted"]


def test_step_production_counts_must_match_actual_token_events(tmp_path):
    _calibrated_attempt(tmp_path)
    path = tmp_path / "steps.jsonl"
    steps = [json.loads(line) for line in path.read_text().splitlines()]
    steps[0]["produced_tokens"] = 0
    path.write_text("".join(json.dumps(row) + "\n" for row in steps))
    _refresh_hashes(tmp_path)
    report = audit_run_directory(tmp_path)
    assert not report["accepted"]
    assert any("produced_tokens" in error for error in report["errors"])


def test_request_context_growth_must_match_step_output_tokens(tmp_path):
    _calibrated_attempt(tmp_path)
    path = tmp_path / "step_requests.jsonl"
    links = [json.loads(line) for line in path.read_text().splitlines()]
    links[0]["context_tokens_after"] += 10
    path.write_text("".join(json.dumps(row) + "\n" for row in links))
    _refresh_hashes(tmp_path)
    report = audit_run_directory(tmp_path)
    assert not report["accepted"]
    assert any("context" in error for error in report["errors"])


def test_fixed_wave_cannot_change_before_prior_requests_complete(tmp_path):
    _calibrated_attempt(tmp_path)
    for filename in ("steps.jsonl", "scheduler_events.jsonl"):
        path = tmp_path / filename
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row.get("step_id") == "2":
                row["wave_id"] = "premature-wave"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_hashes(tmp_path)
    report = audit_run_directory(tmp_path)
    assert not report["accepted"]
    assert any("wave" in error for error in report["errors"])
