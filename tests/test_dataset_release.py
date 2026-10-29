"""Six CPU-only release regressions for the nano dataset contract."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from taei.nano_dataset import model_download
from taei.nano_dataset._artifacts import canonical_json, write_jsonl
from taei.nano_dataset.orchestrator import build_plan, run_attempt
from taei.nano_dataset.qc import _safe_member, audit_run_directory
from taei.nano_dataset.request_scaling import (
    COUNTS,
    PROTOCOL,
    build_scaling_plan,
    build_scaling_workload,
)
from taei.nano_dataset.schema import DATASET_SCHEMA, validate_instance
from taei.nano_dataset.workload import (
    BLOCK_SIZE,
    CONDITIONS,
    SEEDS,
    WORKLOAD_SETS,
    build_run_matrix,
    build_workload,
    validate_shared_prefix_boundary,
)


def test_release_matrix_is_deterministic_paired_and_work_conserving():
    first = build_run_matrix()
    assert first == build_run_matrix()
    assert len(first) == 48
    assert (
        len({row["run_id"] for row in first})
        == len({row["run_order"] for row in first})
        == 48
    )
    assert {row["seed"] for row in first} == set(SEEDS)
    assert {row["workload_id"] for row in first} == set(WORKLOAD_SETS)
    assert all(
        Counter(
            (r["prefix_cache_mode"], r["scheduler_mode"])
            for r in first
            if r["pair_id"] == pair
        )
        == Counter(CONDITIONS)
        for pair in {r["pair_id"] for r in first}
    )
    assert (
        sum(len(build_workload(r["workload_id"], r["seed"])["requests"]) for r in first)
        == 768
    )
    assert (
        sum(
            build_workload(r["workload_id"], r["seed"])["summary"][
                "logical_prompt_tokens"
            ]
            for r in first
        )
        == 1_774_040
    )
    assert (
        sum(
            build_workload(r["workload_id"], r["seed"])["summary"][
                "requested_output_token_total"
            ]
            for r in first
        )
        == 90_240
    )
    sample = build_workload(WORKLOAD_SETS[0], 42)
    assert sample == build_workload(WORKLOAD_SETS[0], 42)
    assert all(
        hit["prompt_length"] == miss["prompt_length"]
        and hit["input_token_ids"][: hit["prefix_token_count"]]
        == hit["prime_input_token_ids"][:-1]
        and miss["prefix_token_count"] == 0
        for hit, miss in zip(
            sample["requests"][::2], sample["requests"][1::2], strict=True
        )
    )


def test_scaling_and_long_context_planners_enforce_context_and_block_boundaries():
    plan = build_scaling_plan(7, 14_000, 8, 16_384)
    assert plan["workload_protocol"] == PROTOCOL and plan["run_count"] == 30
    assert plan["request_count"] == 186
    assert [r["request_count"] for r in plan["matrix"]] == [
        int(r["workload_id"].rsplit("n", 1)[1]) for r in plan["matrix"]
    ]
    for seed in (42, 43, 44):
        nested = [
            build_scaling_workload(f"qwen3_8b_request_scaling_n{n}", seed)["requests"]
            for n in COUNTS
        ]
        assert all(
            nested[i] == nested[-1][: len(nested[i])] for i in range(len(nested))
        )
        assert all(
            r["prefix_token_count"] % BLOCK_SIZE == 0
            and 0.1 <= r["target_cache_hit_ratio_on"] <= 0.95
            for r in nested[-1]
        )
    with pytest.raises(ValueError, match="14000"):
        build_scaling_plan(7, 13_999, 8, 16_384)
    assert validate_shared_prefix_boundary(256, 512, 513)["eligible_blocks"] == 2
    with pytest.raises(ValueError, match="below workload requirement"):
        build_plan(workload_protocol="qwen3_8b_long_context_32k_v1", context_cap=32_256)


def test_attempt_lifecycle_persists_success_and_failure_in_distinct_paths(
    tmp_path, monkeypatch
):
    import taei.nano_dataset.engine as engine
    import taei.nano_dataset.metadata as metadata

    class FakeBackend:
        def __init__(self, *args):
            pass

        def prepare(self):
            return {"max_model_len": 32768}

        def prime_requests(self, requests):
            return {"request_count": len(requests)}

        def synchronize(self):
            pass

        def generate(self, requests):
            if fail:
                raise RuntimeError("controlled engine failure")
            return [
                {
                    "request_id": r["request_id"],
                    "status": "completed",
                    "output_token_ids": [3] * r["requested_output_tokens"],
                    "output_token_count": r["requested_output_tokens"],
                    "input_token_count": len(r["input_token_ids"]),
                    "cache_hit_tokens": 0,
                    "prefix_reused_tokens": 0,
                }
                for r in requests
            ]

        def close(self):
            pass

    class Sampler:
        def __init__(self, *args):
            pass

        def snapshot(self):
            return {
                "source": "test",
                "interval_s": 1,
                "sample_count": 0,
                "running": False,
                "last_error": None,
            }

        def start(self):
            pass

        def stop(self):
            pass

    fail = False
    monkeypatch.setattr(engine, "NanoVLLMBackend", FakeBackend)
    monkeypatch.setattr(metadata, "TelemetrySampler", Sampler)
    monkeypatch.setattr(metadata, "collect_hardware", lambda: {})
    monkeypatch.setattr(metadata, "collect_environment", lambda: {})
    monkeypatch.setattr(metadata, "collect_model", lambda _: {})
    spec = build_run_matrix()[0]
    ok = run_attempt(spec, "unused", {}, tmp_path)
    fail = True
    bad = run_attempt(spec, "unused", {}, tmp_path)
    assert ok["status"] == "success" and bad["status"] == "failed"
    assert ok["path"] != bad["path"] and ok["attempt_id"] != bad["attempt_id"]
    for result in (ok, bad):
        folder = Path(result["path"])
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["attempt_id"] == result["attempt_id"]
        assert "runs.json" in manifest["files"]
    assert (Path(bad["path"]) / "traceback.txt").is_file()


def test_pinned_model_download_reuses_owned_snapshot_and_rejects_mutation(
    tmp_path, monkeypatch
):
    revision = "a" * 40
    calls = []
    monkeypatch.setattr(
        model_download.HfApi,
        "model_info",
        lambda self, **kw: type("Info", (), {"sha": revision})(),
    )

    def download(**kw):
        calls.append(kw)
        dest = Path(kw["local_dir"])
        (dest / "config.json").write_text('{"model_type":"fixture"}')
        (dest / "tokenizer.json").write_text('{"version":"1"}')
        (dest / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(model_download, "snapshot_download", download)
    target = tmp_path / "snapshot"
    first = model_download.ensure_model(target, "org/model", "main")
    second = model_download.ensure_model(target, "org/model", "main")
    assert first["resolved_revision"] == revision and second["status"] == "reused_local"
    assert len(calls) == 1 and calls[0]["revision"] == revision
    (target / "config.json").write_text('{"model_type":"changed"}')
    with pytest.raises(model_download.ModelDownloadError, match="modified"):
        model_download.ensure_model(target, "org/model", "main")


def _write_fixture_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_fixture_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _valid_synthetic_attempt(root: Path) -> None:
    """Build coherent structural evidence, not proof of GPU inference."""
    workload = build_workload(WORKLOAD_SETS[0], 42)
    assert len(workload["requests"]) == 16
    ids = {
        "run_id": "R01",
        "attempt_id": "A1",
        "hardware_id": "H1",
        "environment_id": "E1",
        "model_id": "M1",
        "workload_id": workload["workload_id"],
        "config_id": "CFG1",
    }
    for artifact, identifier in (
        ("hardware", "H1"),
        ("environment", "E1"),
        ("model", "M1"),
    ):
        _write_fixture_json(
            root / f"{artifact}.json",
            {f"{artifact}_id": identifier, "field_status": {}},
        )
    _write_fixture_json(root / "workload.json", workload)

    requests, steps, joins, tokens, scheduler = [], [], [], [], []
    total_outputs = 0

    def stamp(ns: int) -> dict:
        return {
            "timestamp_utc": "2026-10-23T10:00:00+00:00",
            "monotonic_ns": ns,
            "clock_id": "synthetic-test-clock",
            "process_id": 99,
        }

    for request in workload["requests"]:
        rid = request["request_id"]
        output = [11] * request["requested_output_tokens"]
        total_outputs += len(output)
        requests.append(
            {
                **request,
                **ids,
                "input_token_count": len(request["input_token_ids"]),
                "output_token_ids": output,
                "actual_output_tokens": len(output),
                "output_token_count": len(output),
                "status": "completed",
                "field_status": {
                    name: {"status": "not_applicable", "reason": "unique prefix"}
                    for name in ("prefix_family_id", "prefix_group_id")
                    if request[name] is None
                },
            }
        )
        index = request["arrival_index"]
        sid = f"S{index + 1:02}"
        wave = str(index // 8)
        start = 10_000 + index * 10_000
        count = len(output)
        steps.append(
            {
                **stamp(start + 900),
                "step_id": sid,
                "step_index": index,
                "phase": "decode",
                "request_ids": [rid],
                "step_host_start_ns": start,
                "step_host_end_ns": start + 900,
                "actual_batch_size": 1,
                "produced_tokens": count,
                "scheduled_tokens": count,
                "fresh_prefill_tokens": 0,
                "recompute_tokens": 0,
                "prefix_reused_tokens": 0,
                "wave_id": wave,
            }
        )
        joins.append(
            {
                **stamp(start + 800),
                "step_id": sid,
                "request_id": rid,
                "relation": "scheduled",
                "sequence_index": 0,
                "context_tokens_before": len(request["input_token_ids"]),
                "context_tokens_after": len(request["input_token_ids"]) + count,
                "scheduled_tokens": count,
                "produced_tokens": count,
                "fresh_prefill_tokens": 0,
                "recompute_tokens": 0,
                "prefix_reused_tokens": 0,
            }
        )
        tokens.extend(
            {
                **stamp(start + 700 + token_index),
                "step_id": sid,
                "request_id": rid,
                "token_index": token_index,
                "token_id": token,
                "token_type": "sampled_output",
            }
            for token_index, token in enumerate(output)
        )
        scheduler.append(
            {
                **stamp(start + 100),
                "step_id": sid,
                "event_type": "schedule",
                "request_ids": [rid],
                "request_count": 1,
                "admitted_request_ids": [rid],
                "wave_id": wave,
            }
        )
    _write_fixture_jsonl(root / "requests.jsonl", requests)
    _write_fixture_jsonl(root / "steps.jsonl", steps)
    _write_fixture_jsonl(root / "step_requests.jsonl", joins)
    _write_fixture_jsonl(root / "token_events.jsonl", tokens)
    _write_fixture_jsonl(root / "scheduler_events.jsonl", scheduler)
    _write_fixture_jsonl(
        root / "cache_events.jsonl",
        [{**stamp(1_000), "operation": "allocate", "prefix_cache_mode": "on"}],
    )
    _write_fixture_jsonl(root / "telemetry.jsonl", [])
    _write_fixture_json(
        root / "runs.json",
        {
            **ids,
            "status": "success",
            "primary_status": "accepted",
            "condition_id": "C01",
            "pair_id": "P1",
            "repeat_id": "01",
            "run_order": 1,
            "prefix_cache_mode": "on",
            "scheduler_mode": "fixed_batch",
            "metrics": {
                "request_count": len(requests),
                "output_tokens": total_outputs,
                "requested_output_tokens": total_outputs,
                "cache_hit_ratio_definition": (
                    "prefix_reused_tokens / input_token_count"
                ),
            },
            "timing_policy": {"primary_analysis_gate": {"status": "pending"}},
        },
    )
    files = {}
    for path in root.iterdir():
        raw = path.read_bytes()
        files[path.name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    _write_fixture_json(
        root / "manifest.json",
        {
            "schema_version": "1.0.0",
            "collector_version": "synthetic-fixture",
            **ids,
            "files": files,
            "coverage": {"primary_complete": True, "missing_hooks": []},
            "hash_policy": "SHA-256 exact bytes",
        },
    )


def test_schema_and_qc_path_audit_accept_safe_member_then_reject_traversal(tmp_path):
    assert DATASET_SCHEMA["$schema"].endswith("2020-12/schema")
    assert validate_instance({"bad": True})
    root = tmp_path.resolve()
    assert _safe_member(root, "requests.jsonl") == root / "requests.jsonl"
    with pytest.raises(ValueError, match="unsafe"):
        _safe_member(root, "../outside.json")

    _valid_synthetic_attempt(root)
    accepted = audit_run_directory(root)
    assert accepted["accepted"] is True, accepted["errors"]
    assert accepted["primary_status"] == "not_evaluated"
    assert accepted["errors"] == []
    assert accepted["counts"]["requests"] == 16
    assert accepted["counts"]["steps"] == 16

    requests_path = root / "requests.jsonl"
    original_requests = requests_path.read_bytes()
    requests_path.write_bytes(original_requests + b" ")
    tampered = audit_run_directory(root)
    assert tampered["accepted"] is False
    assert "hash mismatch: requests.jsonl" in tampered["errors"]
    requests_path.write_bytes(original_requests)

    manifest_path = root / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    manifest["files"]["../escape.json"] = {"sha256": "0" * 64, "bytes": 0}
    _write_fixture_json(manifest_path, manifest)
    traversal = audit_run_directory(root)
    assert traversal["accepted"] is False
    assert "unsafe manifest path: '../escape.json'" in traversal["errors"]
    manifest_path.write_bytes(original_manifest)


def test_artifact_jsonl_bytes_and_hashes_are_stable(tmp_path):
    payload = [{"z": 1, "unicode": "λ"}, {"a": [1, 2]}]
    one, two = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    write_jsonl(one, payload)
    write_jsonl(two, payload)
    expected = '{"z":1,"unicode":"λ"}\n{"a":[1,2]}\n'.encode()
    assert one.read_bytes() == expected
    assert two.read_bytes() == expected
    expected_hash = hashlib.sha256(expected).hexdigest()
    assert hashlib.sha256(one.read_bytes()).hexdigest() == expected_hash
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(ValueError):
        canonical_json({"not_json": float("nan")})
