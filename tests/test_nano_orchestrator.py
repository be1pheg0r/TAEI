"""Проверяет планирование, безопасный preflight, калибровку и попытки запуска.

Набор проверяет размеры и итоги плана, отсутствие запуска инференса при
preflight, точное сопоставление идентичности/нагрузки/seed/режимов для
калибровочных данных и блокировку при отсутствии модели, CUDA или зависимостей.
Ошибки подготовки должны приводить к сохраненной отклоненной попытке, а не к
ложному сообщению об успешном запуске.
"""

import json
import sys
from pathlib import Path

from taei.nano_dataset import orchestrator
from taei.nano_dataset.orchestrator import build_plan, preflight, run_attempt
from taei.nano_dataset.workload import build_run_matrix


def test_preflight_is_cpu_safe_and_reports_missing_gpu_without_loading_inference():
    report = preflight(model_path="/not/a/model")
    assert report["ready"] is False
    assert "model_path_missing" in report["blocking_reasons"]
    assert "torch_cuda_unavailable" in report["blocking_reasons"]
    assert report["gpu_inference_attempted"] is False


def test_plan_has_all_runs_totals_and_policies():
    plan = build_plan()
    assert plan["run_count"] == 24
    assert plan["condition_count"] == 12
    assert plan["request_count"] == 768
    assert plan["logical_prompt_tokens"] == 835584
    assert plan["requested_output_tokens"] == 245760
    assert plan["overhead_policy"]["primary_analysis_gate"] == "pending_calibration"
    assert plan["scheduler_policy"]["dynamic_max_num_seqs_semantics"].startswith(
        "per-schedule"
    )


def test_schema_declares_canonical_join_ids_in_metadata_manifest_and_requests():
    from taei.nano_dataset.schema import DATASET_SCHEMA

    props = DATASET_SCHEMA["$defs"]
    for artifact, identifier in (
        ("hardware", "hardware_id"),
        ("environment", "environment_id"),
        ("model", "model_id"),
        ("manifest", "condition_id"),
        ("manifest", "pair_id"),
        ("manifest", "repeat_id"),
        ("requests", "split_group_id"),
    ):
        assert identifier in props[artifact]["properties"]


def test_calibration_scope_is_limited_to_matching_identity_workload_seed_and_modes():
    from taei.nano_dataset.orchestrator import evaluate_calibration_scope

    report = {
        "status": "complete",
        "calibration_applicable": True,
        "workload_id": "S-seed-42",
        "seed": 42,
        "request_count": 32,
        "prefix_cache_mode": "on",
        "scheduler_mode": "upstream_dynamic",
        "profile_order": ["minimal", "detailed"],
        "pairs": [
            {
                "trace_identity_match": True,
                "profile_identity_match": True,
                "incomplete_reasons": [],
                "overhead": {"wall_ratio_detailed_over_minimal": 1.0},
            }
        ],
        "identity": {
            "matching_primary_identity": {
                "hardware_id": "H1",
                "environment_id": "E1",
                "model_id": "M1",
                "engine_config_sha256": "HASH",
            }
        },
    }
    spec = {
        "workload_id": "S",
        "seed": 42,
        "prefix_cache_mode": "on",
        "scheduler_mode": "upstream_dynamic",
    }
    identity = {"hardware_id": "H1", "environment_id": "E1", "model_id": "M1"}
    assert (
        evaluate_calibration_scope(spec, "S-seed-42", identity, "HASH", report)[
            "status"
        ]
        == "review_required"
    )
    other_seed = {**spec, "seed": 43}
    assert (
        evaluate_calibration_scope(other_seed, "S-seed-43", identity, "HASH", report)[
            "status"
        ]
        == "out_of_scope"
    )


def test_preflight_blocks_path_without_tokenizer_weights_or_cuda_runtime(
    tmp_path, monkeypatch
):
    (tmp_path / "config.json").write_text('{"model_type":"fixture"}')
    monkeypatch.setattr(
        orchestrator,
        "collect_hardware",
        lambda: {"gpu_devices": [{"name": "fixture"}]},
    )
    monkeypatch.setattr(
        orchestrator, "collect_environment", lambda: {"torch_version": "2.x"}
    )
    monkeypatch.setattr(
        orchestrator, "_installed_distributions", lambda: {"nano-vllm", "torch"}
    )
    monkeypatch.setattr(orchestrator, "_cuda_available", lambda: False)
    report = preflight(str(tmp_path))
    assert report["ready"] is False
    assert "model_weights_missing" in report["blocking_reasons"]
    assert "tokenizer_assets_missing" in report["blocking_reasons"]
    assert "torch_cuda_unavailable" in report["blocking_reasons"]


def test_missing_inference_dependency_writes_rejected_immutable_attempt(
    tmp_path, monkeypatch
):
    # Exercise the missing-dependency path even with the inference extra installed.
    monkeypatch.setitem(sys.modules, "nanovllm", None)
    spec = build_run_matrix()[0]
    result = run_attempt(
        spec,
        model_path=str(tmp_path / "absent-model"),
        engine_config={"tensor_parallel_size": 1},
        output_root=tmp_path / "runs",
    )
    attempt = tmp_path / "runs" / spec["run_id"] / result["attempt_id"]
    assert result["status"] == "failed"
    assert result["primary_status"] == "rejected"
    assert (attempt / "manifest.json").is_file()
    assert (attempt / "runs.json").is_file()
    assert (attempt / "schema_validation.json").is_file()
    run = json.loads((attempt / "runs.json").read_text())
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert run["status"] == "failed"
    assert run["primary_status"] == "rejected"
    assert run["error"]["type"] == "RuntimeError"
    assert manifest["coverage"]["primary_analysis_accepted"] is False
    validation = json.loads((attempt / "schema_validation.json").read_text())
    assert validation["valid"] is True, validation["errors"]
    assert len((attempt / "requests.jsonl").read_text().splitlines()) == 32


def test_prepare_failure_persists_complete_traceback_in_attempt_artifact(
    tmp_path, monkeypatch
):
    from taei.nano_dataset import engine

    class FailingBackend:
        def __init__(self, *args):
            pass

        def prepare(self):
            raise AssertionError("diagnostic traceback sentinel")

        def close(self):
            pass

    monkeypatch.setattr(engine, "NanoVLLMBackend", FailingBackend)
    result = run_attempt(
        build_run_matrix()[0], str(tmp_path / "model"), {}, tmp_path / "runs"
    )
    attempt = Path(result["path"])
    run = json.loads((attempt / "runs.json").read_text())
    traceback_text = run["error"].get("traceback", "")

    assert "Traceback (most recent call last):" in traceback_text
    assert "diagnostic traceback sentinel" in traceback_text
    assert "test_nano_orchestrator.py" in traceback_text
    assert (attempt / "traceback.txt").read_text() == traceback_text


def test_attempt_preserves_engine_field_status_reasons(tmp_path, monkeypatch):
    """A test-only backend proves status provenance, not GPU execution."""
    from taei.nano_dataset import engine

    class BackendFixture:
        def __init__(self, *args):
            pass

        def prepare(self):
            return {"measurement_profile": "test_fixture"}

        def synchronize(self):
            pass

        def generate(self, requests):
            return [
                {
                    "request_id": request["request_id"],
                    "status": "success",
                    "output_token_ids": [11] * request["requested_output_tokens"],
                    "output_token_count": request["requested_output_tokens"],
                    "first_generated_monotonic_ns": None,
                    "field_status": {
                        "first_generated_monotonic_ns": {
                            "status": "unsupported",
                            "reason": "fixture: GPU timestamp hook unavailable",
                        }
                    },
                }
                for request in requests
            ]

        def close(self):
            pass

    monkeypatch.setattr(engine, "NanoVLLMBackend", BackendFixture)
    result = run_attempt(
        build_run_matrix()[0], str(tmp_path / "model"), {}, tmp_path / "runs"
    )
    first = json.loads(
        (Path(result["path"]) / "requests.jsonl").read_text().splitlines()[0]
    )
    assert first["field_status"]["first_generated_monotonic_ns"] == {
        "status": "unsupported",
        "reason": "fixture: GPU timestamp hook unavailable",
    }
    telemetry = [
        json.loads(line)
        for line in (Path(result["path"]) / "telemetry.jsonl").read_text().splitlines()
    ]
    assert {row.get("sample_scope") for row in telemetry} >= {
        "before_measurement",
        "after_measurement",
    }
