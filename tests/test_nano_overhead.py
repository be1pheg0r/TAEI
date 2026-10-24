"""Проверяет парную диагностику накладных расходов подробной трассировки.

Проверяются идентичность запросов и условий между минимальным и подробным
профилями, совпадение фактических результатов и токенных трасс, хэши артефактов
и сохранность предыдущих попыток. Ошибки запросов, неполные трассы, нулевое или
отрицательное время и сбои профилей исключают признание калибровки успешной.
"""

from __future__ import annotations

import hashlib
import json

from taei.nano_dataset import overhead


class TraceBackend:
    """Test-only backend; production always uses real NanoVLLMBackend."""

    def __init__(self, *, measurement_profile, emit, **kwargs):
        self.profile = measurement_profile
        self.emit = emit
        self.kwargs = kwargs
        self.prepared = False
        self.sync_count = 0

    def prepare(self):
        self.prepared = True
        return {
            "engine_revision": "bb823b3e06983d71485a8e1f23715ebd87d98ef8",
            "engine_source_hashes": {"fixture.py": "a" * 64},
            "effective_config": {"max_num_seqs": 8},
            "measurement_profile": self.profile,
        }

    def synchronize(self):
        assert self.prepared
        self.sync_count += 1

    def generate(self, requests):
        for i, request in enumerate(requests):
            self.emit(
                "token_events", {"request_id": request["request_id"], "token_id": i}
            )
        return [
            {
                "request_id": request["request_id"],
                "input_token_ids": list(request["input_token_ids"]),
                "output_token_ids": [100 + i],
                "output_token_count": 1,
                "status": "completed",
            }
            for i, request in enumerate(requests)
        ]

    def close(self):
        pass


def setup_test_backend(monkeypatch):
    monkeypatch.setattr(overhead, "NanoVLLMBackend", TraceBackend)
    monkeypatch.setattr(
        overhead,
        "collect_hardware",
        lambda: {
            "cpu_model": "fixture",
            "gpu_devices": [{"name": "fixture GPU", "uuid": "GPU-fixture"}],
        },
    )
    monkeypatch.setattr(
        overhead,
        "collect_environment",
        lambda: {
            "source_commit": "abc",
            "python_version": "3.12",
            "torch_version": "2",
        },
    )
    monkeypatch.setattr(
        overhead,
        "collect_model",
        lambda path: {
            "config_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "model_revision": "model-rev",
            "tokenizer_revision": "token-rev",
        },
    )


def test_paired_profiles_use_same_requests_capture_actual_outputs_and_hash_artifacts(
    tmp_path, monkeypatch
):
    setup_test_backend(monkeypatch)
    progress_events = []
    report = overhead.measure_overhead(
        "/fixture-model",
        {"max_num_seqs": 8},
        tmp_path,
        workload_name="S",
        seed=42,
        progress_callback=lambda stage, outcome: progress_events.append(
            (stage, outcome)
        ),
    )

    assert report["status"] == "complete"
    assert report["diagnostic_only"] is True
    assert report["pair_count"] == 1
    pair = report["pairs"][0]
    assert pair["trace_identity_match"] is True
    assert (
        pair["profiles"]["minimal"]["request_order"]
        == pair["profiles"]["detailed"]["request_order"]
    )
    assert (
        pair["profiles"]["minimal"]["outputs"]
        == pair["profiles"]["detailed"]["outputs"]
    )
    assert pair["profiles"]["minimal"]["synchronize_count"] == 2
    assert pair["profiles"]["detailed"]["synchronize_count"] == 2
    assert pair["overhead"]["wall_delta_seconds"] is not None
    assert pair["overhead"]["wall_ratio_detailed_over_minimal"] > 0
    assert pair["overhead"]["units"] == "seconds and dimensionless ratio"
    assert report["identity"]["model_details"]["model_revision"] == "model-rev"
    assert report["identity"]["engine_source_hashes"] == {"fixture.py": "a" * 64}
    assert progress_events == [
        ("Calibration pair 1/1: minimal", None),
        ("Calibration pair 1/1: minimal", "success"),
        ("Calibration pair 1/1: detailed", None),
        ("Calibration pair 1/1: detailed", "success"),
    ]
    assert (
        report["version_policy"]["requires_exact_revision_and_source_hash_match"]
        is True
    )
    assert report["version_policy"]["status"] == "matched"
    artifact_dir = tmp_path / report["diagnostic_id"]
    for name, record in report["files"].items():
        raw = (artifact_dir / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        assert len(raw) == record["bytes"]
    persisted = json.loads((artifact_dir / "overhead.json").read_text())
    assert persisted["status"] == "complete"
    assert persisted["pairs"][0]["profiles"]["minimal"]["outputs"]


def test_different_token_traces_are_incomplete_not_calibration(tmp_path, monkeypatch):
    class DivergentBackend(TraceBackend):
        def generate(self, requests):
            records = super().generate(requests)
            if self.profile == "detailed":
                records[0]["output_token_ids"] = [999]
            return records

    setup_test_backend(monkeypatch)
    monkeypatch.setattr(overhead, "NanoVLLMBackend", DivergentBackend)
    report = overhead.measure_overhead(
        "/fixture", {}, tmp_path, workload_name="S", seed=42
    )
    assert report["status"] == "incomplete"
    assert report["calibration_applicable"] is False
    assert "token_trace_mismatch" in report["incomplete_reasons"]
    assert report["pairs"][0]["trace_identity_match"] is False
    assert report["files"]


def test_returned_failed_request_does_not_count_as_calibration(tmp_path, monkeypatch):
    class FailedRequestBackend(TraceBackend):
        def generate(self, requests):
            records = super().generate(requests)
            records[0]["status"] = "failed_or_no_output"
            records[0]["output_token_ids"] = []
            return records

    setup_test_backend(monkeypatch)
    monkeypatch.setattr(overhead, "NanoVLLMBackend", FailedRequestBackend)
    report = overhead.measure_overhead(
        "/fixture", {}, tmp_path, workload_name="S", seed=42
    )
    assert report["status"] == "incomplete"
    assert report["calibration_applicable"] is False
    assert "request_generation_incomplete" in report["incomplete_reasons"]


def test_repeated_diagnostics_preserve_prior_attempt_artifacts(tmp_path, monkeypatch):
    setup_test_backend(monkeypatch)
    first = overhead.measure_overhead(
        "/fixture", {}, tmp_path, workload_name="S", seed=42
    )
    first_path = tmp_path / first["diagnostic_id"] / "overhead.json"
    first_id = json.loads(first_path.read_text())["diagnostic_id"]
    second = overhead.measure_overhead(
        "/fixture", {}, tmp_path, workload_name="S", seed=42
    )
    assert second["diagnostic_id"] != first_id
    assert json.loads(first_path.read_text())["diagnostic_id"] == first_id


def test_failed_profile_is_persisted_as_incomplete_attempt(tmp_path, monkeypatch):
    class FailingBackend(TraceBackend):
        def prepare(self):
            if self.profile == "detailed":
                raise RuntimeError("fixture failure")
            return super().prepare()

    setup_test_backend(monkeypatch)
    monkeypatch.setattr(overhead, "NanoVLLMBackend", FailingBackend)
    report = overhead.measure_overhead(
        "/fixture", {}, tmp_path, workload_name="S", seed=42
    )
    assert report["status"] == "incomplete"
    assert report["calibration_applicable"] is False
    assert report["pairs"][0]["profiles"]["detailed"]["status"] == "failed"
    assert (
        report["pairs"][0]["profiles"]["detailed"]["error"]["message"]
        == "fixture failure"
    )
    assert (tmp_path / report["diagnostic_id"] / "overhead.json").is_file()


def test_nonpositive_wall_time_never_claims_calibration(tmp_path, monkeypatch):
    setup_test_backend(monkeypatch)
    ticks = iter([100, 200, 200, 200])
    monkeypatch.setattr(overhead.time, "monotonic_ns", lambda: next(ticks))
    report = overhead.measure_overhead(
        "/fixture", {}, tmp_path, workload_name="S", seed=42
    )
    assert report["status"] == "incomplete"
    assert report["calibration_applicable"] is False
    assert "nonpositive_wall_time" in report["incomplete_reasons"]
