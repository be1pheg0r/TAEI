"""Each dataset attempt reports its outcome and actionable failure context."""

import importlib.util
import json
import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def load_entry():
    path = Path(__file__).parents[1] / "experiments/nano_vllm_dataset/run.py"
    spec = importlib.util.spec_from_file_location("nano_reporting_entry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(tmp_path):
    return OmegaConf.create(
        {
            "mode": "run",
            "dataset": {
                "model_path": "model",
                "output_root": str(tmp_path),
                "order_seed": 1,
                "workload_protocol": "fixture",
                "run_limit": 0,
                "attempt_timeout_s": 10,
                "telemetry_interval_s": 1,
            },
            "engine": {
                "max_model_len": 16384,
                "max_num_seqs": 8,
                "max_num_batched_tokens": 16384,
            },
            "hardware_profile": {},
            "download": {"enabled": False},
        }
    )


def setup_entry(monkeypatch, tmp_path):
    entry = load_entry()
    monkeypatch.setattr(
        HydraConfig,
        "get",
        lambda: OmegaConf.create({"runtime": {"output_dir": str(tmp_path)}}),
    )
    monkeypatch.setattr(
        entry,
        "build_plan",
        lambda *a, **k: {"matrix": [{"run_id": "R1"}, {"run_id": "R2"}]},
    )
    return entry


def test_attempts_log_success_and_failure_reason_and_artifact(tmp_path, monkeypatch):
    entry = setup_entry(monkeypatch, tmp_path)
    artifact = tmp_path / "attempt"
    artifact.mkdir()
    (artifact / "runs.json").write_text(
        json.dumps(
            {
                "error": {
                    "type": "OutOfMemoryError",
                    "stage": "prepare_or_generate",
                    "message": "CUDA out of memory. password=secret-value",
                    "traceback": (
                        'Traceback\n  File "/app/runner.py", line 91, in execute\n'
                        "OutOfMemoryError\n"
                    ),
                }
            }
        )
    )
    (artifact / "traceback.txt").write_text("trace")
    attempts = iter(
        [
            {"run_id": "R1", "attempt_id": "A1", "status": "success", "path": "ok"},
            {
                "run_id": "R2",
                "attempt_id": "A2",
                "status": "failed",
                "path": str(artifact),
            },
        ]
    )
    monkeypatch.setattr(entry, "run_isolated", lambda *a, **k: next(attempts))
    logs = []
    monkeypatch.setattr(
        entry.project_logging.logger,
        "info",
        lambda msg, *a: logs.append(msg.format(*a)),
    )
    monkeypatch.setattr(
        entry.project_logging.logger,
        "error",
        lambda msg, *a: logs.append(msg.format(*a)),
    )
    result = entry.run(config(tmp_path))
    text = "\n".join(logs)
    assert result["failed_attempt_count"] == 1
    assert "R1" in text and "A1" in text and "success" in text
    assert "R2" in text and "A2" in text and "failed" in text
    assert "OutOfMemoryError" in text and "CUDA out of memory" in text
    assert "runner.py:91" in text and str(artifact / "traceback.txt") in text
    assert "secret-value" not in text


def test_supervisor_exception_is_logged_and_reraised(tmp_path, monkeypatch):
    entry = setup_entry(monkeypatch, tmp_path)

    def fail(*args, **kwargs):
        raise TimeoutError("diagnostic=/tmp/timeout.json")

    monkeypatch.setattr(entry, "run_isolated", fail)
    logs = []
    monkeypatch.setattr(
        entry.project_logging.logger,
        "error",
        lambda msg, *a: logs.append(msg.format(*a)),
    )
    with pytest.raises(TimeoutError):
        entry.run(config(tmp_path))
    assert "R1" in " ".join(logs) and "TimeoutError" in " ".join(logs)


def test_real_isolated_error_is_reported_by_project_logger(tmp_path, monkeypatch):
    from taei.nano_dataset.isolation import run_isolated
    from taei.nano_dataset.workload import build_run_matrix

    fake = tmp_path / "fake" / "nanovllm"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("")
    (fake / "llm.py").write_text("raise AssertionError()\n")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join((str(fake.parent), os.environ.get("PYTHONPATH", ""))),
    )
    spec = build_run_matrix()[0]
    result = run_isolated(
        {
            "run_spec": spec,
            "model_path": str(tmp_path / "model"),
            "engine_config": {},
            "output_root": str(tmp_path / "runs"),
        },
        timeout_s=30,
    )
    entry = load_entry()
    messages = []
    sink = entry.project_logging.logger.add(messages.append, format="{message}")
    try:
        entry._report_attempt(spec["run_id"], result)
    finally:
        entry.project_logging.logger.remove(sink)
    text = "".join(str(message) for message in messages)
    assert "status=failed" in text and "AssertionError" in text
    assert "llm.py:1" in text and str(Path(result["path"]) / "traceback.txt") in text
