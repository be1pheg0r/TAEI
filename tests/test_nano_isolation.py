"""Process isolation regressions use real CPU-only child executions."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from taei.nano_dataset.isolation import run_isolated


def _wait_dead(pid: int) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split()[2] == "Z":
            return
        time.sleep(0.02)
    pytest.fail(f"descendant {pid} survived process-group cleanup")


def test_rejects_nonfinite_and_nonpositive_timeouts(tmp_path):
    for timeout in (0, -1, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="positive"):
            run_isolated({"output_root": str(tmp_path)}, timeout)


def test_isolated_worker_has_own_pid_and_failure_is_recorded(tmp_path):
    from taei.nano_dataset.workload import build_run_matrix

    with pytest.raises(RuntimeError, match="attempt failed; artifact=") as error:
        run_isolated(
            {
                "run_spec": build_run_matrix()[0],
                "model_path": str(tmp_path / "missing"),
                "engine_config": {},
                "output_root": str(tmp_path / "runs"),
                "logging_config": {"logging": {"level": "INFO"}},
            },
            timeout_s=30,
        )
    artifact = Path(str(error.value).split("artifact=", 1)[1])
    assert json.loads((artifact / "runs.json").read_text())["status"] == "failed"


def test_production_isolated_attempt_keeps_exception_traceback(tmp_path, monkeypatch):
    from taei.nano_dataset.workload import build_run_matrix

    fake_engine = tmp_path / "fake" / "nanovllm"
    fake_engine.mkdir(parents=True)
    (fake_engine / "__init__.py").write_text("")
    (fake_engine / "llm.py").write_text(
        "raise AssertionError('child traceback diagnostic sentinel')\n"
    )
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join((str(fake_engine.parent), os.environ.get("PYTHONPATH", ""))),
    )

    result = run_isolated(
        {
            "run_spec": build_run_matrix()[0],
            "model_path": str(tmp_path / "model"),
            "engine_config": {"tensor_parallel_size": 1},
            "output_root": str(tmp_path / "runs"),
            "logging_config": {"logging": {"level": "INFO"}},
        },
        timeout_s=30,
    )

    attempt = Path(result["path"])
    run = json.loads((attempt / "runs.json").read_text())
    assert result["status"] == "failed"
    assert run["error"]["type"] == "AssertionError"
    assert "child traceback diagnostic sentinel" in run["error"]["traceback"]
    assert "fake/nanovllm/llm.py" in run["error"]["traceback"]
    assert (attempt / "traceback.txt").read_text() == run["error"]["traceback"]


def test_failed_worker_stops_series_with_artifact_path(monkeypatch, tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "nano_dataset_entry",
        Path(__file__).parents[1] / "experiments/nano_vllm_dataset/run.py",
    )
    entry = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(entry)
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    monkeypatch.setattr(
        entry,
        "run_isolated",
        lambda *args, **kwargs: {
            "status": "failed",
            "path": str(tmp_path / "failed-artifact"),
        },
    )
    monkeypatch.setattr(entry, "_prepare_for_inference", lambda *args: {})
    monkeypatch.setattr(entry, "_load_calibration_report", lambda *_: None)
    monkeypatch.setattr(
        entry,
        "build_plan",
        lambda *_: {
            "matrix": [
                {"run_id": "a", "run_order": 1, "condition_id": "x"},
                {"run_id": "b", "run_order": 2, "condition_id": "x"},
            ]
        },
    )
    monkeypatch.setattr(entry, "_check_engine_config", lambda *_: {})
    monkeypatch.setattr(
        HydraConfig,
        "get",
        lambda: OmegaConf.create({"runtime": {"output_dir": str(tmp_path)}}),
    )
    cfg = OmegaConf.create(
        {
            "mode": "run",
            "dataset": {
                "model_path": "m",
                "output_root": "r",
                "run_limit": 0,
                "order_seed": 1,
                "calibration_report": None,
                "attempt_timeout_s": 10,
                "telemetry_interval_s": 1,
            },
            "hardware_profile": {},
            "engine": {},
            "experiment": {"name": "test"},
        }
    )
    with pytest.raises(RuntimeError, match="failed.*failed-artifact"):
        entry.run(cfg)


def test_timeout_kills_stubborn_descendant_process_group(tmp_path):
    marker = tmp_path / "pid"
    code = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); "
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.02)
        assert marker.exists()
        descendant = int(marker.read_text())
        from taei.nano_dataset.isolation import _stop_group

        _stop_group(process.pid, process, 0.1)
        _wait_dead(descendant)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_timeout_diagnostic_is_unique_per_attempt(tmp_path):
    from taei.nano_dataset.isolation import _write_timeout_diagnostic

    first = _write_timeout_diagnostic(
        tmp_path / "timeout.json", 1.25, "run-a", "attempt-a"
    )
    second = _write_timeout_diagnostic(
        tmp_path / "timeout.json", 1.25, "run-a", "attempt-a"
    )
    assert first != second and first.exists() and second.exists()


def test_logging_config_is_forwarded_in_worker_payload(monkeypatch, tmp_path):
    import taei.logging
    from taei.nano_dataset import orchestrator
    from taei.nano_dataset.isolation import _worker

    config = {
        "logging": {"level": "DEBUG", "markup": False},
        "clearml": {"enabled": False},
    }
    configured = []
    monkeypatch.setattr(
        taei.logging, "configure_logging", lambda **kw: configured.append(kw)
    )
    monkeypatch.setattr(orchestrator, "run_attempt", lambda: {"status": "success"})
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(json.dumps({"logging_config": config}))
    _worker(request, response)
    assert configured == [{"config": config}]
    assert json.loads(response.read_text()) == {"status": "success"}


def test_supervisor_timeout_persists_diagnostic(tmp_path):
    with pytest.raises(TimeoutError, match="diagnostic="):
        run_isolated(
            {"output_root": str(tmp_path), "run_spec": {"run_id": "R17"}}, 0.001
        )
    records = list(tmp_path.glob("supervisor-timeout-*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["run_id"] == "R17"
