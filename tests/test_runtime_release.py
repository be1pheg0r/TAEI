from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from omegaconf import OmegaConf

import taei.runtime as runtime
from taei.runtime import run_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]


def _runtime_config():
    return OmegaConf.create(
        {
            "experiment": {"name": "release-check", "description": "CPU test"},
            "metrics": {},
            "seed": 17,
            "clearml": {"enabled": False},
            "logging": {"level": "INFO", "markup": False, "clearml": False},
        }
    )


def _set_hydra_output(monkeypatch, output_dir: Path) -> None:
    from types import SimpleNamespace

    from hydra.core.hydra_config import HydraConfig

    monkeypatch.setattr(
        HydraConfig,
        "get",
        staticmethod(
            lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(output_dir)))
        ),
    )


def _run_script(
    script: Path, *args: str, cwd: Path
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), str(REPO_ROOT), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )


def test_runtime_success_saves_resolved_config_results_and_valid_log_archive(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "success"
    output_dir.mkdir()
    (output_dir / "hydra.log").write_text("hydra marker\n", encoding="utf-8")
    _set_hydra_output(monkeypatch, output_dir)
    metric_calls = []
    real_log_metrics = runtime.log_metrics

    def count_metrics(metrics):
        metric_calls.append(dict(metrics))
        return real_log_metrics(metrics)

    monkeypatch.setattr(runtime, "log_metrics", count_metrics)
    config = _runtime_config()
    config.experiment.description = "resolved release config"
    returned = run_experiment(config, lambda _: {"rows": 3, "quality": 0.875})

    assert returned == {"rows": 3, "quality": 0.875}
    assert metric_calls == [returned]
    saved = OmegaConf.load(output_dir / "config.yaml")
    assert saved.experiment.description == "resolved release config"
    assert json.loads((output_dir / "results.json").read_text()) == returned
    with zipfile.ZipFile(output_dir / "logs.zip") as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {"hydra.log", "run.log"}
        assert archive.read("run.log") == (output_dir / "run.log").read_bytes()
        assert archive.read("hydra.log") == b"hydra marker\n"
        assert archive.read("run.log").count(b"metrics: rows=3, quality=0.875") == 1


def test_runtime_failure_and_interrupt_propagate_archive_logs_and_restore_streams(
    tmp_path, monkeypatch
):
    import pytest

    output_dir = tmp_path / "failure"
    _set_hydra_output(monkeypatch, output_dir)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    for exception in (ValueError("expected callback failure"), KeyboardInterrupt()):

        def fail(_config, error=exception):
            print("callback diagnostic")
            raise error

        with pytest.raises(type(exception)):
            run_experiment(_runtime_config(), fail)
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
        assert (output_dir / "logs.zip").is_file()
        assert not (output_dir / "results.json").exists()
        with zipfile.ZipFile(output_dir / "logs.zip") as archive:
            assert archive.testzip() is None
            log = archive.read("run.log").decode("utf-8")
            assert "callback diagnostic" in log
            assert (
                "status=failed"
                if isinstance(exception, ValueError)
                else "status=interrupted"
            ) in log


def test_capture_tees_python_native_and_inherited_child_output_then_restores(
    tmp_path, capfd
):
    from taei.logging import capture_output

    log_path = tmp_path / "capture.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with capture_output(log_path):
        print("python stdout café")
        print("python stderr marker", file=sys.stderr)
        os.write(sys.stdout.fileno(), b"native stdout marker\n")
        os.write(sys.stderr.fileno(), b"native stderr marker\n")
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; print('child stdout marker'); "
                "print('child stderr marker', file=sys.stderr)",
            ],
            stdout=sys.stdout.fileno(),
            stderr=sys.stderr.fileno(),
            check=False,
        )
        assert child.returncode == 0
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    text = log_path.read_text(encoding="utf-8")
    for marker in (
        "python stdout café",
        "python stderr marker",
        "native stdout marker",
        "native stderr marker",
        "child stdout marker",
        "child stderr marker",
    ):
        assert marker in text
    console = capfd.readouterr()
    assert "native stdout marker" in console.out
    assert "native stderr marker" in console.err


def test_cpu_hydra_cli_smoke_plans_standalone_defaults_and_isolates_outputs(tmp_path):
    smoke_dir = tmp_path / "smoke"
    smoke = _run_script(
        REPO_ROOT / "experiments/smoke/run.py",
        f"hydra.run.dir={smoke_dir}",
        cwd=tmp_path,
    )
    assert smoke.returncode == 0, smoke.stderr
    assert json.loads((smoke_dir / "results.json").read_text()) == {"smoke_checks": 1}
    assert "metrics: smoke_checks=1" in smoke.stderr

    dataset_dir = tmp_path / "dataset-plan"
    plan = _run_script(
        REPO_ROOT / "experiments/nano_vllm_dataset/run.py",
        "mode=plan",
        f"hydra.run.dir={dataset_dir}",
        cwd=tmp_path,
    )
    assert plan.returncode == 0, plan.stderr
    plan_data = json.loads((dataset_dir / "plan.json").read_text())
    assert len(plan_data["matrix"]) > 0
    dataset_config = OmegaConf.load(dataset_dir / "config.yaml")
    assert Path(dataset_config.dataset.output_root) == dataset_dir / "runs"
    assert (dataset_dir / "logs.zip").is_file()

    standalone_roots = []
    for name in ("nano_vllm_qwen3_0_6b", "nano_vllm_qwen3_8b"):
        output_dir = tmp_path / name
        result = _run_script(
            REPO_ROOT / f"experiments/{name}/run.py",
            "mode=plan",
            f"hydra.run.dir={output_dir}",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        standalone_config = OmegaConf.load(output_dir / "config.yaml")
        assert (
            standalone_config.dataset.workload_protocol
            == "six_set_paired_prefix_sweep_v1"
        )
        assert Path(standalone_config.dataset.output_root) == output_dir / "runs"
        assert len(json.loads((output_dir / "plan.json").read_text())["matrix"]) == 48
        standalone_roots.append(Path(standalone_config.dataset.output_root))
    assert standalone_roots[0] != standalone_roots[1]


def test_detached_launch_manifest_wait_and_relaunch_boundaries(tmp_path, monkeypatch):
    from taei import launch
    from taei.launch import launch_status, maybe_launch_detached

    worker = tmp_path / "controlled_worker.py"
    worker.write_text(
        "import os, time\nfrom pathlib import Path\n"
        "root = Path(os.environ['TAEI_RELEASE_WORKER_ROOT'])\n"
        "(root / 'started').write_text('yes')\n"
        "deadline = time.monotonic() + 15\n"
        "while not (root / 'release').exists():\n"
        "    if time.monotonic() >= deadline: raise SystemExit(1)\n"
        "    time.sleep(0.01)\n"
        "(root / 'finished').write_text('yes')\n",
        encoding="utf-8",
    )
    children = []
    gates = []
    real_popen = subprocess.Popen

    def track_child(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(launch.subprocess, "Popen", track_child)
    monkeypatch.chdir(tmp_path)
    records = set()
    try:
        for index in range(2):
            control = tmp_path / f"worker-{index}"
            control.mkdir()
            gates.append(control / "release")
            monkeypatch.setenv("TAEI_RELEASE_WORKER_ROOT", str(control))
            monkeypatch.setattr(sys, "argv", [str(worker), "--detach"])
            assert maybe_launch_detached(worker) is True
            current = set((tmp_path / "outputs/background-runs").glob("*/launch.json"))
            assert len(current - records) == 1
            record = (current - records).pop()
            records = current
            metadata = json.loads(record.read_text())
            assert metadata["status"] == "started"
            assert metadata["pid"] == children[-1].pid
            assert Path(metadata["stdout"]).is_file()
            assert Path(metadata["stderr"]).is_file()
            deadline = time.monotonic() + 5
            while not (control / "started").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert (control / "started").read_text() == "yes"
            assert launch_status(record)["pid_alive"] is True
            gates[-1].touch()
            deadline = time.monotonic() + 5
            while launch_status(record)["pid_alive"] and time.monotonic() < deadline:
                time.sleep(0.01)
            assert launch_status(record)["pid_alive"] is False
            assert children[-1].wait(timeout=5) == 0
            assert (control / "finished").read_text() == "yes"
        assert len({path.parent for path in records}) == 2
    finally:
        for gate in gates:
            gate.touch()
        for child in children:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


def test_logging_level_precedence_and_non_tty_progress_are_deterministic(
    monkeypatch, capsys
):
    from loguru import logger

    from taei import logging as project_logging

    monkeypatch.setattr(project_logging, "_current_clearml_task", lambda: None)
    monkeypatch.setenv("TAEI_LOG_LEVEL", "DEBUG")
    project_logging.configure_logging(
        config={"logging": {"level": "WARNING", "markup": False}}
    )
    logger.debug("hidden debug")
    logger.warning("configured warning")
    first = capsys.readouterr().err
    assert "configured warning" in first
    assert "hidden debug" not in first

    project_logging.configure_logging(
        level="ERROR", config={"logging": {"level": "DEBUG", "markup": False}}
    )
    logger.warning("hidden warning")
    logger.error("explicit error")
    with project_logging.experiment_progress(2, "CPU work") as progress:
        progress.stage("planning")
        progress.advance("ok")
    output = capsys.readouterr().err
    assert "explicit error" in output
    assert "hidden warning" not in output
    assert "CPU work started: 0/2" in output
    assert "planning (0/2 completed)" in output
    assert "1/2 completed (ok)" in output
    assert "CPU work finished: 1/2 completed" in output
    assert "\x1b[" not in output
