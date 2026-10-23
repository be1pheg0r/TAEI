from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"


def run_script(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_each_experiment_has_hydra_script_and_root_config() -> None:
    for name in ("dummy", "smoke", "_template"):
        folder = EXPERIMENTS / name
        assert (folder / "run.py").is_file()
        config_path = folder / "config.yaml"
        assert config_path.is_file()
        config = OmegaConf.load(config_path)
        assert config.defaults[0] == "base"
        assert config.defaults[1] == "_self_"
        assert config.hydra.searchpath == ["pkg://taei.conf"]
        assert config.experiment.name
        assert config.experiment.description
        assert config.metrics


def test_scripts_compose_from_outside_repo_without_network(tmp_path: Path) -> None:
    for name in ("dummy", "smoke", "_template"):
        result = run_script(EXPERIMENTS / name / "run.py", "--cfg", "job", cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        assert "experiment:" in result.stdout
        assert "metrics:" in result.stdout


@pytest.mark.parametrize(
    ("name", "expected_metrics"),
    [
        ("smoke", {"smoke_checks": 1}),
        ("_template", {"iterations_completed": 3, "synthetic_total": 6.0}),
    ],
)
def test_cpu_safe_examples_write_metadata_and_results(
    tmp_path: Path, name: str, expected_metrics: dict[str, int | float]
) -> None:
    run_dir = tmp_path / f"{name}-output"
    result = run_script(
        EXPERIMENTS / name / "run.py",
        f"hydra.run.dir={run_dir}",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    saved_cfg = OmegaConf.load(run_dir / "config.yaml")
    metrics = json.loads((run_dir / "results.json").read_text())
    assert saved_cfg.experiment.description
    assert metrics == expected_metrics


def test_dummy_cli_override_is_saved_as_metrics_and_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "dummy-output"
    result = run_script(
        EXPERIMENTS / "dummy" / "run.py",
        "metrics.score=0.91",
        f"hydra.run.dir={run_dir}",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    saved_cfg = OmegaConf.load(run_dir / "config.yaml")
    metrics = json.loads((run_dir / "results.json").read_text())
    assert saved_cfg.experiment.name == "dummy_log_collection"
    assert metrics["score"] == pytest.approx(0.91)
    assert metrics["examples_seen"] == 3


@pytest.mark.parametrize(
    ("name", "metric_names"),
    [
        ("dummy", ("examples_seen", "score")),
        ("smoke", ("smoke_checks",)),
        ("_template", ("iterations_completed", "synthetic_total")),
    ],
)
def test_cli_logs_each_final_metric_once(
    tmp_path: Path, name: str, metric_names: tuple[str, ...]
) -> None:
    result = run_script(
        EXPERIMENTS / name / "run.py",
        f"hydra.run.dir={tmp_path / name}",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    for metric_name in metric_names:
        assert output.count(f"metric {metric_name}=") == 1


def test_dummy_multirun_records_each_override_in_separate_run(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "sweep"
    result = run_script(
        EXPERIMENTS / "dummy" / "run.py",
        "-m",
        "metrics.score=0.2,0.8",
        f"hydra.sweep.dir={sweep_dir}",
        "hydra.sweep.subdir=${hydra.job.num}",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    observed = []
    for index, expected in enumerate((0.2, 0.8)):
        run_dir = sweep_dir / str(index)
        saved_cfg = OmegaConf.load(run_dir / "config.yaml")
        metrics = json.loads((run_dir / "results.json").read_text())
        assert saved_cfg.experiment.name == "dummy_log_collection"
        assert metrics["score"] == pytest.approx(expected)
        observed.append(metrics["score"])
    assert observed == pytest.approx([0.2, 0.8])


def test_invalid_override_exits_nonzero_without_creating_metrics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "invalid-run"
    result = run_script(
        EXPERIMENTS / "dummy" / "run.py",
        "metrics.unknown=1",
        f"hydra.run.dir={run_dir}",
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert not (run_dir / "results.json").exists()


def test_experiment_yaml_comments_explain_intent_and_expected_result() -> None:
    for name in ("dummy", "smoke", "_template"):
        text = (EXPERIMENTS / name / "config.yaml").read_text().lower()
        assert "#" in text
        assert any(
            word in text for word in ("цель", "зачем", "назнач", "aim", "purpose")
        )
        assert any(
            word in text for word in ("ожида", "результат", "expected", "result")
        )
    smoke = (EXPERIMENTS / "smoke" / "config.yaml").read_text().lower()
    assert "synthetic" in smoke or "синтет" in smoke
    assert "qwen" in smoke
    assert "inference" in smoke or "инференс" in smoke
