"""Проверяет создание итогового архива логов при успехе и сбоях запуска.

Набор проверяет локальный объединенный лог и состав ZIP-архива, а также
архивирование при ошибке инициализации ClearML, исключении callback-функции и
KeyboardInterrupt до повторного выбрасывания исходного исключения.
"""

import zipfile
from types import SimpleNamespace

from omegaconf import OmegaConf

from taei.runtime import run_experiment


def _config(**overrides):
    config = {
        "experiment": {"name": "unit", "description": "test"},
        "seed": None,
        "clearml": {"enabled": False},
        "logging": {"level": "INFO", "markup": False, "clearml": False},
    }
    config.update(overrides)
    return OmegaConf.create(config)


def _hydra_output(monkeypatch, output_dir):
    from hydra.core.hydra_config import HydraConfig

    monkeypatch.setattr(
        HydraConfig,
        "get",
        staticmethod(
            lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(output_dir)))
        ),
    )


def test_success_creates_local_log_and_finalized_zip(monkeypatch, tmp_path, capsys):
    output_dir = tmp_path / "hydra-output"
    output_dir.mkdir()
    (output_dir / "hydra.log").write_text("hydra output\n", encoding="utf-8")
    _hydra_output(monkeypatch, output_dir)

    result = run_experiment(
        _config(), lambda _: print("callback stdout") or {"answer": 42}
    )

    assert result == {"answer": 42}
    console = capsys.readouterr()
    assert "callback stdout" in console.out
    run_log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert "callback stdout" in run_log
    assert "status=finished" in run_log
    with zipfile.ZipFile(output_dir / "logs.zip") as archive:
        assert set(archive.namelist()) == {"run.log", "hydra.log"}
        assert archive.read("run.log") == (output_dir / "run.log").read_bytes()
        assert archive.read("hydra.log") == b"hydra output\n"


def test_clearml_init_failure_still_saves_config_log_and_zip(monkeypatch, tmp_path):
    from clearml import Task

    output_dir = tmp_path / "hydra-output"
    _hydra_output(monkeypatch, output_dir)
    monkeypatch.setattr(Task, "current_task", staticmethod(lambda: None))

    def fail_init(**kwargs):
        assert (output_dir / "run.log").is_file()
        assert (output_dir / "config.yaml").is_file()
        raise RuntimeError("connection unavailable")

    monkeypatch.setattr(Task, "init", fail_init)
    config = _config(clearml={"enabled": True, "project_name": "TAEI"})

    try:
        run_experiment(config, lambda _: {})
    except RuntimeError as error:
        assert str(error) == "connection unavailable"
    else:
        raise AssertionError("ClearML initialization exception was swallowed")

    assert (output_dir / "config.yaml").is_file()
    run_log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert "ClearML task initialization starting" in run_log
    assert "status=failed" in run_log
    with zipfile.ZipFile(output_dir / "logs.zip") as archive:
        assert "run.log" in archive.namelist()
        archived_log = archive.read("run.log").decode("utf-8")
        assert archived_log == run_log
    assert "RuntimeError" in run_log
    assert "connection unavailable" in run_log


def test_keyboard_interrupt_is_archived_and_reraised(monkeypatch, tmp_path):
    output_dir = tmp_path / "hydra-output"
    _hydra_output(monkeypatch, output_dir)

    def interrupt(_):
        raise KeyboardInterrupt

    try:
        run_experiment(_config(), interrupt)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("KeyboardInterrupt was swallowed")

    run_log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert "status=interrupted" in run_log
    with zipfile.ZipFile(output_dir / "logs.zip") as archive:
        assert archive.read("run.log").decode("utf-8") == run_log


def test_callback_failure_is_archived_and_reraised(monkeypatch, tmp_path):
    output_dir = tmp_path / "hydra-output"
    _hydra_output(monkeypatch, output_dir)

    def fail(_):
        private_value = "local-only-diagnostic-sentinel"
        assert private_value
        print("callback reached")
        raise ValueError("missing input file: sample.csv")

    try:
        run_experiment(_config(), fail)
    except ValueError as error:
        assert str(error) == "missing input file: sample.csv"
    else:
        raise AssertionError("callback exception was swallowed")

    run_log = (output_dir / "run.log").read_text(encoding="utf-8")
    assert "callback reached" in run_log
    assert "ValueError" in run_log
    assert "missing input file: sample.csv" in run_log
    assert "local-only-diagnostic-sentinel" not in run_log
    with zipfile.ZipFile(output_dir / "logs.zip") as archive:
        archived_log = archive.read("run.log").decode("utf-8")
        assert archived_log == run_log
        assert "ValueError" in archived_log
        assert "missing input file: sample.csv" in archived_log
        assert "local-only-diagnostic-sentinel" not in archived_log
