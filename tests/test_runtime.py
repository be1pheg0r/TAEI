import random
from random import Random
from types import SimpleNamespace

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from taei.runtime import run_experiment


def _hydra_output(monkeypatch, tmp_path):
    from hydra.core.hydra_config import HydraConfig

    monkeypatch.setattr(
        HydraConfig,
        "get",
        staticmethod(
            lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path)))
        ),
    )


def _config(**overrides):
    config = {
        "experiment": {"name": "unit", "description": "test"},
        "metrics": {"answer": 42},
        "seed": 7,
        "clearml": {"enabled": False},
        "logging": {"level": "INFO", "markup": False, "clearml": False},
    }
    config.update(overrides)
    return OmegaConf.create(config)


def test_runtime_returns_logs_and_writes_metrics(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _hydra_output(monkeypatch, tmp_path)

    result = run_experiment(_config(), lambda _: {"answer": 42})

    assert result == {"answer": 42}
    assert "metric answer=42" in capsys.readouterr().err
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "results.json").read_text() == '{"answer": 42}'


def test_runtime_seeds_python_random_generator(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _hydra_output(monkeypatch, tmp_path)

    result = run_experiment(_config(), lambda _: {"draw": random.random()})

    assert result["draw"] == Random(7).random()


def test_runtime_closes_only_the_clearml_task_it_creates(monkeypatch, tmp_path):
    from clearml import Task

    _hydra_output(monkeypatch, tmp_path)
    calls = []

    class OwnedTask:
        def close(self):
            calls.append("close")

    owned = OwnedTask()
    monkeypatch.setattr(Task, "current_task", staticmethod(lambda: None))
    monkeypatch.setattr(Task, "init", lambda **kwargs: owned)
    config = _config(clearml={"enabled": True, "project_name": "TAEI"})

    run_experiment(config, lambda _: {})

    assert calls == ["close"]


def test_runtime_closes_owned_task_when_experiment_fails(monkeypatch, tmp_path):
    from clearml import Task

    _hydra_output(monkeypatch, tmp_path)
    calls = []

    class OwnedTask:
        def mark_failed(self, **kwargs):
            calls.append("failed")

        def close(self):
            calls.append("close")

    owned = OwnedTask()
    monkeypatch.setattr(Task, "current_task", staticmethod(lambda: None))
    monkeypatch.setattr(Task, "init", lambda **kwargs: owned)
    config = _config(clearml={"enabled": True, "project_name": "TAEI"})

    try:
        run_experiment(config, lambda _: (_ for _ in ()).throw(ValueError("boom")))
    except ValueError as error:
        assert str(error) == "boom"
    else:
        raise AssertionError("experiment exception was swallowed")

    assert calls == ["failed", "close"]
    assert not (tmp_path / "results.json").exists()


def test_runtime_closes_owned_task_when_initial_setup_fails(monkeypatch, tmp_path):
    from clearml import Task

    _hydra_output(monkeypatch, tmp_path)
    calls = []

    class OwnedTask:
        def mark_failed(self, **kwargs):
            calls.append("failed")

        def close(self):
            calls.append("close")

    owned = OwnedTask()
    monkeypatch.setattr(Task, "current_task", staticmethod(lambda: None))
    monkeypatch.setattr(Task, "init", lambda **kwargs: owned)
    monkeypatch.setattr(
        OmegaConf,
        "to_yaml",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("serialize")),
    )
    config = _config(clearml={"enabled": True, "project_name": "TAEI"})

    try:
        run_experiment(config, lambda _: {})
    except ValueError as error:
        assert str(error) == "serialize"
    else:
        raise AssertionError("setup exception was swallowed")

    assert calls == ["failed", "close"]
    assert not (tmp_path / "results.json").exists()


def test_runtime_does_not_close_an_existing_clearml_task(monkeypatch, tmp_path):
    from clearml import Task

    _hydra_output(monkeypatch, tmp_path)
    calls = []

    class ExistingTask:
        def close(self):
            calls.append("close")

    monkeypatch.setattr(Task, "current_task", staticmethod(ExistingTask))
    monkeypatch.setattr(
        Task, "init", lambda **kwargs: (_ for _ in ()).throw(AssertionError())
    )
    run_experiment(_config(clearml={"enabled": True}), lambda _: {})

    assert calls == []


def test_packaged_base_config_composes_engine_and_model():
    with initialize_config_module(version_base="1.3", config_module="taei.conf"):
        cfg = compose(
            config_name="base",
            overrides=["+experiment.name=smoke"],
            return_hydra_config=True,
        )

    assert cfg.engine.backend == "nano_vllm"
    assert cfg.model.name == "Qwen3-8B"
    assert cfg.seed == 42
    assert cfg.logging.level == "INFO"
    assert cfg.hydra.run.dir.startswith("outputs/smoke/")
