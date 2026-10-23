from pathlib import Path

from clearml import Task
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from taei.experiments.base import Experiment
from taei.logging import configure_logging
from taei.run_experiment import initialize_clearml

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def test_default_config_composes_to_smoke_experiment() -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config")

    assert cfg.experiment.name == "qwen3_8b_dense_smoke"
    assert cfg.experiment._target_ == "taei.experiments.dummy.DummyExperiment"
    assert cfg.model.name == "Qwen3-8B"


def test_dummy_config_composes_with_hydra_overrides() -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=["experiment=dummy", "experiment.metrics.examples_seen=8"],
        )

    assert cfg.experiment.name == "dummy_log_collection"
    assert cfg.experiment.metrics.examples_seen == 8


def test_dummy_experiment_collects_configured_metrics(capsys) -> None:
    configure_logging(level="INFO")
    cfg = OmegaConf.create(
        {
            "experiment": {
                "_target_": "taei.experiments.dummy.DummyExperiment",
                "name": "log_collection_demo",
                "metrics": {"items_seen": 3, "score": 0.75},
            }
        }
    )
    experiment = instantiate(cfg.experiment)

    assert isinstance(experiment, Experiment)
    result = experiment.run()

    assert result == {"items_seen": 3, "score": 0.75}
    output = capsys.readouterr().err
    assert "items_seen=3" in output
    assert "score=0.75" in output


def test_clearml_enabled_initializes_task_for_log_forwarding(monkeypatch) -> None:
    initialized = {}
    monkeypatch.setattr(Task, "current_task", staticmethod(lambda: None))
    monkeypatch.setattr(Task, "init", lambda **kwargs: initialized.update(kwargs))
    cfg = OmegaConf.create(
        {
            "clearml": {"enabled": True, "project_name": "TAEI"},
            "experiment": {"name": "dummy_run"},
        }
    )

    initialize_clearml(cfg)

    assert initialized == {
        "project_name": "TAEI",
        "task_name": "dummy_run",
        "auto_connect_frameworks": False,
    }
