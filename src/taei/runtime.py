from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

from clearml import Task
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from taei.logging import configure_logging
from taei.metrics import log_metrics


def run_experiment(config: DictConfig, run: Callable[[DictConfig], dict]) -> dict:
    """Run one experiment with logging, optional ClearML, and local artifacts."""
    configure_logging(config=config)
    seed = config.get("seed")
    if seed is not None:
        random.seed(seed)

    task = Task.current_task()
    owned_task = False
    if config.get("clearml", {}).get("enabled", False) and task is None:
        task = Task.init(
            project_name=config.clearml.get("project_name", "TAEI"),
            task_name=config.experiment.name,
            auto_connect_frameworks=False,
        )
        owned_task = True

    try:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.yaml").write_text(
            OmegaConf.to_yaml(config, resolve=True), encoding="utf-8"
        )
        logger.info("experiment={} status=started", config.experiment.name)
        metrics = run(config)
        log_metrics(metrics)
        (output_dir / "results.json").write_text(json.dumps(metrics), encoding="utf-8")
        logger.info("experiment={} status=finished", config.experiment.name)
        return metrics
    except BaseException as error:
        if isinstance(error, Exception):
            logger.exception("experiment={} status=failed", config.experiment.name)
        if owned_task and task is not None:
            task.mark_failed(status_reason=str(error))
        raise
    finally:
        if owned_task and task is not None:
            task.close()
