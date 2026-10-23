"""Запускает Hydra-эксперименты с локальными логами и поддержкой ClearML.

Это общая среда исполнения экспериментальных скриптов: она сохраняет
разрешенную конфигурацию, применяет seed генератора Python, настраивает
журналирование и захват stdout/stderr, при необходимости создает принадлежащую
среде задачу ClearML, вызывает функцию эксперимента, записывает метрики и
создает ZIP-архив логов как при успехе, так и при ошибке. Уже активные задачи
ClearML используются повторно, но не закрываются этой средой; ошибки после
попытки очистки и архивации повторно выбрасываются вызывающему коду.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

from clearml import Task
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from taei import logging as project_logging
from taei.metrics import log_metrics


def _archive_logs(output_dir: Path) -> None:
    """Atomically archive this run's top-level log files, including run.log."""
    archive_path = output_dir / "logs.zip"
    fd, temporary_name = tempfile.mkstemp(
        prefix=".logs-", suffix=".zip.tmp", dir=output_dir
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(output_dir.iterdir()):
                if path.is_symlink() or not path.is_file():
                    continue
                if path in (archive_path, temporary_path):
                    continue
                if path.name == "run.log" or path.suffix.lower() == ".log":
                    archive.write(path, arcname=path.name)
        os.replace(temporary_path, archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _report_archive_failure(error: BaseException) -> None:
    with contextlib.suppress(Exception):
        print(f"Failed to create logs.zip ({type(error).__name__})", file=sys.stderr)


def _log_runtime_failure(error: BaseException) -> None:
    status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    logger.opt(exception=error).error(
        "experiment status={} exception_type={}", status, type(error).__name__
    )


def _cleanup_owned_task(
    task: object, error: BaseException, *, close_attempted: bool
) -> None:
    try:
        task.mark_failed(status_reason=type(error).__name__)
    except BaseException as cleanup_error:
        logger.error(
            "ClearML task failure marking failed ({})", type(cleanup_error).__name__
        )
    if not close_attempted:
        try:
            task.close()
        except BaseException as cleanup_error:
            logger.error("ClearML task close failed ({})", type(cleanup_error).__name__)


def run_experiment(config: DictConfig, run: Callable[[DictConfig], dict]) -> dict:
    """Run one experiment with always-on local logs and a finalized log archive."""
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "run.log"
    task = None
    owned_task = False
    close_attempted = False

    try:
        with project_logging.capture_output(log_path=run_log):
            try:
                project_logging.configure_logging(config=config)
                (output_dir / "config.yaml").write_text(
                    OmegaConf.to_yaml(config, resolve=True), encoding="utf-8"
                )
                seed = config.get("seed")
                if seed is not None:
                    random.seed(seed)

                task = Task.current_task()
                if config.get("clearml", {}).get("enabled", False) and task is None:
                    logger.info("ClearML task initialization starting")
                    task = Task.init(
                        project_name=config.clearml.get("project_name", "TAEI"),
                        task_name=config.experiment.name,
                        auto_connect_frameworks=False,
                    )
                    owned_task = True

                logger.info("experiment={} status=started", config.experiment.name)
                metrics = run(config)
                log_metrics(metrics)
                (output_dir / "results.json").write_text(
                    json.dumps(metrics), encoding="utf-8"
                )
                logger.info("experiment={} status=finished", config.experiment.name)
                if owned_task and task is not None:
                    close_attempted = True
                    task.close()
            except BaseException as error:
                _log_runtime_failure(error)
                if owned_task and task is not None:
                    _cleanup_owned_task(task, error, close_attempted=close_attempted)
                raise
    except BaseException:
        try:
            _archive_logs(output_dir)
        except BaseException as archive_error:
            _report_archive_failure(archive_error)
        raise

    try:
        _archive_logs(output_dir)
    except BaseException as archive_error:
        _report_archive_failure(archive_error)
        raise
    return metrics
