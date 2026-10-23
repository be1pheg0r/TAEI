"""Loguru logging helpers with Rich rendering and optional ClearML forwarding.

Call :func:`configure_logging` from the application's Hydra entrypoint after
Hydra composes its configuration. This module accepts DictConfig without
requiring Hydra as a runtime dependency.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import Progress


def _current_clearml_task() -> Any | None:
    """Return the active ClearML task without initializing or authenticating one."""
    try:
        from clearml import Task

        return Task.current_task()
    except Exception:
        return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        return value if hasattr(value, "get") else {}
    except Exception:
        return {}


def _logging_options(config: Any) -> tuple[Mapping[str, Any], bool]:
    root = _as_mapping(config)
    nested = _as_mapping(root.get("logging", root))
    clearml_config = _as_mapping(root.get("clearml", {}))
    clearml_enabled = bool(nested.get("clearml", False)) or bool(
        clearml_config.get("enabled", False)
    )
    return nested, clearml_enabled


def rich_console(*, markup: bool = True, stderr: bool = True) -> Console:
    """Create the Rich console used for styled log output."""
    return Console(stderr=stderr, markup=markup, highlight=False)


def progress(**kwargs: Any) -> Progress:
    """Create a Rich progress bar, suitable for use as a context manager."""
    return Progress(console=rich_console(), **kwargs)


def configure_logging(
    *,
    level: str | None = None,
    env_file: str | Path | None = None,
    config: Any | None = None,
) -> None:
    """Configure Rich console logging and ClearML forwarding.

    ``config`` may be a composed Hydra ``DictConfig`` or an equivalent mapping
    with ``logging.level``, ``logging.markup`` and optional ``logging.clearml``
    keys. A root ``clearml.enabled`` setting is also honored. An already active
    ClearML task is always wired, regardless of config switches. This function
    never calls ``Task.init`` or logs ClearML credentials.
    """
    if env_file is not None:
        load_dotenv(Path(env_file), override=False)

    options, clearml_enabled = _logging_options(config)
    chosen_level = level or options.get("level") or os.getenv("TAEI_LOG_LEVEL", "INFO")
    console = rich_console(markup=bool(options.get("markup", True)))
    task = _current_clearml_task()

    logger.remove()
    logger.add(
        lambda message: console.print(message, end=""),
        level=str(chosen_level).upper(),
        format="{level: <8} | {message}",
        colorize=False,
        enqueue=False,
        catch=True,
        backtrace=False,
        diagnose=False,
    )

    if task is not None or clearml_enabled:

        def forward_to_clearml(message: Any) -> None:
            active_task = task or _current_clearml_task()
            if active_task is None:
                return
            record = message.record
            active_task.get_logger().report_text(
                record["message"], level=record["level"].name
            )

        logger.add(
            forward_to_clearml,
            level=str(chosen_level).upper(),
            format="{message}",
            enqueue=False,
            catch=True,
            backtrace=False,
            diagnose=False,
        )
