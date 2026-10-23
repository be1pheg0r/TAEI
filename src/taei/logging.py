"""Project logging setup using Loguru, Rich, and optional dotenv settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console


def configure_logging(
    *, level: str | None = None, env_file: str | Path | None = None
) -> None:
    """Configure concise Rich-backed console logging.

    Loads ``.env`` only when this function is called; it never configures ClearML
    or requires ClearML credentials.
    """
    if env_file is not None:
        load_dotenv(Path(env_file), override=False)
    chosen_level = (level or os.getenv("TAEI_LOG_LEVEL", "INFO")).upper()
    logger.remove()
    console = Console(stderr=True, markup=False, highlight=False)
    logger.add(
        lambda message: console.print(message, end=""),
        level=chosen_level,
        format="{level: <8} | {message}",
        colorize=False,
        enqueue=False,
        catch=True,
        backtrace=False,
        diagnose=False,
    )
