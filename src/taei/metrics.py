from collections.abc import Mapping
from typing import Any

from loguru import logger


def log_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Log each metric as ``metric key=value`` and return a plain mapping."""
    result = dict(metrics)
    for key, value in result.items():
        logger.info("metric {}={}", key, value)
    return result
