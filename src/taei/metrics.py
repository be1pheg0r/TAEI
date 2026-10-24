"""Нормализует метрики и выводит компактную сводку вместо полного дампа.

Публичная функция принимает отображение, копирует его в обычный словарь,
передает краткую сводку в Loguru на INFO, а полные пары — на DEBUG и
возвращает словарь для последующего сохранения или передачи вызывающему коду.
Модуль не проверяет семантику метрик и не управляет жизненным циклом эксперимента
или задачи ClearML.
"""

from collections.abc import Mapping
from typing import Any

from loguru import logger


def log_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Log a compact scalar summary and return every metric unchanged."""
    result = dict(metrics)
    scalar_items = [
        (key, value)
        for key, value in result.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    ]
    visible = scalar_items[:5]
    summary = ", ".join(f"{key}={value}" for key, value in visible)
    if len(scalar_items) > len(visible):
        summary += f", +{len(scalar_items) - len(visible)} more"
    logger.info("metrics: {}", summary or f"{len(result)} values saved")
    for key, value in result.items():
        logger.debug("metric {}={}", key, value)
    return result
