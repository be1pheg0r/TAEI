"""Нормализует метрики эксперимента и записывает каждую отдельной строкой лога.

Публичная функция принимает отображение, копирует его в обычный словарь,
передает каждую пару ключ/значение в Loguru в формате ``metric key=value`` и
возвращает словарь для последующего сохранения или передачи вызывающему коду.
Модуль не проверяет семантику метрик и не управляет жизненным циклом эксперимента
или задачи ClearML.
"""

from collections.abc import Mapping
from typing import Any

from loguru import logger


def log_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Log each metric as ``metric key=value`` and return a plain mapping."""
    result = dict(metrics)
    for key, value in result.items():
        logger.info("metric {}={}", key, value)
    return result
