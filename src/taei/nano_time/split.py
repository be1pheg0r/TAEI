"""Разбиения прогонов на train и test.

Основное разбиение — 50/50 внутри каждой страты с фиксированным seed; страта
задаётся колонками прогона (по умолчанию workload ``set_id`` × кэш вкл/выкл, у
8B long context — workload × условие). У локальной сетки разбиение задано
сериями: train — контролируемые сетки, test — смеси (``by_series``).
Дополнительное — только для модели A:
train — прогоны без кэша, test — с кэшем; если прогонов с кэшем нет, его нет.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_KEYS = ("set_id", "cache")


def stratified(
    runs: pd.DataFrame, seed: int, keys=DEFAULT_KEYS
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    train, test = [], []
    for _, group in runs.sort_values("run_id").groupby(list(keys)):
        ids = group["run_id"].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        half = len(ids) // 2
        train += sorted(ids[:half].tolist())
        test += sorted(ids[half:].tolist())
    return {"train": sorted(train), "test": sorted(test)}


def cache_off_on(runs: pd.DataFrame) -> dict[str, list[str]] | None:
    on = sorted(runs.loc[runs["cache"] == "on", "run_id"])
    off = sorted(runs.loc[runs["cache"] == "off", "run_id"])
    return {"train": off, "test": on} if on and off else None


def series_of(run_id: str) -> str:
    """Series of a local-grid run id ``<phase>-<series>-<n>``, else the id."""
    parts = run_id.split("-")
    return parts[1] if len(parts) == 3 else run_id


def by_series(runs: pd.DataFrame, parts: dict[str, list[str]]) -> dict[str, list[str]]:
    """Explicit split: each part is the runs of the listed series."""
    series = runs["run_id"].map(series_of)
    return {
        part: sorted(runs.loc[series.isin(names), "run_id"])
        for part, names in parts.items()
    }
