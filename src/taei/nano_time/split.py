"""Разбиения прогонов на train и test.

Основное разбиение — 50/50 внутри каждой страты с фиксированным seed; страта
задаётся колонками прогона (по умолчанию workload ``set_id`` × кэш вкл/выкл, у
8B long context — workload × условие). Дополнительное — только для модели A:
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
