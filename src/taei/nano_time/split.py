"""Разбиения прогонов на train и test.

Основное разбиение — 50/50 внутри каждой страты (workload ``set_id`` × кэш
вкл/выкл) с фиксированным seed; им пользуются все модели и все способы decode.
Дополнительное — только для модели A: train — прогоны без кэша, test — с кэшем.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratified(runs: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    train, test = [], []
    for _, group in runs.sort_values("run_id").groupby(["set_id", "cache"]):
        ids = group["run_id"].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        half = len(ids) // 2
        train += sorted(ids[:half].tolist())
        test += sorted(ids[half:].tolist())
    return {"train": sorted(train), "test": sorted(test)}


def cache_off_on(runs: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "train": sorted(runs.loc[runs["cache"] == "off", "run_id"]),
        "test": sorted(runs.loc[runs["cache"] == "on", "run_id"]),
    }
