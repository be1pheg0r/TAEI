"""Модель A для prefill-шага: ``T = a·Σ L(1−R) + b·Σ L²(1−R²) + c``.

Σ берётся по запросам шага: ``L(1−R) = L − C`` — новые токены запроса,
``L²(1−R²) = L² − C²``. Коэффициенты a, b, c >= 0. Время prefill прогона —
сумма предсказаний по его prefill-шагам.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fit import fit_nonneg

FEATURES = ("new", "quad")


def design(steps: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            steps["new"].to_numpy(float),
            steps["quad"].to_numpy(float),
            np.ones(len(steps)),
        ]
    )


@dataclass
class PrefillModel:
    coef: dict[str, float] = field(default_factory=dict)

    def fit(self, steps: pd.DataFrame, method: str, **kwargs) -> PrefillModel:
        a, b, c = fit_nonneg(design(steps), steps["T"], method, **kwargs)
        self.coef = {"a": float(a), "b": float(b), "c": float(c)}
        return self

    def predict_steps(self, steps: pd.DataFrame) -> np.ndarray:
        c = self.coef
        return design(steps) @ np.array([c["a"], c["b"], c["c"]])

    def predict_runs(self, steps: pd.DataFrame) -> pd.Series:
        """T̂_prefill per run: sum of step predictions."""
        pred = pd.Series(self.predict_steps(steps), index=steps.index)
        return pred.groupby(steps["run_id"]).sum()
