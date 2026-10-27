"""Decode-шаг: ``T = d0 + d1·Σℓ + d2·B``; время прогона — сумма всех шагов.

Коэффициенты d0, d1, d2 >= 0 подбираются по decode-шагам train-прогонов.
T̂_total прогона — сумма предсказаний модели A по его prefill-шагам и этой
формы по decode-шагам.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fit import fit_nonneg
from .prefill import PrefillModel

LATEX = [
    r"T_{\mathrm{dc}} = d_0 + d_1 \sum_{i \in \mathrm{step}} \ell_i + d_2 B, "
    r"\qquad d_0, d_1, d_2 \ge 0",
]


def design(steps: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(steps)),
            steps["sum_l"].to_numpy(float),
            steps["B"].to_numpy(float),
        ]
    )


@dataclass
class DecodeModel:
    coef: dict[str, float] = field(default_factory=dict)

    def fit(self, steps: pd.DataFrame, method: str, **kwargs) -> DecodeModel:
        d0, d1, d2 = fit_nonneg(design(steps), steps["T"], method, **kwargs)
        self.coef = {"d0": float(d0), "d1": float(d1), "d2": float(d2)}
        return self

    def predict_steps(self, steps: pd.DataFrame) -> np.ndarray:
        c = self.coef
        return design(steps) @ np.array([c["d0"], c["d1"], c["d2"]])


def predict_total(
    prefill: PrefillModel, decode: DecodeModel, data, run_ids
) -> pd.Series:
    """T̂_total per run: model A over prefill steps plus decode steps."""
    pf = data.prefill_steps[data.prefill_steps["run_id"].isin(run_ids)]
    dc = data.decode_steps[data.decode_steps["run_id"].isin(run_ids)]
    total = prefill.predict_runs(pf)
    if len(dc):
        steps = pd.Series(decode.predict_steps(dc), index=dc.index)
        total = total.add(steps.groupby(dc["run_id"]).sum(), fill_value=0.0)
    return total.reindex(sorted(run_ids))
