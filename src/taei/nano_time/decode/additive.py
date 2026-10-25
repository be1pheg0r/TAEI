"""Способ additive: T̂_total = T̂_prefill + Σ предсказаний decode-шагов.

Decode-шаг: ``T = d0 + d1·Σℓ + d2·B``, все коэффициенты >= 0, подбор по
decode-шагам train-прогонов.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Dataset
from ..fit import fit_nonneg
from ..prefill import PrefillModel


def design(steps: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(steps)),
            steps["sum_l"].to_numpy(float),
            steps["B"].to_numpy(float),
        ]
    )


class Additive:
    name = "additive"

    def __init__(self) -> None:
        self.coef: dict[str, float] = {}
        self.prefill: PrefillModel | None = None

    def fit(
        self,
        data: Dataset,
        train_runs,
        prefill: PrefillModel,
        fit_method: str,
        **fit_kwargs,
    ) -> Additive:
        steps = data.decode_steps[data.decode_steps["run_id"].isin(train_runs)]
        d0, d1, d2 = fit_nonneg(design(steps), steps["T"], fit_method, **fit_kwargs)
        self.coef = {"d0": float(d0), "d1": float(d1), "d2": float(d2)}
        self.prefill = prefill
        return self

    def predict_steps(self, steps: pd.DataFrame) -> np.ndarray:
        c = self.coef
        return design(steps) @ np.array([c["d0"], c["d1"], c["d2"]])

    def predict_total(self, data: Dataset, run_ids) -> pd.Series:
        pf = data.prefill_steps[data.prefill_steps["run_id"].isin(run_ids)]
        dc = data.decode_steps[data.decode_steps["run_id"].isin(run_ids)]
        prefill = self.prefill.predict_runs(pf)
        decode = (
            pd.Series(self.predict_steps(dc), index=dc.index)
            .groupby(dc["run_id"])
            .sum()
        )
        total = prefill.add(decode, fill_value=0.0)
        return total.reindex(sorted(run_ids))

    def flags(self, data: Dataset, run_ids) -> pd.Series:
        return pd.Series("", index=sorted(run_ids))

    def describe(self) -> dict:
        return {
            "formula": "T_total = T_prefill_hat + sum over decode steps of "
            "(d0 + d1*sum_l + d2*B)",
            "coef": self.coef,
            "summary": r"Время прогона — сумма предсказаний по всем шагам: prefill по "
            r"модели A, каждый decode-шаг — линейно по \(\sum\ell\) и \(B\).",
            "latex": [
                r"\hat T^{\,\mathrm{decode}}_{\mathrm{step}} = d_0 + d_1 \sum_{i \in "
                r"\mathrm{step}} \ell_i + d_2 B, \qquad d_0, d_1, d_2 \ge 0",
                r"\hat T_{\mathrm{total}} = \hat T_{\mathrm{prefill}} + \sum_{s \in "
                r"\mathrm{decode}} \hat T^{\,\mathrm{decode}}_{s}",
            ],
        }


METHOD = Additive
