"""Способ multiplicative: T̂_total = T̂_prefill · k(p_decode).

По каждому train-прогону ``k_j = wall_seconds_j / T̂_prefill_j`` даёт точку
(p_decode_j, k_j). Точки с одинаковым p объединяются медианой k и сортируются
по p; k(p) — линейная интерполяция между соседними точками, вне диапазона —
крайнее значение, и такие прогоны помечаются. Для диагностики строится вторая
кривая — k по измеренному prefill (``wall_seconds / сумма измеренных
prefill-шагов``), чтобы видеть, не прячет ли k ошибку модели A.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Dataset
from ..prefill import PrefillModel


def k_points(p: pd.Series, k: pd.Series) -> pd.DataFrame:
    """Median k per distinct p, sorted by p."""
    frame = pd.DataFrame({"p": p.to_numpy(float), "k": k.to_numpy(float)})
    return frame.groupby("p", as_index=False)["k"].median().sort_values("p")


def interpolate(points: pd.DataFrame, p) -> np.ndarray:
    return np.interp(np.asarray(p, dtype=float), points["p"], points["k"])


class Multiplicative:
    name = "multiplicative"

    def __init__(self) -> None:
        self.points = pd.DataFrame(columns=["p", "k"])
        self.points_measured = pd.DataFrame(columns=["p", "k"])
        self.prefill: PrefillModel | None = None

    def fit(
        self,
        data: Dataset,
        train_runs,
        prefill: PrefillModel,
        fit_method: str,
        **fit_kwargs,
    ) -> Multiplicative:
        # k is built from points, not fitted; fit_method applies to model A only.
        runs = data.runs.set_index("run_id").loc[sorted(train_runs)]
        pf = data.prefill_steps[data.prefill_steps["run_id"].isin(train_runs)]
        t_prefill = prefill.predict_runs(pf).reindex(runs.index)
        self.points = k_points(runs["p_decode"], runs["wall_seconds"] / t_prefill)
        self.points_measured = k_points(
            runs["p_decode"], runs["wall_seconds"] / runs["prefill_seconds"]
        )
        self.prefill = prefill
        return self

    def k(self, p) -> np.ndarray:
        return interpolate(self.points, p)

    def predict_total(self, data: Dataset, run_ids) -> pd.Series:
        runs = data.runs.set_index("run_id").loc[sorted(run_ids)]
        pf = data.prefill_steps[data.prefill_steps["run_id"].isin(run_ids)]
        t_prefill = self.prefill.predict_runs(pf).reindex(runs.index)
        return t_prefill * self.k(runs["p_decode"])

    def diagnostics(self, data: Dataset, run_ids) -> dict[str, pd.Series]:
        """Measured prefill time times k built from measured prefill."""
        runs = data.runs.set_index("run_id").loc[sorted(run_ids)]
        pred = runs["prefill_seconds"] * interpolate(
            self.points_measured, runs["p_decode"]
        )
        return {"measured prefill x k_meas": pred}

    def curves(self, data: Dataset, run_ids) -> list[dict]:
        """k(p) curves with train points and the given runs overlaid."""
        runs = data.runs.set_index("run_id").loc[sorted(run_ids)]
        pf = data.prefill_steps[data.prefill_steps["run_id"].isin(run_ids)]
        t_hat = self.prefill.predict_runs(pf).reindex(runs.index)
        overlay_a = runs["wall_seconds"] / t_hat
        overlay_m = runs["wall_seconds"] / runs["prefill_seconds"]
        return [
            {
                "name": "k by model A (wall / T_prefill_hat)",
                "x_label": "p_decode",
                "y_label": "k",
                "points": self.points.rename(columns={"p": "x", "k": "y"}).to_dict(
                    orient="records"
                ),
                "overlay": [
                    {"x": float(p), "y": float(k), "run_id": r}
                    for r, p, k in zip(
                        runs.index, runs["p_decode"], overlay_a, strict=True
                    )
                ],
            },
            {
                "name": "k by measured prefill (diagnostic)",
                "x_label": "p_decode",
                "y_label": "k",
                "points": self.points_measured.rename(
                    columns={"p": "x", "k": "y"}
                ).to_dict(orient="records"),
                "overlay": [
                    {"x": float(p), "y": float(k), "run_id": r}
                    for r, p, k in zip(
                        runs.index, runs["p_decode"], overlay_m, strict=True
                    )
                ],
            },
        ]

    def flags(self, data: Dataset, run_ids) -> pd.Series:
        runs = data.runs.set_index("run_id").loc[sorted(run_ids)]
        lo, hi = self.points["p"].min(), self.points["p"].max()
        out = np.where(
            runs["p_decode"] < lo,
            "p_decode below k range",
            np.where(runs["p_decode"] > hi, "p_decode above k range", ""),
        )
        return pd.Series(out, index=runs.index)

    def describe(self) -> dict:
        return {
            "formula": "T_total = T_prefill_hat * k(p_decode), "
            "k linear interpolation between train points",
            "coef": {
                "k_points": len(self.points),
                "p_min": float(self.points["p"].min()),
                "p_max": float(self.points["p"].max()),
            },
            "k_points": self.points.to_dict(orient="records"),
            "k_points_measured_prefill": self.points_measured.to_dict(orient="records"),
        }


METHOD = Multiplicative
