"""Три варианта decode-шага и цена общего префикса κ (S7).

- (а) ``base``: ``T = d0 + d1·Σℓ + d2·B`` — форма ядра (``decode.DecodeModel``);
- (б) ``floor``: ``T = max(F, d0 + d1·Σℓ + d2·B)`` — шаг не короче порога F;
- (в) ``kappa``: ``T = max(F, d0 + d1·(U + κ·(Σℓ − U)) + d2·B)``, U — токены KV
  шага без повторов общего префикса (``data.unique_tokens``). κ = 1 — decode
  платит за Σℓ, κ = 0 — только за уникальные токены.

d0, d1, d2 и F подбираются на decode-шагах без общих блоков (S2 + S6): F — по
сетке квантилей T, d — ``fit_nonneg`` (NNLS на относительной ошибке) на шагах,
где T выше F с запасом; из сетки берётся F с наименьшей суммой квадратов
относительной ошибки ``max(F, ·)`` на всех шагах. κ подбирается по S7 при этих
d и F, только по попыткам, где T заметно выше порога и повторы — заметная доля
Σℓ. Первые ``warmup_steps`` decode-шагов попытки (прогрев точки плана) в подбор
не идут.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .fit import fit_nonneg, relative_errors, summary
from .split import series_of

VARIANTS = ("base", "floor", "kappa")
LABELS = {
    "base": "(а) база",
    "floor": "(б) с порогом",
    "kappa": "(в) с порогом и κ",
}
LATEX = {
    "base": r"T_{\mathrm{dc}} = d_0 + d_1 \sum \ell_i + d_2 B",
    "floor": r"T_{\mathrm{dc}} = \max\bigl(F,\ d_0 + d_1 \sum \ell_i + d_2 B\bigr)",
    "kappa": r"T_{\mathrm{dc}} = \max\bigl(F,\ d_0 + d_1 \bigl(U + \kappa "
    r"(\textstyle\sum \ell_i - U)\bigr) + d_2 B\bigr)",
}
FLOOR_QUANTILES = np.linspace(0.0, 0.5, 51)
FLOOR_MARGIN = 1.2  # steps with T above (1 + 0.2)·F carry the linear part
MIN_SHARED = 0.2  # (Σℓ − U)/Σℓ of a κ point


@dataclass
class DecodeVariant:
    name: str
    coef: dict[str, float] = field(default_factory=dict)

    def linear(self, B, sum_l, sum_u) -> np.ndarray:
        c = self.coef
        eff = np.asarray(sum_u, float) + c["kappa"] * (
            np.asarray(sum_l, float) - np.asarray(sum_u, float)
        )
        return c["d0"] + c["d1"] * eff + c["d2"] * np.asarray(B, float)

    def predict(self, B, sum_l, sum_u=None) -> np.ndarray:
        """Step time; without ``sum_u`` there are no shared blocks (U = Σℓ)."""
        u = sum_l if sum_u is None else sum_u
        return np.maximum(self.coef["F"], self.linear(B, sum_l, u))

    def predict_steps(self, steps: pd.DataFrame) -> np.ndarray:
        u = steps["sum_u"] if "sum_u" in steps else steps["sum_l"]
        return self.predict(steps["B"], steps["sum_l"], u)


def _design(steps: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(steps)),
            steps["sum_l"].to_numpy(float),
            steps["B"].to_numpy(float),
        ]
    )


def _base(steps: pd.DataFrame) -> dict[str, float]:
    d0, d1, d2 = fit_nonneg(_design(steps), steps["T"], "regression")
    return {"d0": float(d0), "d1": float(d1), "d2": float(d2), "F": 0.0, "kappa": 1.0}


def _floor(steps: pd.DataFrame) -> dict[str, float]:
    T = steps["T"].to_numpy(float)
    best = None
    for F in [0.0, *np.quantile(T, FLOOR_QUANTILES[1:])]:
        above = steps[T > FLOOR_MARGIN * F]
        if len(above) < 10:
            continue
        coef = {**_base(above), "F": float(F)}
        rel = relative_errors(DecodeVariant("floor", coef).predict_steps(steps), T)
        loss = float((rel**2).sum())
        if best is None or loss < best[0]:
            best = (loss, coef)
    return best[1]


def kappa_points(steps: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """One row per attempt: medians of its decode steps and its plan point.

    ``points`` — run_id, ell, R_target, families (from the plan). The actual
    reuse is already in U (the requests' C), so R = 0.999 at ℓ = 1025 counts
    as it is.
    """
    agg = (
        steps.groupby("run_id")
        .agg(
            T=("T", "median"),
            B=("B", "max"),
            sum_l=("sum_l", "median"),
            sum_u=("sum_u", "median"),
        )
        .reset_index()
        .merge(points, on="run_id")
    )
    agg["share"] = 1 - agg["sum_u"] / agg["sum_l"]
    return agg


def _fit_kappa(rows: pd.DataFrame, coef: dict) -> float:
    def loss(k):
        v = DecodeVariant("kappa", {**coef, "kappa": float(k)})
        rel = relative_errors(
            v.predict(rows["B"], rows["sum_l"], rows["sum_u"]), rows["T"]
        )
        return float((rel**2).sum())

    return float(minimize_scalar(loss, bounds=(-1.0, 3.0), method="bounded").x)


def solve_kappa(row, coef: dict) -> float:
    """κ of one attempt from its T, with d and F fixed (T above the floor)."""
    d0, d1, d2 = coef["d0"], coef["d1"], coef["d2"]
    return ((row["T"] - d0 - d2 * row["B"]) / d1 - row["sum_u"]) / (
        row["sum_l"] - row["sum_u"]
    )


def drop_warmup(steps: pd.DataFrame, warmup: dict[str, int]) -> pd.DataFrame:
    """Decode steps without the first ``warmup[run_id]`` steps of each run."""
    rank = steps.sort_values(["run_id", "step_index"]).groupby("run_id").cumcount()
    keep = rank.reindex(steps.index) >= steps["run_id"].map(warmup).fillna(0)
    return steps[keep]


@dataclass
class Variants:
    variants: dict[str, DecodeVariant]
    kappa_rows: pd.DataFrame  # S7 attempts used for κ, with κ per attempt
    kappa_all: pd.DataFrame  # every S7 attempt with shared blocks
    errors: list[dict]  # decode-step errors per variant × series
    choice: str


def fit_all(
    decode_steps: pd.DataFrame,
    points: pd.DataFrame,
    warmup: dict[str, int],
    train=("S2", "S6"),
    kappa_series="S7",
    train_series=("S1", "S2", "S6", "S7"),
    test_series=("S3",),
) -> Variants:
    """Fit (а), (б), (в) and choose one by the step error on train and test S3.

    ``decode_steps`` — decode steps of the step-model source (``ev.decode_steps``
    of the train model, any part): S2, S3, S6 and S7 are taken by series.
    """
    dc = drop_warmup(decode_steps, warmup).assign(
        series=lambda f: f["run_id"].map(series_of)
    )
    fit_rows = dc[dc["series"].isin(train)]
    base = _base(fit_rows)
    floor = _floor(fit_rows)
    s7 = kappa_points(dc[dc["series"] == kappa_series], points)
    s7 = s7[s7["share"] > 0].copy()
    s7["kappa_attempt"] = s7.apply(solve_kappa, axis=1, coef=floor)
    sel = s7[(s7["T"] >= FLOOR_MARGIN * floor["F"]) & (s7["share"] >= MIN_SHARED)]
    kappa = _fit_kappa(sel, floor)
    variants = {
        "base": DecodeVariant("base", base),
        "floor": DecodeVariant("floor", floor),
        "kappa": DecodeVariant("kappa", {**floor, "kappa": kappa}),
    }
    errors = []
    for name, v in variants.items():
        rel = pd.Series(relative_errors(v.predict_steps(dc), dc["T"]), index=dc.index)
        for series, idx in dc.groupby("series").groups.items():
            part = (
                "train"
                if series in train_series
                else "test"
                if series in test_series
                else None
            )
            if part:
                errors.append(
                    {
                        "variant": name,
                        "series": series,
                        "part": part,
                        **summary(rel[idx]),
                    }
                )
    frame = pd.DataFrame(errors)
    score = (
        frame[frame["part"] == "test"].groupby("variant")["mean_abs"].mean()
        + 1e-3 * frame[frame["part"] == "train"].groupby("variant")["mean_abs"].mean()
    )
    return Variants(variants, sel, s7, errors, str(score.idxmin()))


def kappa_table(rows: pd.DataFrame) -> list[dict]:
    """κ by factor: per point (mean over seeds), then median and quartiles."""
    rows = rows.assign(per_family=rows["B"] / rows["families"].clip(lower=1))
    keys = ["B", "ell", "R_target", "families"]
    pts = rows.groupby(keys).agg(
        kappa=("kappa_attempt", "mean"), per_family=("per_family", "first")
    )
    pts = pts.reset_index()
    out = []
    for factor in [*keys, "per_family"]:
        for value, g in pts.groupby(factor):
            out.append(
                {
                    "factor": factor,
                    "value": float(value),
                    "points": len(g),
                    "median": float(g["kappa"].median()),
                    "q1": float(g["kappa"].quantile(0.25)),
                    "q3": float(g["kappa"].quantile(0.75)),
                }
            )
    return out
