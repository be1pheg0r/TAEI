"""Какая функция описывает время prefill-шага: сравнение форм и уровень шума.

Используются только prefill-шаги (каждый даёт один токен на запрос, decode в
них не входит). Кандидаты — линейные по признакам формы с коэффициентами >= 0
(подбор NNLS по относительной ошибке) и степенная ``c + a·N^α``; N = ΣL(1−R) —
новые токены шага. Проверка — leave-one-group-out по наборам (``set_id``):
форма оценивается на наборе, которого не видела. Нижняя граница ошибки берётся
из шагов с одинаковым составом запросов (одинаковые L и C) в разных прогонах:
их разброс не может объяснить никакая функция от состава шага.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

from .data import Dataset

# name, TeX, feature columns (a constant is always added)
LINEAR_FORMS = [
    ("linear", r"a N + c", ["N"]),
    ("model A", r"a N + b \sum (L^2 - C^2) + c", ["N", "Q"]),
    ("poly N²", r"a N + b N^2 + c", ["N", "N2"]),
    ("cache cost", r"a N + e \sum C + c", ["N", "sumC"]),
    ("per request", r"a N + d B + c", ["N", "B"]),
    ("A + per request", r"a N + b \sum (L^2 - C^2) + d B + c", ["N", "Q", "B"]),
]
POWER = ("power", r"c + a N^{\alpha}")
COEF_NAMES = {"N": "a", "Q": "b", "N2": "b", "sumC": "e", "B": "d"}


def step_table(data: Dataset) -> pd.DataFrame:
    reqs = data.prefill_requests
    content = (
        reqs.sort_values("request_id")
        .groupby(["run_id", "step_id"])
        .apply(
            lambda g: str(list(zip(g.L.astype(int), g.C.astype(int), strict=True))),
            include_groups=False,
        )
        .rename("content")
    )
    sums = reqs.groupby(["run_id", "step_id"]).agg(sumC=("C", "sum"), sumL=("L", "sum"))
    steps = (
        data.prefill_steps.merge(sums.reset_index(), on=["run_id", "step_id"])
        .merge(content.reset_index(), on=["run_id", "step_id"])
        .merge(data.runs[["run_id", "set_id", "cache", "scheduler"]], on="run_id")
    )
    steps["N"], steps["Q"] = steps["new"], steps["quad"]
    steps["N2"] = steps["N"] ** 2
    steps["R_step"] = steps["sumC"] / steps["sumL"]
    return steps


def _fit_linear(steps, cols):
    X = np.column_stack(
        [steps[c].to_numpy(float) for c in cols] + [np.ones(len(steps))]
    )
    y = steps["T"].to_numpy(float)
    coef, _ = nnls(X / y[:, None], np.ones(len(y)))

    def predict(frame):
        Z = np.column_stack(
            [frame[c].to_numpy(float) for c in cols] + [np.ones(len(frame))]
        )
        return Z @ coef

    names = [COEF_NAMES[c] for c in cols] + ["c"]
    return predict, dict(zip(names, map(float, coef), strict=True))


def _fit_power(steps):
    N = steps["N"].to_numpy(float)
    y = steps["T"].to_numpy(float)

    def residuals(t):
        return (np.exp(t[0]) + np.exp(t[1]) * N ** t[2] - y) / y

    t = least_squares(residuals, [np.log(0.05), np.log(1e-4), 1.0]).x
    c, a, alpha = np.exp(t[0]), np.exp(t[1]), t[2]
    return (lambda f: c + a * f["N"].to_numpy(float) ** alpha), {
        "a": float(a),
        "alpha": float(alpha),
        "c": float(c),
    }


def analyze(data: Dataset) -> dict:
    steps = step_table(data)
    y = steps["T"].to_numpy(float)
    groups = steps["set_id"].to_numpy()
    forms = [(n, tex, cols) for n, tex, cols in LINEAR_FORMS] + [(*POWER, None)]
    table, fitted = [], {}
    for name, tex, cols in forms:
        fit = (lambda f, cols=cols: _fit_linear(f, cols)) if cols else _fit_power
        oof = np.full(len(steps), np.nan)
        for g in np.unique(groups):
            train, test = steps[groups != g], steps[groups == g]
            oof[groups == g] = fit(train)[0](test)
        predict, coef = fit(steps)
        in_sample = predict(steps)
        err_cv = np.abs(oof - y) / y
        k = len(coef)
        rss = float(np.sum(np.log(in_sample / y) ** 2))
        table.append(
            {
                "name": name,
                "latex": rf"\hat T = {tex}",
                "params": k,
                "cv_mean": float(err_cv.mean()),
                "cv_median": float(np.median(err_cv)),
                "cv_max": float(err_cv.max()),
                "in_sample_mean": float(np.mean(np.abs(in_sample - y) / y)),
                "aic_log": float(len(y) * np.log(rss / len(y)) + 2 * k),
                "coef": coef,
            }
        )
        fitted[name] = predict

    # Curves against N for forms that depend on N only.
    grid = pd.DataFrame({"N": np.linspace(steps["N"].min(), steps["N"].max(), 80)})
    grid["N2"] = grid["N"] ** 2
    curves = [
        {"name": name, "x": grid["N"].tolist(), "y": fitted[name](grid).tolist()}
        for name in ("linear", "poly N²", "power")
    ]

    # Noise floor: identical step content measured in different runs.
    dup = steps.groupby("content").filter(lambda g: len(g) > 1)
    mean_t = dup.groupby("content")["T"].transform("mean")
    dev = (dup["T"] / mean_t - 1).abs()
    pairs = [
        {
            "x": float(g["T"].iloc[0]),
            "y": float(g["T"].iloc[1]),
            "a": g["run_id"].iloc[0],
            "b": g["run_id"].iloc[1],
            "N": float(g["N"].iloc[0]),
        }
        for _, g in dup.groupby("content")
        if len(g) == 2
    ]
    best = min(table, key=lambda r: r["cv_mean"])
    log_res = np.log(y / fitted["linear"](steps))
    by_run = pd.Series(log_res).groupby(steps["run_id"].to_numpy())
    return {
        "table": sorted(table, key=lambda r: r["cv_mean"]),
        "best": best["name"],
        "points": steps[
            ["run_id", "step_id", "N", "T", "cache", "scheduler", "R_step", "B"]
        ].to_dict(orient="records"),
        "curves": curves,
        "pairs": pairs,
        "noise": {
            "identical_groups": int(dup["content"].nunique()),
            "identical_steps": int(len(dup)),
            "mean_abs_dev": float(dev.mean()),
            "median_abs_dev": float(dev.median()),
            "var_between_runs": float(by_run.mean().var()),
            "var_within_runs": float(by_run.var().mean()),
        },
        "n_steps": int(len(steps)),
        "n_groups": int(len(np.unique(groups))),
        "N_range": [float(steps["N"].min()), float(steps["N"].max())],
    }
