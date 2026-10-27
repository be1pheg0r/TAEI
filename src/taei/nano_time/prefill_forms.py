"""Какая функция описывает время prefill-шага: формы, множитель прогона и R.

Используются только prefill-шаги: decode в них не входит. Устройство данных: в
каждом наборе 8 длин × пара hit/miss с одинаковыми token ID, в прогонах без кэша
те же запросы идут с C = 0 от чужого префикса. У 0.6B в один шаг попадают hit и
miss вперемешку, поэтому R шага узкий, а R отдельного запроса (0,09…0,998)
входит в форму линейно через ΣC; у 8B long context C > 0 только у вторых кусков
chunked prefill (свой готовый KV). Время шага домножается на множитель прогона
(состояние общей машины: частота, чужие задачи), поэтому коэффициенты формы
подбираются с отдельным множителем на прогон: форма определяется сравнениями
шагов внутри прогона.

Оценка формы: (1) с множителем прогона — каждый шаг предсказывается по форме,
подобранной без него, и по множителю, оценённому на остальных шагах того же
прогона; (2) без множителя — leave-one-set-out, как будет у нового прогона.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

from .data import Dataset

# name, TeX, feature columns ("one" is a constant term)
FORMS = [
    ("new tokens", r"a N", ["N"]),
    ("new + const", r"a N + c", ["N", "one"]),
    ("model A", r"a N + b \sum (L^2 - C^2)", ["N", "Q"]),
    ("poly N²", r"a N + b N^2", ["N", "N2"]),
    ("cache cost", r"a N + e \sum C", ["N", "sumC"]),
    ("cache cost + per request", r"a N + e \sum C + d B", ["N", "sumC", "B"]),
    (
        "A + cache cost + per request",
        r"a N + e \sum C + b \sum (L^2 - C^2) + d B",
        ["N", "sumC", "Q", "B"],
    ),
    ("total tokens", r"a \sum L + c", ["sumL", "one"]),
]
POWER = ("power", r"c + a N^{\alpha}")
COEF_NAMES = {
    "N": "a",
    "Q": "b",
    "N2": "b",
    "sumC": "e",
    "B": "d",
    "sumL": "a",
    "one": "c",
}


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
    sums = (
        reqs.assign(L2=reqs["L"] ** 2)
        .groupby(["run_id", "step_id"])
        .agg(sumC=("C", "sum"), sumL=("L", "sum"), sumL2=("L2", "sum"))
    )
    steps = (
        data.prefill_steps.merge(sums.reset_index(), on=["run_id", "step_id"])
        .merge(content.reset_index(), on=["run_id", "step_id"])
        .merge(data.runs[["run_id", "set_id", "cache", "scheduler"]], on="run_id")
    )
    steps["N"], steps["Q"] = steps["new"], steps["quad"]
    steps["N2"] = steps["N"] ** 2
    steps["one"] = 1.0
    steps["R_step"] = steps["sumC"] / steps["sumL"]
    return steps


def fit_with_run_factor(steps: pd.DataFrame, cols, iters: int = 40):
    """NNLS shape coefficients times a factor per run (geometric mean factor = 1)."""
    X = steps[cols].to_numpy(float)
    y = steps["T"].to_numpy(float)
    runs = steps["run_id"].to_numpy()
    u = np.zeros(len(steps))
    coef = np.zeros(len(cols))
    for _ in range(iters):
        scaled = y * np.exp(-u)
        coef, _ = nnls(X / scaled[:, None], np.ones(len(steps)))
        pred = X @ coef
        if not np.all(pred > 0):
            break
        mean_log = pd.Series(np.log(y / pred)).groupby(runs).transform("mean")
        u = (mean_log - mean_log.mean()).to_numpy()
    factors = pd.Series(np.exp(u), index=runs).groupby(level=0).first()
    return coef, factors


def _fit_pooled(steps, cols):
    X = steps[cols].to_numpy(float)
    y = steps["T"].to_numpy(float)
    coef, _ = nnls(X / y[:, None], np.ones(len(y)))
    return coef


def _coef_dict(cols, coef):
    return {COEF_NAMES[c]: float(v) for c, v in zip(cols, coef, strict=True)}


def _within_run_loo(steps, cols) -> np.ndarray:
    """Each step from a shape fitted without it and its run's other steps."""
    errors = []
    for i in range(len(steps)):
        run = steps["run_id"].iat[i]
        others = steps.drop(index=steps.index[i])
        mine = others["run_id"] == run
        if not mine.any():
            continue
        coef, _ = fit_with_run_factor(others, cols, iters=15)
        own = others.loc[mine]
        ratio = own["T"].to_numpy(float) / (own[cols].to_numpy(float) @ coef)
        pred = (steps[cols].iloc[i].to_numpy(float) @ coef) * np.exp(
            np.log(ratio).mean()
        )
        errors.append(abs(pred - steps["T"].iat[i]) / steps["T"].iat[i])
    return np.asarray(errors)


def _set_cv(steps, cols) -> np.ndarray:
    groups = steps["set_id"].to_numpy()
    oof = np.full(len(steps), np.nan)
    for g in np.unique(groups):
        coef = _fit_pooled(steps[groups != g], cols)
        oof[groups == g] = steps.loc[groups == g, cols].to_numpy(float) @ coef
    y = steps["T"].to_numpy(float)
    return np.abs(oof - y) / y


def _power_set_cv(steps) -> tuple[np.ndarray, dict]:
    def fit(frame):
        N, y = frame["N"].to_numpy(float), frame["T"].to_numpy(float)
        t = least_squares(
            lambda t: (np.exp(t[0]) + np.exp(t[1]) * N ** t[2] - y) / y,
            [np.log(0.05), np.log(1e-4), 1.0],
        ).x
        return np.exp(t[0]), np.exp(t[1]), t[2]

    groups = steps["set_id"].to_numpy()
    oof = np.full(len(steps), np.nan)
    for g in np.unique(groups):
        c, a, alpha = fit(steps[groups != g])
        oof[groups == g] = c + a * steps.loc[groups == g, "N"].to_numpy(float) ** alpha
    c, a, alpha = fit(steps)
    y = steps["T"].to_numpy(float)
    return np.abs(oof - y) / y, {"a": float(a), "alpha": float(alpha), "c": float(c)}


def analyze(data: Dataset) -> dict:
    steps = step_table(data)
    table = []
    for name, tex, cols in FORMS:
        coef, _ = fit_with_run_factor(steps, cols)
        in_run = _within_run_loo(steps, cols)
        pooled = _set_cv(steps, cols)
        table.append(
            {
                "name": name,
                "latex": rf"\hat T = {tex}",
                "params": len(cols),
                # None when no run has a second prefill step to compare with.
                "in_run_mean": float(in_run.mean()) if len(in_run) else None,
                "in_run_median": float(np.median(in_run)) if len(in_run) else None,
                "in_run_max": float(in_run.max()) if len(in_run) else None,
                "set_cv_mean": float(pooled.mean()),
                "coef": _coef_dict(cols, coef),
            }
        )
    pooled, power_coef = _power_set_cv(steps)
    table.append(
        {
            "name": POWER[0],
            "latex": rf"\hat T = {POWER[1]}",
            "params": 3,
            "in_run_mean": None,
            "in_run_median": None,
            "in_run_max": None,
            "set_cv_mean": float(pooled.mean()),
            "coef": power_coef,
        }
    )
    ranked = [r for r in table if r["in_run_mean"] is not None]
    best = (
        min(ranked, key=lambda r: r["in_run_mean"])
        if ranked
        else min(table[:-1], key=lambda r: r["set_cv_mean"])
    )
    best_cols = next(cols for name, _, cols in FORMS if name == best["name"])
    coef, factors = fit_with_run_factor(steps, best_cols)
    c = _coef_dict(best_cols, coef)

    # R dependence: step time with the run factor and per-request overhead
    # removed, relative to the same requests with nothing cached; the curve is
    # the best form for one request of the median hit length L_ref:
    # (a L (1-R) + e L R + b L^2 (1-R^2)) / (a L + b L^2).
    a = c.get("a", np.nan)
    e = c.get("e", 0.0)
    b = c.get("b", 0.0) if "Q" in best_cols else 0.0
    d = c.get("d", 0.0)
    steps["T_norm"] = steps["T"] / steps["run_id"].map(factors).to_numpy()
    full = a * steps["sumL"] + b * steps["sumL2"]
    steps["per_token"] = (steps["T_norm"] - d * steps["B"]) / full
    r_grid = np.linspace(0, 1, 51)
    reqs = data.prefill_requests
    hits = reqs.loc[reqs["C"] > 0, "R"]
    l_ref = float(reqs.loc[reqs["C"] > 0, "L"].median()) if len(hits) else 2000.0
    curve = (
        a * l_ref * (1 - r_grid) + e * l_ref * r_grid + b * l_ref**2 * (1 - r_grid**2)
    ) / (a * l_ref + b * l_ref**2)
    on = steps["sumC"] > 0

    # Noise floor: identical step content measured in different runs.
    dup = steps.groupby("content").filter(lambda g: len(g) > 1)
    dev = (dup["T"] / dup.groupby("content")["T"].transform("mean") - 1).abs()
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
    return {
        "table": sorted(
            table,
            key=lambda r: (
                r["in_run_mean"] is None,
                r["in_run_mean"] or r["set_cv_mean"],
            ),
        ),
        "best": best["name"],
        "best_latex": best["latex"],
        "best_coef": c,
        "cached_to_new_cost": float(e / a) if a else None,
        "run_factor": {
            "p05": float(np.percentile(factors, 5)),
            "median": float(np.median(factors)),
            "p95": float(np.percentile(factors, 95)),
            "values": [{"run_id": k, "factor": float(v)} for k, v in factors.items()],
        },
        "points": steps[
            [
                "run_id",
                "step_id",
                "N",
                "T",
                "T_norm",
                "per_token",
                "cache",
                "scheduler",
                "R_step",
                "B",
                "sumL",
                "sumC",
            ]
        ].to_dict(orient="records"),
        "r_curve": {"x": r_grid.tolist(), "y": curve.tolist(), "L_ref": l_ref},
        "request_R": {
            "hit_count": int(len(hits)),
            "hit_min": float(hits.min()) if len(hits) else None,
            "hit_max": float(hits.max()) if len(hits) else None,
            "zero_count": int((reqs["C"] == 0).sum()),
            "step_R_min_on": float(steps.loc[on, "R_step"].min()) if on.any() else None,
            "step_R_max_on": float(steps.loc[on, "R_step"].max()) if on.any() else None,
        },
        "pairs": pairs,
        "noise": {
            "identical_groups": int(dup["content"].nunique()),
            "mean_abs_dev": float(dev.mean()) if len(dev) else None,
        },
        "n_steps": int(len(steps)),
        "n_runs": int(steps["run_id"].nunique()),
        "n_groups": int(steps["set_id"].nunique()),
    }
