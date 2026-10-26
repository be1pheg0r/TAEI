"""Данные отчёта в фиксированном формате (``time_model.json``, версия 1).

Отчёт строится только из этого словаря, поэтому новая модель, модель переноса
или способ decode появляется в отчёте без правки шаблона: разделы строятся
циклом по ``methods``, ``sources`` и ``curves``. Таблицы шагов урезаны для
графиков (decode — случайная выборка), а статистики по корзинам считаются по
полным данным. Все числа — обычные JSON-числа или ``null``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from .evaluate import TRAIN_SOURCE, Evaluation
from .fit import LATEX as FIT_LATEX
from .prefill import LATEX as PREFILL_LATEX
from .prefill_forms import analyze as analyze_prefill_forms

# Notation shared by every model in the report (inline TeX).
NOTATION = [
    [r"L", "токены запроса после prefill-шага"],
    [r"C", "токены запроса в кэше до шага"],
    [r"R = C / L", "доля токенов из кэша"],
    [r"\ell", "текущая длина контекста запроса в decode-шаге"],
    [r"B", "число запросов в decode-шаге"],
    [r"P = \sum L(1 - R)", "новые входные токены прогона"],
    [r"O", "выходные токены прогона"],
    [r"p_{\mathrm{decode}} = O / (P + O)", "доля decode в прогоне"],
    [r"T,\ T^{\mathrm{wall}}", "измеренное время шага; wall_seconds прогона"],
]

SCHEMA_VERSION = 1
DECODE_SAMPLE = 4000


def clean(value):
    """Recursively convert numpy/pandas values and NaN into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    return value


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    have = [c for c in columns if c in frame]
    return clean(frame[have].to_dict(orient="records"))


def binned(frame: pd.DataFrame, x: str, y: str, bins: int = 12) -> list[dict]:
    """Median and quartiles of y in quantile bins of x, per (source, part)."""
    out = []
    for (source, part), sub in frame.groupby(["source", "part"], sort=False):
        sub = sub[[x, y]].dropna()
        if part == "unused" or sub.empty:
            continue
        values = sub[x].to_numpy(float)
        if np.unique(values).size <= bins:
            groups = sub.groupby(x)
        else:
            edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
            labels = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
            groups = sub.groupby(labels)
        for _, g in groups:
            out.append(
                {
                    "source": source,
                    "part": part,
                    "x": float(g[x].median()),
                    "median": float(g[y].median()),
                    "q25": float(g[y].quantile(0.25)),
                    "q75": float(g[y].quantile(0.75)),
                    "n": int(len(g)),
                }
            )
    return out


def grid(frame: pd.DataFrame, x: str, y: str, z: str, nx: int = 6, ny: int = 5) -> dict:
    """Median z on a quantile grid of x and y (empty cells are null)."""
    sub = frame[[x, y, z]].dropna()
    if sub.empty:
        return {}

    def edges(values, n):
        e = np.unique(np.quantile(values, np.linspace(0, 1, n + 1)))
        return e if len(e) > 1 else np.array([values.min(), values.max() + 1e-9])

    ex, ey = edges(sub[x].to_numpy(float), nx), edges(sub[y].to_numpy(float), ny)
    ix = np.clip(np.digitize(sub[x], ex[1:-1]), 0, len(ex) - 2)
    iy = np.clip(np.digitize(sub[y], ey[1:-1]), 0, len(ey) - 2)
    median = np.full((len(ey) - 1, len(ex) - 1), np.nan)
    count = np.zeros_like(median, dtype=int)
    for j in range(len(ey) - 1):
        for i in range(len(ex) - 1):
            cell = sub[z].to_numpy()[(ix == i) & (iy == j)]
            count[j, i] = len(cell)
            if len(cell):
                median[j, i] = np.median(cell)
    centers = lambda e: ((e[:-1] + e[1:]) / 2).tolist()  # noqa: E731
    return {
        "x": centers(ex),
        "y": centers(ey),
        "x_edges": ex.tolist(),
        "y_edges": ey.tolist(),
        "median": median.tolist(),
        "count": count.tolist(),
    }


def paired(runs: pd.DataFrame) -> list[dict]:
    """Every pair of methods: |error| on the same test runs, Wilcoxon test."""
    from scipy import stats

    test = runs[runs["part"] == "test"]
    piv = test.pivot_table(index="run_id", columns="method", values="rel_err").abs()
    out = []
    for a, b in itertools.combinations(sorted(piv.columns), 2):
        pair = piv[[a, b]].dropna()
        if len(pair) < 2:
            continue
        diff = pair[b] - pair[a]
        p_value = stats.wilcoxon(pair[a], pair[b]).pvalue if diff.abs().sum() else 1.0
        out.append(
            {
                "a": a,
                "b": b,
                "n": int(len(pair)),
                "a_better": int((pair[a] < pair[b]).sum()),
                "mean_diff": float(diff.mean()),
                "wilcoxon_p": float(p_value),
            }
        )
    return out


def build(ev: Evaluation, meta: dict, notes: list[str]) -> dict:
    """Assemble the versioned report document from an evaluation."""
    sources = [
        {
            "id": source,
            "label": meta["labels"].get(source, source),
            "role": "train" if source == TRAIN_SOURCE else "transfer",
            "runs": len(data.runs),
            "prefill_steps": len(data.prefill_steps),
            "decode_steps": len(data.decode_steps),
        }
        for source, data in ev.datasets.items()
    ]
    methods = []
    for name, method in ev.methods.items():
        describe = method.describe()
        methods.append(
            {
                "name": name,
                "formula": describe["formula"],
                "coef": describe["coef"],
                "summary": describe.get("summary"),
                "latex": describe.get("latex", []),
                "has_step_predictions": hasattr(method, "predict_steps"),
                "details": {
                    k: v
                    for k, v in describe.items()
                    if k not in ("formula", "coef", "summary", "latex")
                },
            }
        )

    # Prefill steps with per-step L and R summaries.
    reqs = pd.concat(
        [d.prefill_requests.assign(source=s) for s, d in ev.datasets.items()],
        ignore_index=True,
    )
    step_lr = (
        reqs.groupby(["source", "run_id", "step_id"])
        .agg(L_mean=("L", "mean"), L_sum=("L", "sum"), C_sum=("C", "sum"))
        .reset_index()
    )
    step_lr["R_step"] = step_lr["C_sum"] / step_lr["L_sum"]
    pf = ev.prefill_steps.merge(step_lr, on=["source", "run_id", "step_id"], how="left")
    # R views use only steps with cached tokens: cache-off runs put every step at
    # R = 0 and would hide the dependence.
    cached = pf[pf["R_step"] > 0]
    prefill_binned = {
        "L_mean": binned(pf, "L_mean", "rel_err", bins=8),
        "R_step": binned(cached, "R_step", "rel_err", bins=6),
    }
    prefill_grid = grid(
        cached[cached["source"] == TRAIN_SOURCE], "L_mean", "R_step", "rel_err"
    )

    # Decode: a plotting sample per source, binned statistics on all steps.
    dc = ev.decode_steps
    sample = pd.concat(
        [
            g.sample(min(DECODE_SAMPLE, len(g)), random_state=0)
            for _, g in dc.groupby("source", sort=False)
        ]
    )
    step_columns = ["source", "part", "run_id", "T", "B", "sum_l", "sm_clock_mhz"]
    step_columns += [c for c in dc.columns if c.startswith(("pred__", "rel_err__"))]
    decode_binned = {}
    for m in methods:
        column = f"rel_err__{m['name']}"
        if column in dc:
            decode_binned[m["name"]] = {
                "sum_l": binned(dc, "sum_l", column),
                "B": binned(dc, "B", column),
                "sm_clock_mhz": binned(dc, "sm_clock_mhz", column, bins=10),
            }

    runs = ev.runs.copy()
    runs["abs_err"] = runs["rel_err"].abs()
    test = runs[runs["part"] == "test"]
    breakdown = (
        test.groupby(["method", "scheduler", "cache"])["abs_err"]
        .agg(["mean", "median", "max", "count"])
        .reset_index()
    )
    curves = []
    for name, method in ev.methods.items():
        if hasattr(method, "curves"):
            for curve in method.curves(ev.datasets[TRAIN_SOURCE], ev.split["test"]):
                curves.append({"method": name, **curve})

    coverage_values = {}
    for source, data in ev.datasets.items():
        dsample = sample[sample["source"] == source]
        coverage_values[source] = {
            "L": data.prefill_requests["L"].tolist(),
            "R": data.prefill_requests["R"].tolist(),
            "B": dsample["B"].tolist(),
            "Σℓ": dsample["sum_l"].tolist(),
            "p_decode": data.runs["p_decode"].tolist(),
        }
    coverage = []
    for source, data in ev.datasets.items():
        full = {
            "L": data.prefill_requests["L"],
            "R": data.prefill_requests["R"],
            "B": data.decode_steps["B"],
            "Σℓ": data.decode_steps["sum_l"],
            "p_decode": data.runs["p_decode"],
        }
        for name, values in full.items():
            v = np.asarray(values, float)
            coverage.append(
                {
                    "source": source,
                    "variable": name,
                    "min": v.min(),
                    "p05": np.percentile(v, 5),
                    "median": np.median(v),
                    "p95": np.percentile(v, 95),
                    "max": v.max(),
                    "n": len(v),
                }
            )

    run_columns = [
        "run_id",
        "method",
        "source",
        "part",
        "set_id",
        "seed",
        "cache",
        "scheduler",
        "wall_seconds",
        "pred",
        "rel_err",
        "prefill_seconds",
        "prefill_hat",
        "p_decode",
        "P",
        "O",
        "sm_clock_mhz_median",
        "L_mean",
        "R_run",
        "flag",
    ] + [c for c in runs.columns if c.startswith("diag__")]
    return clean(
        {
            "schema_version": SCHEMA_VERSION,
            "meta": meta["meta"],
            "sources": sources,
            "split": ev.split,
            "cache_split": ev.cache_split,
            "prefill_model": {
                "name": "A",
                "formula": "T_step = a*sum L(1-R) + b*sum L^2(1-R^2) + c",
                "latex": PREFILL_LATEX,
                "coef": ev.prefill.coef,
                "cache_split_coef": ev.prefill_cache_split.coef,
            },
            "methods": methods,
            "math": {
                "notation": NOTATION,
                "error": r"e = \frac{\hat T - T}{T}",
                "fit": FIT_LATEX.get(meta["meta"].get("fit"), ""),
                "fit_method": meta["meta"].get("fit"),
            },
            "metrics": ev.metrics,
            "paired": paired(ev.runs),
            "runs": _records(runs, run_columns),
            "test_breakdown": breakdown.to_dict(orient="records"),
            "curves": curves,
            "prefill_steps": _records(
                pf,
                [
                    "source",
                    "part",
                    "run_id",
                    "step_id",
                    "T",
                    "pred",
                    "rel_err",
                    "L_mean",
                    "R_step",
                    "sm_clock_mhz",
                    "new",
                    "quad",
                ],
            ),
            "decode_sample": _records(sample, step_columns),
            "decode_binned": decode_binned,
            "prefill_binned": prefill_binned,
            "prefill_grid": prefill_grid,
            "prefill_forms": analyze_prefill_forms(ev.datasets[TRAIN_SOURCE]),
            "coverage": coverage,
            "coverage_values": coverage_values,
            "notes": notes,
        }
    )
