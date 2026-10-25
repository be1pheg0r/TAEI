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
                "has_step_predictions": hasattr(method, "predict_steps"),
                "details": {
                    k: v for k, v in describe.items() if k not in ("formula", "coef")
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
                "coef": ev.prefill.coef,
                "cache_split_coef": ev.prefill_cache_split.coef,
            },
            "methods": methods,
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
            "coverage": coverage,
            "coverage_values": coverage_values,
            "notes": notes,
        }
    )
