"""Проверки модели по всем имеющимся данным, без новой подгонки основной модели.

Для каждого набора данных (источник × часть: train, test, перенос) считаются
сводка ошибок по видам проверок, отдельные проверки prefill (одиночный запрос
без кэша, модель без кэша на шагах с кэшем, k запросов в шаге, шаги с малым
числом новых токенов) и decode (сетка ℓ̄ × B), сгруппированные TTPS–TTFT и
качество данных (время вне шагов, выбросы шагов). Всё — по полным данным, для
графиков — выборки. Единственная новая подгонка — проверочная модель A только
на шагах без кэша (NNLS), её коэффициенты в основную модель не идут.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import TRAIN_SOURCE, Evaluation
from .fit import relative_errors, summary
from .prefill import PrefillModel

SMALL_NEW = 64  # "few new tokens" prefill steps
WINDOW_MID = (0.3, 0.7)
SPIKE_SECONDS = 0.1  # a step at least this long ...
SPIKE_RATIO = 5.0  # ... and this many times its run's median at the same B


def step_r(ev: Evaluation) -> pd.DataFrame:
    """Prefill steps with L, k (requests), R_кэш and R_нарезка of the step."""
    reqs = pd.concat(
        [d.prefill_requests.assign(source=s) for s, d in ev.datasets.items()],
        ignore_index=True,
    )
    by_step = (
        reqs.groupby(["source", "run_id", "step_id"])
        .agg(
            k=("L", "size"),
            L_sum=("L", "sum"),
            L_min=("L", "min"),
            L_max=("L", "max"),
            C_sum=("C", "sum"),
            C_cache=("C_cache", "sum"),
            C_chunk=("C_chunk", "sum"),
        )
        .reset_index()
    )
    pf = ev.prefill_steps.merge(by_step, on=["source", "run_id", "step_id"])
    pf["R_cache"] = pf["C_cache"] / pf["L_sum"]
    pf["R_chunk"] = pf["C_chunk"] / pf["L_sum"]
    return pf


def _parts(ev: Evaluation, include_train: bool) -> list[tuple[str, str]]:
    """Data sets of a model: (source, part); train only for designed grids."""
    out = []
    for source in ev.datasets:
        parts = ev.runs.loc[ev.runs["source"] == source, "part"].unique()
        for part in ("train", "test", "transfer"):
            if part in parts and (part != "train" or include_train):
                out.append((source, part))
    return out


def summary_cells(
    ev: Evaluation,
    windows: pd.DataFrame,
    requests: pd.DataFrame,
    labels: dict[str, str],
    include_train: bool,
) -> list[dict]:
    """One cell per (check, data set): mean |e|, median |e|, mean e, n."""
    pf = step_r(ev)
    checks = {
        "prefill без кэша": lambda s, p: pf.loc[
            (pf["source"] == s) & (pf["part"] == p) & (pf["C_cache"] == 0), "rel_err"
        ],
        "prefill с кэшем": lambda s, p: pf.loc[
            (pf["source"] == s) & (pf["part"] == p) & (pf["C_cache"] > 0), "rel_err"
        ],
        "decode": lambda s, p: ev.decode_steps.loc[
            (ev.decode_steps["source"] == s) & (ev.decode_steps["part"] == p),
            "rel_err",
        ],
        "T_total": lambda s, p: ev.runs.loc[
            (ev.runs["source"] == s) & (ev.runs["part"] == p), "rel_err"
        ],
        "смеси (окна)": lambda s, p: _window_gap(windows, s, p),
        "TTFT": lambda s, p: _request_err(requests, s, p, "ttft"),
        "TPOT": lambda s, p: _request_err(requests, s, p, "tpot"),
    }
    cells = []
    for source, part in _parts(ev, include_train):
        column = f"{labels.get(source, source)} · {part}"
        for check, values in checks.items():
            rel = pd.Series(values(source, part)).dropna()
            cells.append(
                {
                    "check": check,
                    "column": column,
                    **(summary(rel.to_numpy()) if len(rel) else {"n": 0}),
                }
            )
    return cells


def _window_gap(windows: pd.DataFrame, source: str, part: str) -> pd.Series:
    """x + y − 1 of windows with 30–70% prefill: the mixed windows."""
    lo, hi = WINDOW_MID
    w = windows[
        (windows["source"] == source)
        & (windows["part"] == part)
        & (windows["share"] >= lo)
        & (windows["share"] < hi)
    ]
    return w["x"] + w["y"] - 1


def _request_err(requests: pd.DataFrame, source: str, part: str, name: str):
    r = requests[(requests["source"] == source) & (requests["part"] == part)]
    r = r[[name, name + "_hat"]].dropna()
    r = r[r[name] > 0]
    return (r[name + "_hat"] - r[name]) / r[name]


def prefill_checks(ev: Evaluation, cap: int = 1500) -> dict:
    pf = step_r(ev)
    coef = ev.prefill.coef

    # 1. One request, nothing cached: T(L) measured and the model curve.
    single = pf[(pf["k"] == 1) & (pf["C_sum"] == 0)]
    lo, hi = (single["L_sum"].min(), single["L_sum"].max()) if len(single) else (1, 1)
    grid = np.geomspace(max(lo, 1), max(hi, 2), 60)
    curve = coef["a"] * grid + coef["b"] * grid**2 + coef["c"]

    # 2. Cache: model A fitted only on steps without prefix-cache hits
    # (R_кэш = 0; own chunked KV allowed), predicting steps with hits.
    train_src = pf[pf["source"] == TRAIN_SOURCE]
    cache = None
    # Three coefficients need at least three steps without hits.
    if (pf["C_cache"] > 0).any() and (train_src["C_cache"] == 0).sum() >= 3:
        model = PrefillModel().fit(train_src[train_src["C_cache"] == 0], "regression")
        hits = pf[pf["C_cache"] > 0].copy()
        hits["pred_nocache"] = model.predict_steps(hits)
        hits["e"] = relative_errors(hits["pred_nocache"], hits["T"])
        hits["L_bin"] = _pow2_bin(hits["L_sum"] / hits["k"])
        hits["R_bin"] = pd.cut(
            hits["R_cache"], [0, 0.25, 0.5, 0.75, 0.9, 1.0001], right=False
        ).astype(str)
        table = (
            hits.groupby(["source", "L_bin", "R_bin"])
            .agg(
                n=("e", "size"), e=("e", "mean"), abs_e=("e", lambda v: v.abs().mean())
            )
            .reset_index()
        )
        cache = {
            "coef": model.coef,
            "table": table.to_dict(orient="records"),
            "by_source": [
                {"source": s, **summary(g["e"].to_numpy())}
                for s, g in hits.groupby("source", sort=False)
            ],
        }

    # 3. k requests in one step, all of one length L, nothing cached.
    same = pf[(pf["L_min"] == pf["L_max"]) & (pf["C_sum"] == 0)].copy()
    same["L"] = same["L_min"]
    multi = (
        same.groupby(["source", "L", "k"])
        .agg(n=("T", "size"), T=("T", "median"), pred=("pred", "median"))
        .reset_index()
    )
    multi = multi[multi.groupby(["source", "L"])["k"].transform("nunique") > 1]
    multi["T_per"] = multi["T"] / multi["k"]
    multi["pred_per"] = multi["pred"] / multi["k"]

    # 4. Steps that compute fewer than SMALL_NEW new tokens.
    small = pf[pf["new"] < SMALL_NEW]
    return {
        "single": _sample(single, cap)[
            ["source", "part", "run_id", "L_sum", "T", "pred"]
        ].to_dict(orient="records"),
        "single_n": int(len(single)),
        "curve": {"L": grid.tolist(), "T": curve.tolist()},
        "cache": cache,
        "multi": multi.to_dict(orient="records"),
        "small": _sample(small, cap)[
            ["source", "part", "run_id", "new", "k", "L_sum", "R_cache", "T", "pred"]
        ].to_dict(orient="records"),
        "small_n": int(len(small)),
        "small_threshold": SMALL_NEW,
    }


def _pow2_bin(values: pd.Series) -> pd.Series:
    """Label of the power-of-two band [2^k, 2^(k+1)) a value falls in."""
    k = np.floor(np.log2(np.maximum(values.to_numpy(float), 1))).astype(int)
    return pd.Series([f"{2**i}–{2 ** (i + 1)}" for i in k], index=values.index)


B_BANDS = (
    (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128),
    (129, 10**6),
)  # fmt: skip


def decode_checks(ev: Evaluation) -> list[dict]:
    """Decode steps on a grid ℓ̄ (mean context, power-of-two bands) × B band."""
    dc = ev.decode_steps.copy()
    dc["ell"] = dc["sum_l"] / dc["B"]
    dc["ell_bin"] = _pow2_bin(dc["ell"])
    dc["ell_lo"] = 2 ** np.floor(np.log2(np.maximum(dc["ell"], 1)))
    band = np.zeros(len(dc), dtype=object)
    for lo, hi in B_BANDS:
        band[(dc["B"] >= lo).to_numpy() & (dc["B"] <= hi).to_numpy()] = (
            f"{lo}" if lo == hi else (f"{lo}–{hi}" if hi < 10**6 else f"≥{lo}")
        )
    dc["B_band"] = band
    dc["B_lo"] = dc["B_band"].map(lambda b: int(str(b).lstrip("≥").split("–")[0]))
    table = (
        dc.groupby(["source", "part", "ell_lo", "ell_bin", "B_lo", "B_band"])
        .agg(
            n=("T", "size"),
            T=("T", "median"),
            pred=("pred", "median"),
            e=("rel_err", "mean"),
        )
        .reset_index()
        .sort_values(["source", "part", "ell_lo", "B_lo"])
    )
    return table.drop(columns=["ell_lo", "B_lo"]).to_dict(orient="records")


def request_groups(requests: pd.DataFrame, rho: dict[str, float]) -> list[dict]:
    """Medians of TTFT and TTPS per (data set, ρ, scheduler, cache): one point
    per load; ρ only for runs with timed arrivals."""
    r = requests[requests["part"] != "unused"].copy()
    r["rho"] = r["run_id"].map(rho)
    keys = ["source", "part", "rho", "scheduler", "cache"]
    g = (
        r.groupby(keys, dropna=False)
        .agg(
            n=("ttft", "size"),
            runs=("run_id", "nunique"),
            ttft=("ttft", "median"),
            ttft_hat=("ttft_hat", "median"),
            ttft_p90=("ttft", lambda v: v.quantile(0.9)),
            ttft_hat_p90=("ttft_hat", lambda v: v.quantile(0.9)),
            ttps=("ttps", "median"),
            ttps_hat=("ttps_hat", "median"),
        )
        .reset_index()
    )
    return g.to_dict(orient="records")


def spikes(ev: Evaluation, labels: dict[str, str]) -> dict:
    """Decode steps far above their run's median at the same B."""
    dc = ev.decode_steps
    base = dc.groupby(["source", "run_id", "B"])["T"].transform("median")
    spike = (dc["T"] > SPIKE_SECONDS) & (dc["T"] > SPIKE_RATIO * base)
    first = dc["step_index"] == dc.groupby(["source", "run_id"])[
        "step_index"
    ].transform("min")
    frame = dc.assign(spike=spike, base=base, first=first)
    rows = []
    for source, g in frame.groupby("source", sort=False):
        s = g[g["spike"]]
        rows.append(
            {
                "source": labels.get(source, source),
                "steps": len(g),
                "spikes": len(s),
                "runs_with_spike": int(s["run_id"].nunique()),
                "runs": int(g["run_id"].nunique()),
                "rate": len(s) / len(g),
                "T_median": s["T"].median() if len(s) else None,
                "T_max": s["T"].max() if len(s) else None,
                "time_share": (s["T"] - s["base"]).sum() / g["T"].sum(),
                "first_step_share": s["first"].mean() if len(s) else None,
                "B1_share": (s["B"] == 1).mean() if len(s) else None,
            }
        )
    top = (
        frame[frame["spike"]]
        .sort_values("T", ascending=False)
        .head(12)[["source", "run_id", "step_index", "B", "sum_l", "T", "base"]]
    )
    top["source"] = top["source"].map(lambda s: labels.get(s, s))
    return {
        "by_source": rows,
        "top": top.to_dict(orient="records"),
        "seconds": SPIKE_SECONDS,
        "ratio": SPIKE_RATIO,
    }


def off_steps(ev: Evaluation) -> list[dict]:
    """Share of wall time outside measured steps, per run, against its length."""
    r = ev.runs[ev.runs["part"] != "unused"]
    out = r[["source", "part", "run_id", "wall_seconds", "steps_seconds"]].copy()
    out["off_share"] = 1 - out["steps_seconds"] / out["wall_seconds"]
    return out.to_dict(orient="records")


def _sample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if not len(frame):
        return frame
    return pd.concat(
        [
            g.sample(n, random_state=0) if len(g) > n else g
            for _, g in frame.groupby("source", sort=False)
        ]
    )
