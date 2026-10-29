"""Цены токенов, бюджет GPU и метрики запросов по модели и по замеру.

Входные токены I — только новые, не кэшированные: те, что prefill-шаги
действительно считают (N = L(1 − R) по prefill-строкам, с пересчётом после
вытеснения; кэш префиксов и готовый KV своего куска не входят). Цена входного
токена ã = T_pf / (L(1 − R)), выходного b̃ = T_dc / B. Для нагрузки они взвешены
по токенам: ã = Σ T_pf / I, b̃ = Σ T_dc / O, O — выданные токены. Тогда
ã·I + b̃·O = Σ T_pf + Σ T_dc, и бюджет ã·ITPS + b̃·OTPS при ITPS = I / T,
OTPS = O / T равен доле времени, которую GPU считает шаги: по замеру это
Σ шагов / wall, по модели — T̂_total / wall = 1 + e.

Прямые ã·ITPS + b̃·OTPS = 1 строятся по коэффициентам модели для одного
запроса медианной длины L и decode-шага с медианным B и медианной ℓ̄:
ã = a + b·L(1 + R) + c / (L(1 − R)), b̃ = d0/B + d1·ℓ̄ + d2.

TTFT и TPOT по запросам: замер — по меткам времени запроса; модель — сумма
предсказанных шагов по реальному расписанию трассы (ожидание в очереди взято из
трассы, а не предсказано): от первого шага, начатого после прихода запроса, до
шага первого токена. При закрытом батче это все шаги с начала прогона.
TPOT по модели — предсказанные шаги после первого токена до последнего,
делённые на O − 1: без чужих prefill-шагов это ровно T̂_dc при B из трассы.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import TRAIN_SOURCE, Evaluation
from .fit import summary

R_FAMILY = (0.0, 0.3, 0.6, 0.9)
B_FAMILY = (1, 2, 4, 8)


def token_prices(coef: dict, L: float, R: float, B: float, ell: float) -> tuple:
    """(ã, b̃) per new input token of one request of length L with reuse R, and
    per output token of a decode step of B."""
    a_tilde = coef["a"] + coef["b"] * L * (1 + R) + coef["c"] / (L * (1 - R))
    b_tilde = coef["d0"] / B + coef["d1"] * ell + coef["d2"]
    return a_tilde, b_tilde


def run_prices(ev: Evaluation) -> pd.DataFrame:
    """Weighted ã, b̃ and the budget per run, by the model and by measurement."""
    keys = ["source", "run_id"]
    pf = ev.prefill_steps.groupby(keys)[["pred", "T", "new"]].sum()
    dc = ev.decode_steps.groupby(keys)[["pred", "T"]].sum()
    runs = ev.runs.set_index(keys)
    sums = pd.DataFrame(
        {
            "pf_hat": pf["pred"],
            "pf": pf["T"],
            "dc_hat": dc["pred"],
            "dc": dc["T"],
        }
    ).reindex(runs.index)
    sums = sums.fillna(0.0)
    new = pf["new"].reindex(runs.index).fillna(0.0)
    out = runs[["part", "cache", "scheduler", "I", "O", "lost_tokens", "wall_seconds"]]
    out = out.assign(
        I_new=new,
        a_model=sums["pf_hat"] / new,
        b_model=sums["dc_hat"] / runs["O"],
        a_meas=sums["pf"] / new,
        b_meas=sums["dc"] / runs["O"],
        ITPS=new / runs["wall_seconds"],
        OTPS=runs["O"] / runs["wall_seconds"],
    )
    out["budget_model"] = out["a_model"] * out["ITPS"] + out["b_model"] * out["OTPS"]
    out["budget_meas"] = out["a_meas"] * out["ITPS"] + out["b_meas"] * out["OTPS"]
    return out.reset_index()


def kv_batch_limit(data) -> float | None:
    """B_max = KV capacity in tokens / (median L + median O): how many typical
    requests fit in the KV cache at once."""
    runs = data.runs
    if runs["kv_blocks"].isna().all():
        return None
    blocks = float(runs["kv_blocks"].median())
    size = float(runs["kv_block_size"].median()) if "kv_block_size" in runs else 256.0
    per_request = data.requests["prompt"].median() + data.requests["O"].median()
    return blocks * size / float(per_request)


def lines(ev: Evaluation) -> dict:
    """Model lines ã·ITPS + b̃·OTPS = 1: a family over R and one over B.

    In the B family a line with B above the KV limit ``B_max`` is marked
    unreachable: that many typical requests do not fit in memory at once.
    """
    data = ev.datasets[TRAIN_SOURCE]
    coef = {**ev.prefill.coef, **ev.decode.coef}
    L = float(data.requests["prompt"].median())
    dc = data.decode_steps
    B = float(dc["B"].median()) if len(dc) else 1.0
    ell = float((dc["sum_l"] / dc["B"]).median()) if len(dc) else L
    b_max = kv_batch_limit(data)

    def line(R, b, label):
        a_t, b_t = token_prices(coef, L, R, b, ell)
        return {
            "label": label,
            "R": R,
            "B": b,
            "a_tilde": a_t,
            "b_tilde": b_t,
            "itps_max": 1 / a_t,
            "otps_max": 1 / b_t,
            "reachable": b_max is None or b <= b_max,
        }

    return {
        "ref": {
            "L": L,
            "B": B,
            "ell": ell,
            "B_max": b_max,
            "O": float(data.requests["O"].median()),
        },
        "by_R": [line(R, B, f"R_кэш = {R:g}") for R in R_FAMILY],
        "by_B": [line(0.0, b, f"B = {b}") for b in B_FAMILY],
    }


def windows(ev: Evaluation, seconds: float, min_steps: int) -> pd.DataFrame:
    """ITPS–OTPS of time windows inside each run, with model token prices.

    A run is cut into windows of ``seconds`` from its first step; the last
    window ends with the last step. A step that crosses a window border is
    split between the windows by overlap: its time, prediction and tokens go
    to each window in proportion (8B prefill steps last seconds). Windows
    overlapped by fewer than ``min_steps`` steps are dropped. Per window: I_w —
    new (not cached) prompt tokens computed by its prefill steps, O_w —
    tokens of its decode steps (sum of B), and ã_w = ΣT̂_pf / I_w,
    b̃_w = ΣT̂_dc / O_w from the model. The point
    (ã_w·ITPS_w, b̃_w·OTPS_w) = (ΣT̂_pf, ΣT̂_dc) / T_w lies on x + y = 1 when
    the model is right and the GPU never idles; the colour is the measured
    prefill share of the window's step time.
    """
    pf = ev.prefill_steps.assign(kind="pf", I=ev.prefill_steps["new"], O=0.0)
    dc = ev.decode_steps.assign(kind="dc", I=0.0, O=ev.decode_steps["B"])
    cols = ["source", "part", "run_id", "kind", "t_start_ns", "t_end_ns"]
    cols += ["T", "pred", "I", "O"]
    steps = pd.concat([pf[cols], dc[cols]], ignore_index=True)
    keys = ["source", "run_id"]
    t0 = steps.groupby(keys)["t_start_ns"].transform("min").to_numpy(float)
    end = steps.groupby(keys)["t_end_ns"].transform("max").to_numpy(float)
    width = seconds * 1e9
    start = (steps["t_start_ns"].to_numpy(float) - t0) / width
    stop = (steps["t_end_ns"].to_numpy(float) - t0) / width
    first = np.floor(start).astype(int)
    last = np.maximum(first, np.ceil(stop).astype(int) - 1)
    # One row per (step, window it overlaps), weighted by the overlap.
    reps = last - first + 1
    idx = np.repeat(np.arange(len(steps)), reps)
    w = first[idx] + (np.arange(len(idx)) - np.repeat(np.cumsum(reps) - reps, reps))
    span = np.maximum(stop - start, 1e-12)[idx]
    part = np.minimum(stop[idx], w + 1) - np.maximum(start[idx], w)
    frac = np.clip(part / span, 0, 1)
    rows = steps.iloc[idx].reset_index(drop=True)
    rows["w"] = w
    rows["frac"] = frac
    run_len = (end - t0)[idx] / 1e9
    rows["T_w"] = np.minimum(seconds, run_len - w * seconds)
    is_pf = rows["kind"] == "pf"
    for col, src, mask in (
        ("T_pf", "T", is_pf),
        ("T_dc", "T", ~is_pf),
        ("pf_hat", "pred", is_pf),
        ("dc_hat", "pred", ~is_pf),
    ):
        rows[col] = (rows[src] * rows["frac"]).where(mask, 0.0)
    rows["I"] = rows["I"] * rows["frac"]
    rows["O"] = rows["O"] * rows["frac"]
    win = (
        rows.groupby(["source", "part", "run_id", "w"])
        .agg(
            T_w=("T_w", "first"),
            steps=("T", "size"),
            I_w=("I", "sum"),
            O_w=("O", "sum"),
            T_pf=("T_pf", "sum"),
            T_dc=("T_dc", "sum"),
            pf_hat=("pf_hat", "sum"),
            dc_hat=("dc_hat", "sum"),
        )
        .reset_index()
    )
    win = win[(win["steps"] >= min_steps) & (win["T_w"] > 0)].copy()
    win["ITPS"] = win["I_w"] / win["T_w"]
    win["OTPS"] = win["O_w"] / win["T_w"]
    win["a_w"] = (win["pf_hat"] / win["I_w"]).where(win["I_w"] > 0)
    win["b_w"] = (win["dc_hat"] / win["O_w"]).where(win["O_w"] > 0)
    win["x"] = win["pf_hat"] / win["T_w"]  # ã_w · ITPS_w
    win["y"] = win["dc_hat"] / win["T_w"]  # b̃_w · OTPS_w
    win["share"] = win["T_pf"] / (win["T_pf"] + win["T_dc"])
    return win


SHARE_BINS = ((0.0, 0.3), (0.3, 0.7), (0.7, 1.0))


def window_summary(win: pd.DataFrame) -> list[dict]:
    """Windows per prefill-share band and how far points are from x + y = 1."""
    out = []
    for (source, part), g in win.groupby(["source", "part"], sort=False):
        if part == "unused":
            continue
        row = {"source": source, "part": part, "windows": len(g)}
        for lo, hi in SHARE_BINS:
            inside = (g["share"] >= lo) & (
                (g["share"] < hi) if hi < 1 else (g["share"] <= hi)
            )
            row[f"share_{round(lo * 100)}_{round(hi * 100)}"] = int(inside.sum())
        gap = (g["x"] + g["y"] - 1).abs()
        row["mean_abs_gap"] = float(gap.mean())
        mid = g[(g["share"] >= 0.3) & (g["share"] < 0.7)]
        row["mean_abs_gap_mid"] = (
            float((mid["x"] + mid["y"] - 1).abs().mean()) if len(mid) else None
        )
        out.append(row)
    return out


def requests(ev: Evaluation) -> pd.DataFrame:
    """TTFT and TPOT per request: measured, and summed predicted steps of the trace."""
    rows = []
    for source, data in ev.datasets.items():
        pf = ev.prefill_steps[ev.prefill_steps["source"] == source]
        dc = ev.decode_steps[ev.decode_steps["source"] == source]
        cols = ["run_id", "step_index", "t_start_ns", "pred"]
        steps = pd.concat([pf[cols], dc[cols]]).sort_values(["run_id", "step_index"])
        steps["cum"] = steps.groupby("run_id")["pred"].cumsum()
        cum = steps.set_index(["run_id", "step_index"])["cum"]
        reqs = data.requests.merge(
            data.runs[["run_id", "scheduler", "cache"]], on="run_id"
        ).merge(
            ev.runs.loc[ev.runs["source"] == source, ["run_id", "part"]], on="run_id"
        )
        first = cum.reindex(
            list(zip(reqs["run_id"], reqs["first_step"], strict=True))
        ).to_numpy()
        last = cum.reindex(
            list(zip(reqs["run_id"], reqs["last_step"], strict=True))
        ).to_numpy()
        # Predicted time of the steps that started before the request arrived.
        before = np.zeros(len(reqs))
        for run_id, idx in reqs.groupby("run_id").indices.items():
            run = steps[steps["run_id"] == run_id]
            k = np.searchsorted(
                run["t_start_ns"].to_numpy(), reqs["enqueue_ns"].to_numpy()[idx]
            )
            done = np.concatenate([[0.0], run["cum"].to_numpy()])
            before[idx] = done[k]
        frame = reqs.assign(source=source)
        frame["ttft"] = (frame["first_ns"] - frame["enqueue_ns"]) * 1e-9
        frame["ttft_hat"] = first - before
        # TPOT only for complete requests with at least two output tokens.
        full = (frame["O"] > 1) & (frame["O"] == frame["O_requested"])
        gaps = (frame["O"] - 1).where(full)
        frame["tpot"] = (frame["finish_ns"] - frame["first_ns"]) * 1e-9 / gaps
        frame["tpot_hat"] = (last - first) / gaps
        frame["ttps"] = 1 / frame["tpot"]
        frame["ttps_hat"] = 1 / frame["tpot_hat"]
        # Number of the request in its run by arrival (0 = first to arrive).
        frame["arrival_rank"] = (
            frame.groupby("run_id")["enqueue_ns"].rank(method="first").astype(int) - 1
        )
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    return out.drop(columns=["enqueue_ns", "first_ns", "finish_ns"])


def request_errors(reqs: pd.DataFrame) -> list[dict]:
    """Relative error of TTFT and TPOT by the model, per source and part."""
    out = []
    for (source, part), sub in reqs.groupby(["source", "part"], sort=False):
        if part == "unused":
            continue
        for name in ("ttft", "tpot"):
            pair = sub[[name, name + "_hat"]].dropna()
            pair = pair[pair[name] > 0]
            if len(pair):
                rel = (pair[name + "_hat"] - pair[name]) / pair[name]
                out.append(
                    {"what": name, "source": source, "part": part, **summary(rel)}
                )
    return out
