"""Цены токенов, бюджет GPU и метрики запросов по модели и по замеру.

Цена входного токена ã = T_pf / L, выходного b̃ = T_dc / B. Для нагрузки они
взвешены по токенам: ã = Σ T_pf / I, b̃ = Σ T_dc / O, где I — все токены
промптов (включая кэш; сумма L по prefill-строкам считала бы куски chunked
prefill и пересчёт дважды), O — выданные токены. Тогда ã·I + b̃·O = Σ T_pf +
Σ T_dc, и бюджет ã·ITPS + b̃·OTPS при ITPS = I / T, OTPS = O / T равен доле
времени, которую GPU считает шаги: по замеру это Σ шагов / wall, по модели —
T̂_total / wall = 1 + e.

Прямые ã·ITPS + b̃·OTPS = 1 строятся по коэффициентам модели для одного
запроса медианной длины L и decode-шага с медианным B и медианной ℓ̄:
ã = a(1−R) + b·L(1−R²) + c/L, b̃ = d0/B + d1·ℓ̄ + d2.

TTFT и TPOT по запросам: замер — по меткам времени запроса; модель — сумма
предсказанных шагов по реальному расписанию трассы (ожидание в очереди взято из
трассы, а не предсказано). TPOT по модели — предсказанные шаги после первого
токена до последнего, делённые на O − 1: без чужих prefill-шагов это ровно
T̂_dc при B из трассы.
"""

from __future__ import annotations

import pandas as pd

from .evaluate import TRAIN_SOURCE, Evaluation
from .fit import summary

R_FAMILY = (0.0, 0.3, 0.6, 0.9)
B_FAMILY = (1, 2, 4, 8)


def token_prices(coef: dict, L: float, R: float, B: float, ell: float) -> tuple:
    """(ã, b̃) for one request of length L with reuse R and a decode step of B."""
    a_tilde = coef["a"] * (1 - R) + coef["b"] * L * (1 - R**2) + coef["c"] / L
    b_tilde = coef["d0"] / B + coef["d1"] * ell + coef["d2"]
    return a_tilde, b_tilde


def run_prices(ev: Evaluation) -> pd.DataFrame:
    """Weighted ã, b̃ and the budget per run, by the model and by measurement."""
    keys = ["source", "run_id"]
    pf = ev.prefill_steps.groupby(keys)[["pred", "T"]].sum()
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
    out = runs[["part", "cache", "scheduler", "I", "O", "lost_tokens", "wall_seconds"]]
    out = out.assign(
        a_model=sums["pf_hat"] / runs["I"],
        b_model=sums["dc_hat"] / runs["O"],
        a_meas=sums["pf"] / runs["I"],
        b_meas=sums["dc"] / runs["O"],
        ITPS=runs["I"] / runs["wall_seconds"],
        OTPS=runs["O"] / runs["wall_seconds"],
    )
    out["budget_model"] = out["a_model"] * out["ITPS"] + out["b_model"] * out["OTPS"]
    out["budget_meas"] = out["a_meas"] * out["ITPS"] + out["b_meas"] * out["OTPS"]
    return out.reset_index()


def lines(ev: Evaluation) -> dict:
    """Model lines ã·ITPS + b̃·OTPS = 1: a family over R and one over B."""
    data = ev.datasets[TRAIN_SOURCE]
    coef = {**ev.prefill.coef, **ev.decode.coef}
    L = float(data.requests["prompt"].median())
    dc = data.decode_steps
    B = float(dc["B"].median()) if len(dc) else 1.0
    ell = float((dc["sum_l"] / dc["B"]).median()) if len(dc) else L

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
        }

    return {
        "ref": {"L": L, "B": B, "ell": ell},
        "by_R": [line(R, B, f"R = {R:g}") for R in R_FAMILY],
        "by_B": [line(0.0, b, f"B = {b}") for b in B_FAMILY],
    }


def requests(ev: Evaluation) -> pd.DataFrame:
    """TTFT and TPOT per request: measured, and summed predicted steps of the trace."""
    rows = []
    for source, data in ev.datasets.items():
        pf = ev.prefill_steps[ev.prefill_steps["source"] == source]
        dc = ev.decode_steps[ev.decode_steps["source"] == source]
        steps = pd.concat(
            [pf[["run_id", "step_index", "pred"]], dc[["run_id", "step_index", "pred"]]]
        ).sort_values(["run_id", "step_index"])
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
        frame = reqs.assign(source=source)
        frame["ttft"] = (frame["first_ns"] - frame["enqueue_ns"]) * 1e-9
        frame["ttft_hat"] = first
        # TPOT only for complete requests with at least two output tokens.
        full = (frame["O"] > 1) & (frame["O"] == frame["O_requested"])
        gaps = (frame["O"] - 1).where(full)
        frame["tpot"] = (frame["finish_ns"] - frame["first_ns"]) * 1e-9 / gaps
        frame["tpot_hat"] = (last - first) / gaps
        frame["ttps"] = 1 / frame["tpot"]
        frame["ttps_hat"] = 1 / frame["tpot_hat"]
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
