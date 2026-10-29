"""Замер закрытого цикла (S5, S8) для проверки модели «от N».

Лёгкое чтение попытки: ``runs.json`` (точка, окно замера, пул запросов),
``requests.jsonl`` и ``steps.jsonl``; полный загрузчик ``data.load`` для S8 не
нужен (попытка W2 — до 220 МБ). Окно — от конца шага, где завершился последний
запрос прогрева, до конца шага последнего измеряемого. Метрики окна — по
запросам, завершившимся в нём:
- RPS = завершений в окне / длина окна (при пачках равных запросов это не
  плановые 200);
- ITPS новые = Σ(L − C) / окно, полный = ΣL / окно, OTPS = ΣO / окно;
- TTFT = первый токен − подача, p50 и p90; TPOT = (конец − первый токен)/(O − 1),
  медиана; очередь = первое планирование − подача, среднее.
Колено по факту — средний B (с весом по времени шага) decode-шагов окна, перед
которыми в очереди кто-то ждёт: prefill идёт первым, поэтому такой шаг значит,
что приём упёрся в KV. Это разметка проверки (ниже или выше колена), во вход
модели она не идёт.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("rps", "itps_new", "itps_full", "otps", "ttft_p50", "ttft_p90", "tpot")


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def attempt_row(run: dict, attempt: Path) -> dict:
    loop = run["closed_loop"]
    lo, hi = loop["window_start_ns"], loop["window_end_ns"]
    reqs = [
        r
        for r in _jsonl(attempt / "requests.jsonl")
        if r["finish_monotonic_ns"] is not None and lo < r["finish_monotonic_ns"] <= hi
    ]
    T = (hi - lo) * 1e-9
    L = np.array([r["input_token_count"] for r in reqs], float)
    C = np.array([r["prefix_reused_tokens"] or 0 for r in reqs], float)
    out = np.array([r["output_token_count"] for r in reqs], float)
    enq = np.array([r["enqueue_monotonic_ns"] for r in reqs], float)
    first = np.array([r["first_host_observed_monotonic_ns"] for r in reqs], float)
    fin = np.array([r["finish_monotonic_ns"] for r in reqs], float)
    sched = np.array([r["first_scheduled_monotonic_ns"] for r in reqs], float)
    ttft = (first - enq) * 1e-9
    many = out > 1
    tpot = (fin[many] - first[many]) * 1e-9 / (out[many] - 1)
    tb = tt = 0.0
    for s in _jsonl(attempt / "steps.jsonl"):
        if (
            s["phase"] == "decode"
            and s["queue_before"] > 0
            and lo <= s["step_host_end_ns"] <= hi
        ):
            dt = s["step_host_end_ns"] - s["step_host_start_ns"]
            tb += dt * s["actual_batch_size"]
            tt += dt
    return {
        "run_id": run["run_id"],
        "phase": run["run_id"].split("-")[0],
        "load": (run.get("block") or run["series"]).replace("S8-", ""),
        "scheduler": run["scheduler_mode"],
        "cache": run["prefix_cache_mode"],
        "N": int(loop["N"]),
        "seed": int(run["seed"]),
        "completed": len(reqs),
        "window": T,
        "rps": len(reqs) / T,
        "itps_new": float((L - C).sum()) / T,
        "itps_full": float(L.sum()) / T,
        "otps": float(out.sum()) / T,
        "ttft_p50": float(np.quantile(ttft, 0.5)),
        "ttft_p90": float(np.quantile(ttft, 0.9)),
        "tpot": float(np.median(tpot)),
        "queue": float(((sched - enq) * 1e-9).mean()),
        "kv_tb": tb,
        "kv_t": tt,
        "pool": run["workload_spec"]["requests"],
    }


def read(roots, patterns=("*-S5-*", "*-S8-*")) -> pd.DataFrame:
    """One row per successful closed-loop attempt under ``roots``."""
    rows = []
    for root in roots:
        for pattern in patterns:
            for f in sorted(Path(root).glob(f"{pattern}/*/runs.json")):
                run = json.loads(f.read_text())
                run = run[0] if isinstance(run, list) else run
                if run.get("status") == "success" and run.get("closed_loop"):
                    rows.append(attempt_row(run, f.parent))
    return pd.DataFrame(rows)


def knees(frame: pd.DataFrame) -> dict[tuple, float]:
    """Measured KV knee per (load, scheduler, cache): B of KV-limited steps."""
    out = {}
    for key, g in frame.groupby(["load", "scheduler", "cache"]):
        if g["kv_t"].sum() > 0:
            out[key] = float(g["kv_tb"].sum() / g["kv_t"].sum())
    return out


def points(frame: pd.DataFrame) -> pd.DataFrame:
    """Measured point (phase, load, scheduler, cache, N): mean over the seeds."""
    keys = ["phase", "load", "scheduler", "cache", "N"]
    cols = [*METRICS, "queue", "completed", "window"]
    g = frame.groupby(keys)
    out = g[cols].mean()
    out["seeds"] = g.size()
    out["completed_min"] = g["completed"].min()
    return out.reset_index()
