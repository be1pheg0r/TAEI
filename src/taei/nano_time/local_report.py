"""Разделы отчёта о локальном прогоне Qwen3-0.6B (своя GPU, сетка ``local_grid_v1``).

Берёт подобранную модель (``Evaluation``: train — сетки S1 + S2, test — смеси
S3, перенос — поток запросов S4 и серверная матрица S0), планы фаз (``plan.json``
— параметры каждой точки) и сами попытки (состояние машины, повторы REF).
Возвращает словарь ``local`` для документа отчёта: условия сбора, шум и дрейф,
сетку prefill, сетку decode, смеси по доле prefill, поток запросов, сравнение
с сервером на той же нагрузке, разброс повторов. Всё — по полным данным.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import prices, system_report
from .evaluate import Evaluation
from .split import series_of

R_TARGETS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97)
# A load point is stationary when the server keeps up: it serves at least this
# share of the offered request rate; otherwise the queue grows through the run.
STATIONARY = 0.95
MAX_BATCHED_TOKENS = 16384
FIXED_WAVE = 8


def _plans(paths) -> tuple[dict[str, dict], list[dict]]:
    plans = [json.loads(Path(p).read_text()) for p in paths]
    rows = {row["run_id"]: row for plan in plans for row in plan["matrix"]}
    return rows, plans


def _attempts(roots) -> pd.DataFrame:
    rows = []
    for root in roots:
        for f in sorted(Path(root).glob("*/*/runs.json")):
            run = json.loads(f.read_text())
            state = run.get("machine_state") or {}
            rows.append(
                {
                    "run_id": run["run_id"],
                    "phase": run["run_id"].split("-")[0],
                    "series": series_of(run["run_id"]),
                    "status": run["status"],
                    "wall": (run.get("metrics") or {}).get("wall_seconds"),
                    "start_ns": run.get("generate_start_monotonic_ns"),
                    "end_ns": run.get("generate_end_monotonic_ns"),
                    "sm_min": state.get("sm_clock_mhz_min"),
                    "sm_max": state.get("sm_clock_mhz_max"),
                    "temp_max": state.get("temperature_c_max"),
                    "kv_blocks": (
                        (run.get("effective_config") or {}).get("effective_config")
                        or {}
                    ).get("num_kvcache_blocks"),
                    "bytes": sum(p.stat().st_size for p in f.parent.iterdir()),
                }
            )
    return pd.DataFrame(rows)


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.replace({np.nan: None}).to_dict(orient="records")


def _conditions(attempts, plans, roots) -> dict:
    gpu = json.loads(next(Path(roots[0]).glob("*/*/hardware.json")).read_text()).get(
        "gpu_devices"
    ) or [{}]
    phases = []
    for plan in plans:
        sub = attempts[attempts["phase"] == plan["phase"]]
        phases.append(
            {
                "phase": plan["phase"],
                "attempts": len(sub),
                "success": int((sub["status"] == "success").sum()),
                "throttled": int((sub["status"] == "throttled").sum()),
                "failed": int((~sub["status"].isin(["success", "throttled"])).sum()),
                "series_counts": plan["series_counts"],
                "dropped_points": len({d["workload_id"] for d in plan["dropped"]}),
                "hours": (sub["end_ns"].max() - sub["start_ns"].min()) / 3.6e12,
                "gigabytes": sub["bytes"].sum() / 1e9,
            }
        )
    engine = plans[0]["configuration"]["engine"]
    return {
        "gpu": gpu[0].get("name"),
        "gpu_memory_gb": (gpu[0].get("memory_total_bytes") or 0) / 2**30,
        "sm_clock_min": attempts["sm_min"].min(),
        "sm_clock_max": attempts["sm_max"].max(),
        "temperature_max": attempts["temp_max"].max(),
        "kv_blocks": sorted(attempts["kv_blocks"].dropna().astype(int).unique()),
        "engine": {
            k: engine.get(k)
            for k in (
                "tensor_parallel_size",
                "enforce_eager",
                "gpu_memory_utilization",
                "max_num_seqs",
                "max_num_batched_tokens",
                "max_model_len",
                "kvcache_block_size",
            )
        },
        "phases": phases,
    }


def _reference(attempts) -> dict:
    ref = attempts[(attempts["series"] == "REF") & (attempts["status"] == "success")]
    ref = ref.sort_values("start_ns")
    t0 = ref["start_ns"].min()
    rows = ref.assign(minutes=(ref["start_ns"] - t0) / 6e10)[
        ["run_id", "phase", "minutes", "wall"]
    ]
    stats = [
        {
            "phase": phase,
            "n": len(g),
            "mean": g["wall"].mean(),
            "std_rel": g["wall"].std() / g["wall"].mean(),
            "min": g["wall"].min(),
            "max": g["wall"].max(),
        }
        for phase, g in ref.groupby("phase")
    ]
    return {"runs": _records(rows), "stats": stats}


def _point(row: dict) -> dict:
    spec = row.get("workload_spec") or {}
    reqs = spec.get("requests") or [{}]
    first = reqs[0]
    return {
        "point_id": row["workload_id"],
        "k": len(reqs),
        "L": first.get("L"),
        "O_req": first.get("O"),
        "R": first.get("prefix_blocks", 0) * 256 / first["L"]
        if first.get("L")
        else None,
        "mixed": bool((spec.get("point") or {}).get("mixed")),
        **{k: v for k, v in (spec.get("point") or {}).items() if k != "mixed"},
    }


def _runs_with_points(ev, plan, source=None) -> pd.DataFrame:
    runs = ev.runs if source is None else ev.runs[ev.runs["source"] == source]
    points = pd.DataFrame([{"run_id": r, **_point(plan[r])} for r in runs["run_id"]])
    return runs.merge(points, on="run_id")


def _prefill_grid(ev, plan) -> dict:
    runs = _runs_with_points(ev, plan)
    s1 = runs[runs["run_id"].map(series_of) == "S1"].copy()
    s1["R_target"] = s1["R"].map(lambda r: min(R_TARGETS, key=lambda t: abs(t - r)))
    s1["T_pf_err"] = s1["prefill_hat"] / s1["prefill_seconds"] - 1
    points = (
        s1.groupby(["point_id", "cache", "k", "L", "R", "R_target", "mixed"])
        .agg(
            n=("run_id", "count"),
            T=("prefill_seconds", "median"),
            T_min=("prefill_seconds", "min"),
            T_max=("prefill_seconds", "max"),
            T_hat=("prefill_hat", "median"),
            steps=("n_prefill_steps", "max"),
        )
        .reset_index()
    )
    points["e"] = points["T_hat"] / points["T"] - 1
    points["spread"] = (points["T_max"] - points["T_min"]) / points["T"]
    # Every repeat of the points whose repeats disagree by more than 2x.
    wide = points.loc[points["spread"] > 1, "point_id"]
    repeats = s1[s1["point_id"].isin(wide)].sort_values(["point_id", "run_id"])
    return {
        "points": _records(points),
        "repeat_spread_median": float(points["spread"].median()),
        "mean_abs_e": float(points["e"].abs().mean()),
        "repeats": _records(
            repeats[
                [
                    "point_id",
                    "run_id",
                    "L",
                    "R",
                    "prefill_seconds",
                    "prefill_hat",
                    "n_prefill_steps",
                ]
            ]
        ),
    }


def _decode_grid(ev, plan) -> dict:
    dc = ev.decode_steps
    dc = dc[dc["run_id"].map(series_of) == "S2"]
    per_run = (
        dc.groupby("run_id")
        .agg(B=("B", "first"), T=("T", "median"), T_hat=("pred", "median"))
        .reset_index()
    )
    per_run["L"] = per_run["run_id"].map(lambda r: _point(plan[r])["L"])
    points = (
        per_run.groupby(["L", "B"])
        .agg(
            n=("run_id", "count"),
            T=("T", "median"),
            T_min=("T", "min"),
            T_max=("T", "max"),
            T_hat=("T_hat", "median"),
        )
        .reset_index()
    )
    points["e"] = points["T_hat"] / points["T"] - 1
    points["otps"] = points["B"] / points["T"]
    points["otps_hat"] = points["B"] / points["T_hat"]
    points["spread"] = (points["T_max"] - points["T_min"]) / points["T"]
    return {"points": _records(points), "mean_abs_e": float(points["e"].abs().mean())}


def _repeat_spread(prefill: dict, decode: dict, reference: dict) -> list[dict]:
    """How far identical repeats disagree: (max − min) / median per point."""
    rows = []
    for name, points, what in (
        ("S1 prefill", prefill["points"], "T_pf прогона, 5 повторов"),
        ("S2 decode", decode["points"], "медианный decode-шаг, 3 повтора"),
    ):
        spread = pd.Series([p["spread"] for p in points])
        rows.append(
            {
                "series": name,
                "what": what,
                "points": len(spread),
                "median": float(spread.median()),
                "p90": float(spread.quantile(0.9)),
                "max": float(spread.max()),
            }
        )
    walls = pd.Series([r["wall"] for r in reference["runs"]])
    rows.append(
        {
            "series": "REF",
            "what": f"T^wall, {len(walls)} повторов за весь сбор",
            "points": 1,
            "median": float((walls.max() - walls.min()) / walls.median()),
            "p90": None,
            "max": None,
        }
    )
    return rows


def _mixes(ev, plan) -> dict:
    runs = _runs_with_points(ev, plan)
    s3 = runs[runs["run_id"].map(series_of) == "S3"].copy()
    s3["share"] = s3["prefill_seconds"] / s3["steps_seconds"]
    s3["share_hat"] = s3["prefill_hat"] / s3["pred"]
    s3["target"] = s3["target_prefill_share"]
    cols = ["run_id", "N", "L", "O_req", "cache", "target", "share", "share_hat"]
    by_n = (
        s3.groupby("N")
        .agg(
            n=("run_id", "count"),
            mean_abs=("rel_err", lambda e: e.abs().mean()),
            mean_signed=("rel_err", "mean"),
            share_dev=("share", "mean"),
            target=("target", "mean"),
        )
        .reset_index()
    )
    by_n["share_dev"] = by_n["share_dev"] - by_n["target"]
    by_target = (
        s3.groupby("target")
        .agg(
            n=("run_id", "count"),
            share=("share", "mean"),
            share_hat=("share_hat", "mean"),
            mean_abs=("rel_err", lambda e: e.abs().mean()),
            mean_signed=("rel_err", "mean"),
        )
        .reset_index()
    )
    return {
        "runs": _records(s3[[*cols, "rel_err", "wall_seconds"]]),
        "by_n": _records(by_n),
        "by_target": _records(by_target),
        "share_dev_median": float((s3["share"] - s3["target"]).abs().median()),
    }


def _arrivals(ev, plan, source, requests) -> dict:
    runs = _runs_with_points(ev, plan, source)
    req = requests[requests["source"] == source]
    ttft = (
        req.groupby("run_id")
        .agg(
            ttft=("ttft", "median"),
            ttft_p90=("ttft", lambda v: v.quantile(0.9)),
            ttft_hat=("ttft_hat", "median"),
            ttft_hat_p90=("ttft_hat", lambda v: v.quantile(0.9)),
            n=("request_id", "count"),
        )
        .reset_index()
    )
    runs = runs.merge(ttft, on="run_id")
    # Rates from the trace: offered — the realised arrival rate of the run (a
    # Poisson sample of 256 differs from the nominal ρ·capacity), served —
    # requests per second from the first arrival to the last answer.
    times = (
        ev.datasets[source]
        .requests.groupby("run_id")
        .agg(
            first=("enqueue_ns", "min"),
            last=("enqueue_ns", "max"),
            end=("finish_ns", "max"),
        )
    )
    runs = runs.merge(times.reset_index(), on="run_id")
    runs["nominal_rps"] = runs["rho"] * runs["capacity_rps"]
    runs["offered_rps"] = (runs["n"] - 1) / ((runs["last"] - runs["first"]) * 1e-9)
    runs["achieved_rps"] = runs["n"] / ((runs["end"] - runs["first"]) * 1e-9)
    cols = [
        "run_id",
        "rho",
        "scheduler",
        "cache",
        "nominal_rps",
        "offered_rps",
        "achieved_rps",
        "ttft",
        "ttft_p90",
        "ttft_hat",
        "ttft_hat_p90",
        "wall_seconds",
        "pred",
        "rel_err",
    ]
    table = (
        runs.groupby(["rho", "scheduler", "cache"])
        .agg(
            n=("run_id", "count"),
            nominal_rps=("nominal_rps", "mean"),
            offered_rps=("offered_rps", "mean"),
            achieved_rps=("achieved_rps", "mean"),
            ttft=("ttft", "mean"),
            ttft_hat=("ttft_hat", "mean"),
            ttft_p90=("ttft_p90", "mean"),
            rel_err=("rel_err", "mean"),
        )
        .reset_index()
    )
    table["served_share"] = table["achieved_rps"] / table["offered_rps"]
    table["stationary"] = table["served_share"] >= STATIONARY
    return {
        "runs": _records(runs[cols]),
        "table": _records(table),
        "capacity_rps": float(runs["capacity_rps"].iloc[0]),
        "requests_per_run": int(runs["n"].max()),
        "stationary_share": STATIONARY,
    }


def _ttft_by_rank(requests, source, table) -> list[dict]:
    """TTFT by the request's number in its run for load points past the knee:
    median over the seeds of each (ρ, scheduler, cache)."""
    late = {
        (r["rho"], r["scheduler"], r["cache"]) for r in table if not r["stationary"]
    }
    req = requests[requests["source"] == source]
    req = req.assign(
        key=list(zip(req["rho"], req["scheduler"], req["cache"], strict=True))
    )
    req = req[req["key"].isin(late)]
    out = (
        req.groupby(["rho", "scheduler", "cache", "arrival_rank"])
        .agg(ttft=("ttft", "median"), ttft_hat=("ttft_hat", "median"))
        .reset_index()
    )
    return _records(out)


def _tpot_split(ev, source, requests) -> list[dict]:
    """TPOT of upstream_dynamic without cache by ρ, split into the request's
    own decode steps and other requests' prefill steps between its first and
    last token (the rest — time between steps). Means over requests, so the
    parts add up to the mean TPOT; measured and by the model."""
    keys = ["run_id", "step_index", "T", "pred"]
    steps = pd.concat(
        [
            ev.prefill_steps.loc[ev.prefill_steps["source"] == source, keys].assign(
                pf=1
            ),
            ev.decode_steps.loc[ev.decode_steps["source"] == source, keys].assign(pf=0),
        ]
    ).sort_values(["run_id", "step_index"])
    data = ev.datasets[source].requests
    req = requests[
        (requests["source"] == source)
        & (requests["scheduler"] == "upstream_dynamic")
        & (requests["cache"] == "off")
        & requests["tpot"].notna()
    ].merge(data[["run_id", "request_id", "first_step", "last_step"]])
    rows = []
    for run_id, r in req.groupby("run_id"):
        st = steps[steps["run_id"] == run_id]
        idx = st["step_index"].to_numpy()
        cum = {
            name: np.concatenate([[0.0], np.cumsum(vals)])
            for name, vals in (
                ("dc", st["T"].where(st["pf"] == 0, 0.0)),
                ("pf", st["T"].where(st["pf"] == 1, 0.0)),
                ("dc_hat", st["pred"].where(st["pf"] == 0, 0.0)),
                ("pf_hat", st["pred"].where(st["pf"] == 1, 0.0)),
            )
        }
        # Steps after the first-token step up to and including the last one.
        lo = np.searchsorted(idx, r["first_step"].to_numpy(), side="right")
        hi = np.searchsorted(idx, r["last_step"].to_numpy(), side="right")
        gaps = (r["O"] - 1).to_numpy(float)
        part = {k: (v[hi] - v[lo]) / gaps for k, v in cum.items()}
        rows.append(
            pd.DataFrame(
                {"rho": r["rho"].to_numpy(), "tpot": r["tpot"].to_numpy(), **part}
            )
        )
    if not rows:
        return []
    frame = pd.concat(rows)
    frame["other"] = frame["tpot"] - frame["dc"] - frame["pf"]
    out = frame.groupby("rho").mean().reset_index()
    out["n"] = frame.groupby("rho").size().to_numpy()
    return _records(out)


def _server(ev, plan, source, server_runs_dir) -> dict:
    server = {}
    for f in Path(server_runs_dir).glob("*/*/runs.json"):
        run = json.loads(f.read_text())
        if run.get("status") == "success":
            server[run["run_id"]] = run["metrics"]["wall_seconds"]
    runs = ev.runs[ev.runs["source"] == source].copy()
    runs["server_run_id"] = runs["run_id"].map(lambda r: plan[r]["server_run_id"])
    runs["server_wall"] = runs["server_run_id"].map(server)
    runs["ratio"] = runs["wall_seconds"] / runs["server_wall"]
    cols = [
        "run_id",
        "server_run_id",
        "set_id",
        "seed",
        "condition_id",
        "cache",
        "scheduler",
        "wall_seconds",
        "server_wall",
        "ratio",
        "pred",
        "rel_err",
    ]
    by_condition = (
        runs.groupby(["condition_id", "cache", "scheduler"])
        .agg(n=("run_id", "count"), ratio=("ratio", "median"), e=("rel_err", "mean"))
        .reset_index()
    )
    return {
        "runs": _records(runs[cols]),
        "by_condition": _records(by_condition),
        "ratio_median": float(runs["ratio"].median()),
    }


# S5: closed loop of N requests. Points shown as examples of the number of
# requests in the system over time: (scheduler, cache, N), first seed.
S5_EXAMPLES = (
    ("upstream_dynamic", "off", 8),
    ("upstream_dynamic", "on", 24),
    ("fixed_batch", "off", 12),
)
KV_FULL = 0.95
SPREAD = ["rps", "itps", "itps_all", "otps", "ttft", "ttft_p90", "ttps", "tpot"]
SPREAD += ["batch", "kv_max", "e_rps", "e_ttft", "e_tpot"]


def _s5_attempts(roots) -> dict[str, Path]:
    """Successful S5 attempt directory of each run id."""
    out = {}
    for root in roots:
        for f in sorted(Path(root).glob("*-S5-*/*/runs.json")):
            run = json.loads(f.read_text())
            if run.get("status") == "success" and run.get("closed_loop"):
                out[run["run_id"]] = f.parent
    return out


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _in_system(enq: np.ndarray, fin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Step function of requests in the system: change times and counts after."""
    t = np.concatenate([enq, fin])
    d = np.concatenate([np.ones(len(enq)), -np.ones(len(fin))])
    order = np.lexsort((d, t))  # at equal times a finish before a submit
    t, d = t[order], d[order]
    return t, np.cumsum(d)


def _share_at(t, count, n, lo, hi) -> float:
    """Share of [lo, hi] with exactly n requests in the system."""
    edges = np.clip(np.concatenate([t, [hi]]), lo, hi)
    dur = np.diff(edges)
    return float(dur[count == n].sum() / (hi - lo))


def _overlap(steps: pd.DataFrame, lo: float, hi: float) -> np.ndarray:
    a, b = steps["t_start_ns"].to_numpy(float), steps["t_end_ns"].to_numpy(float)
    return np.clip((np.minimum(b, hi) - np.maximum(a, lo)) / np.maximum(b - a, 1), 0, 1)


def _closed_loop_run(ev, source, run, attempt, req_frame) -> dict:
    loop = json.loads((attempt / "runs.json").read_text())["closed_loop"]
    lo, hi = float(loop["window_start_ns"]), float(loop["window_end_ns"])
    T_w = (hi - lo) * 1e-9
    run_id = run["run_id"]
    pf = ev.prefill_steps[
        (ev.prefill_steps["run_id"] == run_id) & (ev.prefill_steps["source"] == source)
    ]
    dc = ev.decode_steps[
        (ev.decode_steps["run_id"] == run_id) & (ev.decode_steps["source"] == source)
    ]
    fp, fd = _overlap(pf, lo, hi), _overlap(dc, lo, hi)
    pf_t, dc_t = float((pf["T"] * fp).sum()), float((dc["T"] * fd).sum())
    pf_hat, dc_hat = float((pf["pred"] * fp).sum()), float((dc["pred"] * fd).sum())
    T_hat = pf_hat + dc_hat + (T_w - pf_t - dc_t)  # gaps between steps as measured
    new = float((pf["new"] * fp).sum())
    served = float((pf["served"] * fp).sum())
    out_tokens = float((dc["B"] * fd).sum() + (pf["B"] * fp).sum())
    m = int(loop["measure"])

    data = ev.datasets[source].requests
    reqs = data[data["run_id"] == run_id].sort_values("finish_ns")
    warm = int(loop["warmup"])
    measured = set(reqs["request_id"].iloc[warm : warm + m])
    # Completions inside the window: the window ends on step boundaries, so with
    # requests finishing in bursts (equal L and O) this is not the planned m.
    done = int(((reqs["finish_ns"] > lo) & (reqs["finish_ns"] <= hi)).sum())
    t, count = _in_system(
        reqs["enqueue_ns"].to_numpy(float), reqs["finish_ns"].to_numpy(float)
    )
    rq = req_frame[
        (req_frame["run_id"] == run_id) & req_frame["request_id"].isin(measured)
    ]

    kv = pd.DataFrame(_jsonl(attempt / "kv_blocks.jsonl"))
    kv["step_index"] = kv["step_id"].str.split("-").str[1].astype(int)
    in_win = set(pf.loc[fp > 0, "step_index"]) | set(dc.loc[fd > 0, "step_index"])
    kv_w = kv[kv["step_index"].isin(in_win)]
    preempted = sum(
        len(e.get("preempted_request_ids") or [])
        for e in _jsonl(attempt / "scheduler_events.jsonl")
        if e.get("event_type") == "schedule" and lo <= e["monotonic_ns"] <= hi
    )
    tpot = rq["tpot"].median()
    tpot_hat = rq["tpot_hat"].median()
    row = {
        "run_id": run_id,
        "seed": int(run["seed"]),
        "scheduler": run["scheduler"],
        "cache": run["cache"],
        "N": int(loop["N"]),
        "measure": m,
        "completed": done,
        "T_window": T_w,
        "T_window_hat": T_hat,
        "rps": done / T_w,
        "rps_hat": done / T_hat,
        "itps": new / T_w,
        "itps_hat": new / T_hat,
        "itps_all": served / T_w,
        "otps": out_tokens / T_w,
        "otps_hat": out_tokens / T_hat,
        "ttft": float(rq["ttft"].median()),
        "ttft_p90": float(rq["ttft"].quantile(0.9)),
        "ttft_hat": float(rq["ttft_hat"].median()),
        "ttft_hat_p90": float(rq["ttft_hat"].quantile(0.9)),
        "tpot": float(tpot),
        "tpot_hat": float(tpot_hat),
        "ttps": float(1 / tpot),
        "ttps_hat": float(1 / tpot_hat),
        "batch": float(np.average(dc["B"], weights=fd)) if fd.sum() else None,
        "ell": float(np.average(dc["sum_l"] / dc["B"], weights=fd))
        if fd.sum()
        else None,
        "prefill_share": pf_t / (pf_t + dc_t),
        "kv_max": float(kv_w["used_block_count"].max() / kv_w["capacity_blocks"].max()),
        "preempted": preempted,
        "exact_n_share": _share_at(t, count, int(loop["N"]), lo, hi),
        # Model prices of the window: ã = ΣT̂_pf / I_new, b̃ = ΣT̂_dc / O.
        "a_tilde": pf_hat / new if new else None,
        "b_tilde": dc_hat / out_tokens if out_tokens else None,
        "L_mean": float(reqs["prompt"].mean()),
        "new_per_request": new / done,
        "out_per_request": out_tokens / done,
    }
    for name in ("rps", "itps", "otps", "ttft", "tpot"):
        row[f"e_{name}"] = row[f"{name}_hat"] / row[name] - 1
    row["timeline"] = {
        "t": ((t - t[0]) * 1e-9).tolist(),
        "count": count.astype(int).tolist(),
        "window": [(lo - t[0]) * 1e-9, (hi - t[0]) * 1e-9],
    }
    return row


def _closed_loop(ev, source, requests, roots, kv_tokens) -> dict:
    """S5: closed loop of exactly N requests, metrics in the measurement window.

    Window — from the completion of the last warm-up request to that of the
    last measured one. Rates: completions inside the window (not the planned
    measure: with bursts of equal requests the two differ), new (not cached)
    and all prompt tokens of prefill steps, output tokens (decode B plus first
    tokens) per second of the window; steps crossing a border are split by
    overlap. Model:
    predicted steps of the same schedule, window time T̂ = ΣT̂_steps plus the
    measured gaps between steps; TTFT and TPOT of measured requests by
    ``prices.requests``. Point = (scheduler, cache, N): mean and spread (std)
    over the seeds.
    """
    attempts = _s5_attempts(roots)
    runs = ev.runs[ev.runs["source"] == source]
    req = requests[requests["source"] == source]
    rows = [
        _closed_loop_run(ev, source, r, attempts[r["run_id"]], req)
        for r in runs.to_dict(orient="records")
        if r["run_id"] in attempts
    ]
    timelines = {r["run_id"]: r.pop("timeline") for r in rows}
    frame = pd.DataFrame(rows)
    keys = ["scheduler", "cache", "N"]
    value_cols = [c for c in frame.columns if c not in keys + ["run_id", "seed"]]
    grouped = frame.groupby(keys)
    table = grouped[value_cols].mean().reset_index()
    spread = grouped[SPREAD].std().add_suffix("_sd").reset_index(drop=True)
    table = pd.concat([table, spread], axis=1)
    table["n"] = grouped.size().to_numpy()
    table["exact_n_min"] = grouped["exact_n_share"].min().to_numpy()

    coef = {**ev.prefill.coef, **ev.decode.coef}
    variants = []
    for (sched, cache), g in table.groupby(["scheduler", "cache"]):
        g = g.sort_values("N")
        top = g.iloc[-1]

        # Saturation: the smallest N whose RPS / OTPS is within 5% of the top.
        def knee(col, g=g):
            best = g[col].max()
            return int(g.loc[g[col] >= 0.95 * best, "N"].min())

        full = g[(g["preempted"] > 0) | (g["kv_max"] >= KV_FULL)]
        sub = frame[(frame["scheduler"] == sched) & (frame["cache"] == cache)]
        R = 1 - float(sub["new_per_request"].mean() / sub["L_mean"].mean())
        L = float(sub["L_mean"].mean())
        new_req = float(sub["new_per_request"].mean())
        out_req = float(sub["out_per_request"].mean())

        def ceiling(B, ell, R=R, L=L, new_req=new_req, out_req=out_req):
            a_t, b_t = prices.token_prices(coef, L, R, B, ell)
            return {
                "B": B,
                "a_tilde": a_t,
                "b_tilde": b_t,
                "rps": 1 / (a_t * new_req + b_t * out_req),
            }

        # Line at the batch seen in the trace at the top N: a replay check of the
        # step model, not a prediction (the model "from N" is in ``system``).
        at_top = ceiling(float(top["batch"]), float(top["ell"]))
        # TTPS–TTFT: log-log slope over N ≥ 2 (−1 for a hyperbola).
        h = g[g["N"] >= 2]
        slope = (
            float(np.polyfit(np.log(h["ttft"]), np.log(h["ttps"]), 1)[0])
            if len(h) >= 3
            else None
        )
        variants.append(
            {
                "scheduler": sched,
                "cache": cache,
                "knee_rps": knee("rps"),
                "knee_otps": knee("otps"),
                "rps_top": float(top["rps"]),
                "otps_top": float(top["otps"]),
                "n_top": int(top["N"]),
                "n_kv": int(full["N"].min()) if len(full) else None,
                "kv_max_top": float(top["kv_max"]),
                "line_top": at_top,
                "e_ceiling": at_top["rps"] / float(top["rps"]) - 1,
                "ttps_ttft_slope": slope,
                "R": R,
                "L": L,
                "new_per_request": new_req,
                "out_per_request": out_req,
            }
        )
    examples = []
    for sched, cache, n in S5_EXAMPLES:
        pick = frame[
            (frame["scheduler"] == sched)
            & (frame["cache"] == cache)
            & (frame["N"] == n)
        ].sort_values("seed")
        if len(pick):
            rid = pick.iloc[0]["run_id"]
            examples.append(
                {
                    "scheduler": sched,
                    "cache": cache,
                    "N": n,
                    "run_id": rid,
                    **timelines[rid],
                }
            )
    return {
        "source": source,
        "runs": _records(frame),
        "table": _records(table),
        "variants": variants,
        "examples": examples,
        "kv_tokens": kv_tokens,
        "kv_full": KV_FULL,
        "exact_n_required": 0.95,
    }


def build_local(
    ev: Evaluation,
    roots: list[str | Path],
    plan_paths: list[str | Path],
    server_runs_dir: str | Path,
    labels: dict[str, str],
    system: dict | None = None,
    economics: dict | None = None,
) -> dict:
    """The ``local`` block of the 0.6B local model in the report document;
    with ``system`` (and ``economics``) it carries the model "from N"."""
    plan, plans = _plans(plan_paths)
    attempts = _attempts(roots)
    by_series = {}
    for source, data in ev.datasets.items():
        for s in {series_of(r) for r in data.runs["run_id"]}:
            by_series[s] = source
    requests = prices.requests(ev)
    reference = _reference(attempts)
    conditions = _conditions(attempts, plans, roots)
    arrivals = _arrivals(ev, plan, by_series["S4"], requests)
    rho = {r["run_id"]: r["rho"] for r in arrivals["runs"]}
    s4_requests = requests[requests["source"] == by_series["S4"]].assign(
        rho=lambda f: f["run_id"].map(rho)
    )
    prefill_grid = _prefill_grid(ev, plan)
    decode_grid = _decode_grid(ev, plan)
    return {
        "conditions": conditions,
        "reference": reference,
        "repeat_spread": _repeat_spread(prefill_grid, decode_grid, reference),
        "prefill_grid": prefill_grid,
        "decode_grid": decode_grid,
        "mixes": _mixes(ev, plan),
        "arrivals": arrivals,
        "ttft_by_rank": _ttft_by_rank(s4_requests, by_series["S4"], arrivals["table"]),
        "tpot_split": _tpot_split(ev, by_series["S4"], s4_requests),
        "server": _server(ev, plan, by_series["S0"], server_runs_dir),
        **(
            {
                "closed_loop": _closed_loop(
                    ev,
                    by_series["S5"],
                    requests,
                    roots,
                    conditions["kv_blocks"][0]
                    * conditions["engine"]["kvcache_block_size"],
                )
            }
            if "S5" in by_series
            else {}
        ),
        **(
            {"system": system_report.build(ev, plan, system, economics)}
            if system
            else {}
        ),
    }
