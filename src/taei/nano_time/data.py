"""Загружает успешные прогоны nano-vLLM и строит таблицы прогонов и шагов.

Для каждого прогона читаются ``runs.json``, ``environment.json``,
``steps.jsonl``, ``step_requests.jsonl`` и ``telemetry.jsonl``. Время шага —
полное, ``step_host_end_ns - step_host_start_ns`` (часы хоста). Для запроса в
prefill-шаге C — токены из кэша до шага, L = C + новые токены (длина запроса в
KV после шага; совпадает с ``context_tokens_before``, а ``context_tokens_after``
на единицу больше из-за сэмплированного токена). Для decode-шага берётся
текущая длина контекста ℓ каждого запроса и число запросов B. Частота SM шага —
среднее по видимым GPU прогона в ближайшей по времени выборке телеметрии.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    runs: pd.DataFrame  # one row per run
    prefill_steps: pd.DataFrame  # one row per prefill step
    prefill_requests: pd.DataFrame  # one row per (prefill step, request)
    decode_steps: pd.DataFrame  # one row per decode step


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _telemetry(path: Path, visible: set[str]) -> pd.DataFrame:
    rows = []
    for sample in _jsonl(path):
        gpus = [g for g in sample.get("gpu_devices") or [] if g.get("uuid") in visible]
        clocks = [g["sm_clock_mhz"] for g in gpus if g.get("sm_clock_mhz") is not None]
        if not clocks:
            continue
        rows.append(
            {
                "t_ns": sample["monotonic_ns"],
                "sm_clock_mhz": float(np.mean(clocks)),
                "temperature_c": float(np.mean([g["temperature_c"] for g in gpus])),
                "power_w": float(np.mean([g["power_w"] for g in gpus])),
            }
        )
    return pd.DataFrame(rows).sort_values("t_ns").reset_index(drop=True)


def _nearest(tel: pd.DataFrame, t_ns: np.ndarray, column: str) -> np.ndarray:
    if tel.empty:
        return np.full(len(t_ns), np.nan)
    times = tel["t_ns"].to_numpy()
    idx = np.searchsorted(times, t_ns)
    left = np.clip(idx - 1, 0, len(times) - 1)
    right = np.clip(idx, 0, len(times) - 1)
    pick = np.where(
        np.abs(times[left] - t_ns) <= np.abs(times[right] - t_ns), left, right
    )
    return tel[column].to_numpy()[pick]


def load_run(attempt: Path, model: str) -> dict[str, pd.DataFrame | dict]:
    run = json.loads((attempt / "runs.json").read_text())
    env = json.loads((attempt / "environment.json").read_text())
    visible = set(filter(None, str(env.get("cuda_visible_devices", "")).split(",")))
    run_id = run["run_id"]
    tel = _telemetry(attempt / "telemetry.jsonl", visible)

    steps = pd.DataFrame(
        [
            {
                "step_id": s["step_id"],
                "step_index": s["step_index"],
                "phase": s["phase"],
                "t_start_ns": s["step_host_start_ns"],
                "t_end_ns": s["step_host_end_ns"],
                "B": s["actual_batch_size"],
                "sum_l": float(sum(s["contexts_before"].values())),
            }
            for s in _jsonl(attempt / "steps.jsonl")
        ]
    )
    steps["T"] = (steps["t_end_ns"] - steps["t_start_ns"]) * 1e-9
    mid = ((steps["t_start_ns"] + steps["t_end_ns"]) // 2).to_numpy()
    for column in ("sm_clock_mhz", "temperature_c", "power_w"):
        steps[column] = _nearest(tel, mid, column)
    steps["run_id"] = run_id
    steps["model"] = model

    reqs = pd.DataFrame(
        [
            r
            for r in _jsonl(attempt / "step_requests.jsonl")
            if r["relation"] == "prefill"
        ]
    )
    reqs["C"] = reqs["prefix_reused_tokens"].astype(float)
    reqs["L"] = reqs["C"] + reqs["fresh_prefill_tokens"].astype(float)
    mismatch = (reqs["L"] != reqs["context_tokens_before"]).sum()
    if mismatch:
        raise ValueError(f"{run_id}: L != context_tokens_before in {mismatch} rows")
    reqs["R"] = reqs["C"] / reqs["L"]
    reqs["new"] = reqs["L"] - reqs["C"]  # L(1 - R)
    reqs["quad"] = reqs["L"] ** 2 - reqs["C"] ** 2  # L^2 (1 - R^2)
    reqs = reqs[["step_id", "request_id", "L", "C", "R", "new", "quad"]].assign(
        run_id=run_id, model=model
    )

    prefill = steps[steps["phase"] == "prefill"].merge(
        reqs.groupby("step_id")[["new", "quad"]].sum().reset_index(), on="step_id"
    )
    decode = steps[steps["phase"] == "decode"].copy()

    metrics = run["metrics"]
    new_input = metrics["input_tokens"] - metrics["cache_hit_tokens"]  # P
    output = metrics["output_tokens"]  # O
    row = {
        "run_id": run_id,
        "model": model,
        "set_id": run["set_id"],
        "seed": run["seed"],
        "condition_id": run["condition_id"],
        "cache": run["prefix_cache_mode"],
        "scheduler": run["scheduler_mode"],
        "wall_seconds": metrics["wall_seconds"],
        "P": new_input,
        "O": output,
        "p_decode": output / (new_input + output),
        "steps_seconds": float(steps["T"].sum()),
        "prefill_seconds": float(prefill["T"].sum()),
        "decode_seconds": float(decode["T"].sum()),
        "n_prefill_steps": len(prefill),
        "n_decode_steps": len(decode),
        # Run-level L and R over its prefilled requests (R = sum C / sum L).
        "L_mean": float(reqs["L"].mean()),
        "R_run": float(reqs["C"].sum() / reqs["L"].sum()),
        "sm_clock_mhz_median": float(np.median(tel["sm_clock_mhz"]))
        if len(tel)
        else np.nan,
    }
    return {"run": row, "prefill": prefill, "requests": reqs, "decode": decode}


def load(root: Path | str, model: str, run_ids: list[str] | None = None) -> Dataset:
    """Load successful runs under ``root`` (a ``runs/`` folder of one launch)."""
    parts: dict[str, list] = {"run": [], "prefill": [], "requests": [], "decode": []}
    for runs_json in sorted(Path(root).glob("R*/*/runs.json")):
        info = json.loads(runs_json.read_text())
        if info.get("status") != "success":
            continue
        if run_ids is not None and info["run_id"] not in run_ids:
            continue
        loaded = load_run(runs_json.parent, model)
        for key in parts:
            parts[key].append(loaded[key])
    if not parts["run"]:
        raise ValueError(f"no successful runs under {root}")
    return Dataset(
        runs=pd.DataFrame(parts["run"]),
        prefill_steps=pd.concat(parts["prefill"], ignore_index=True),
        prefill_requests=pd.concat(parts["requests"], ignore_index=True),
        decode_steps=pd.concat(parts["decode"], ignore_index=True),
    )
