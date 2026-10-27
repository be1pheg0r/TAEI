"""Загружает успешные прогоны nano-vLLM и строит таблицы прогонов, шагов и запросов.

Для каждого прогона читаются ``runs.json``, ``steps.jsonl``,
``step_requests.jsonl``, ``requests.jsonl`` и ``token_events.jsonl``. Время
шага — полное, ``step_host_end_ns - step_host_start_ns`` (часы хоста).

Для запроса в prefill-шаге C — токены, чей KV уже есть до шага
(``private_kv_cached_tokens``, то есть ``num_cached_tokens`` последовательности):
чужой префикс из кэша (``prefix_reused_tokens`` входит сюда, он не больше C) и
свой предыдущий кусок chunked prefill; N — токены, посчитанные в шаге: новые
(``fresh_prefill_tokens``) и пересчитанные после вытеснения
(``recompute_tokens``); L = C + N — длина запроса в KV после шага. У последнего
куска (шаг выдаёт токен) L совпадает с ``context_tokens_before``, у
промежуточного — меньше. Для decode-шага берётся текущая длина контекста ℓ
каждого запроса и число запросов B.

I — все токены промптов прогона, включая кэш (``input_tokens``), O — фактически
выданные токены. В fixed_batch сборщик может потерять вытесненный запрос: тогда
O меньше запрошенного, разница — ``lost_tokens``; шаги трассы при этом верны.
Состояние машины за прогон (медианы частоты SM, загрузки и мощности видимых GPU
из ``telemetry.jsonl``, число KV-блоков) идёт только в примечания отчёта.
"""

from __future__ import annotations

import json
import re
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
    requests: pd.DataFrame  # one row per request: timestamps, O, first/last step


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _prefill_requests(rows: list[dict], run_id: str) -> pd.DataFrame:
    reqs = pd.DataFrame([r for r in rows if r["relation"] == "prefill"])
    reqs["C"] = reqs["private_kv_cached_tokens"].astype(float)
    reqs["N"] = (reqs["fresh_prefill_tokens"] + reqs["recompute_tokens"]).astype(float)
    reqs["L"] = reqs["C"] + reqs["N"]
    last = reqs["produced_tokens"] > 0
    bad = (
        (last & (reqs["L"] != reqs["context_tokens_before"]))
        | (~last & (reqs["L"] >= reqs["context_tokens_before"]))
        | (reqs["prefix_reused_tokens"] > reqs["C"])
    )
    if bad.any():
        raise ValueError(
            f"{run_id}: L inconsistent with context_tokens_before in {bad.sum()} rows"
        )
    reqs["R"] = reqs["C"] / reqs["L"]
    reqs["new"] = reqs["N"]  # L(1 - R)
    reqs["quad"] = reqs["L"] ** 2 - reqs["C"] ** 2  # L^2 (1 - R^2)
    return reqs[["step_id", "request_id", "L", "C", "R", "new", "quad"]]


def _gpu_state(attempt: Path) -> dict[str, float]:
    """Run medians of SM clock, utilization and power over the run's visible GPUs."""
    env = json.loads((attempt / "environment.json").read_text())
    visible = set(str(env.get("cuda_visible_devices", "")).split(","))
    gpus = [
        g
        for sample in _jsonl(attempt / "telemetry.jsonl")
        for g in sample.get("gpu_devices") or []
        if g.get("uuid") in visible
    ]
    median = lambda key: float(np.median([g[key] for g in gpus])) if gpus else np.nan  # noqa: E731
    return {
        "gpu_sm_mhz": median("sm_clock_mhz"),
        "gpu_util": median("utilization_gpu_percent"),
        "gpu_power_w": median("power_w"),
    }


def _requests(attempt: Path, step_index: dict[str, int]) -> pd.DataFrame:
    tokens = pd.DataFrame(_jsonl(attempt / "token_events.jsonl"))
    tokens["step_index"] = tokens["step_id"].map(step_index)
    span = tokens.groupby("request_id")["step_index"].agg(
        first_step="min", last_step="max"
    )
    rows = [
        {
            "request_id": q["request_id"],
            "prompt": q["prompt_length"],
            "O": q["actual_output_tokens"],
            "O_requested": q["requested_output_tokens"],
            "finish_reason": q["finish_reason"],
            "enqueue_ns": q["enqueue_monotonic_ns"],
            "first_ns": q["first_host_observed_monotonic_ns"],
            "finish_ns": q["finish_monotonic_ns"],
        }
        for q in _jsonl(attempt / "requests.jsonl")
    ]
    return pd.DataFrame(rows).merge(span.reset_index(), on="request_id", how="left")


def load_run(attempt: Path, model: str) -> dict[str, pd.DataFrame | dict]:
    run = json.loads((attempt / "runs.json").read_text())
    run_id = run["run_id"]
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
    steps["run_id"] = run_id
    steps["model"] = model

    reqs = _prefill_requests(_jsonl(attempt / "step_requests.jsonl"), run_id)
    reqs = reqs.assign(run_id=run_id, model=model)
    prefill = steps[steps["phase"] == "prefill"].merge(
        reqs.groupby("step_id")[["new", "quad"]].sum().reset_index(), on="step_id"
    )
    decode = steps[steps["phase"] == "decode"].copy()
    requests = _requests(
        attempt, dict(zip(steps["step_id"], steps["step_index"], strict=True))
    )
    requests = requests.assign(run_id=run_id, model=model)

    metrics = run["metrics"]
    row = {
        "run_id": run_id,
        "model": model,
        "set_id": run["set_id"],
        "seed": run["seed"],
        "condition_id": run["condition_id"],
        "workload": re.sub(r"-seed-\d+$", "", run["workload_id"]),
        "cache": run["prefix_cache_mode"],
        "scheduler": run["scheduler_mode"],
        "wall_seconds": metrics["wall_seconds"],
        "I": metrics["input_tokens"],
        "O": metrics["output_tokens"],
        "O_requested": metrics["requested_output_tokens"],
        "lost_tokens": metrics["requested_output_tokens"] - metrics["output_tokens"],
        "steps_seconds": float(steps["T"].sum()),
        "prefill_seconds": float(prefill["T"].sum()),
        "decode_seconds": float(decode["T"].sum()),
        "n_prefill_steps": len(prefill),
        "n_decode_steps": len(decode),
        # Run-level L and R over its prefill rows (R = sum C / sum L).
        "L_mean": float(requests["prompt"].mean()),
        "R_run": float(reqs["C"].sum() / reqs["L"].sum()),
        # Machine state, for notes only: the model does not use it.
        "kv_blocks": (
            (run.get("effective_config") or {}).get("effective_config") or {}
        ).get("num_kvcache_blocks"),
        **_gpu_state(attempt),
    }
    return {
        "run": row,
        "prefill": prefill,
        "requests": reqs,
        "decode": decode,
        "request_table": requests,
    }


def attempts(root: Path | str) -> pd.DataFrame:
    """Status, cache mode and error type of every attempt under ``root``."""
    rows = []
    for runs_json in sorted(Path(root).glob("*/*/runs.json")):
        info = json.loads(runs_json.read_text())
        rows.append(
            {
                "run_id": info["run_id"],
                "status": info.get("status"),
                "cache": info.get("prefix_cache_mode"),
                "error": (info.get("error") or {}).get("type"),
            }
        )
    return pd.DataFrame(rows)


def load(root: Path | str, model: str, run_ids: list[str] | None = None) -> Dataset:
    """Load successful runs under ``root`` (a ``runs/`` folder of one launch)."""
    parts: dict[str, list] = {
        "run": [],
        "prefill": [],
        "requests": [],
        "decode": [],
        "request_table": [],
    }
    for runs_json in sorted(Path(root).glob("*/*/runs.json")):
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
        requests=pd.concat(parts["request_table"], ignore_index=True),
    )
