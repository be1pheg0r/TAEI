"""Local Qwen3-0.6B grid: plan or run one phase, several attempts per process."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from taei import logging as project_logging
from taei.launch import maybe_launch_detached
from taei.nano_dataset.isolation import run_isolated
from taei.nano_dataset.local_grid import build_local_plan
from taei.runtime import run_experiment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.nano_vllm_dataset.run import (  # noqa: E402
    _download_model,
    _engine_config,
    _prepare_model,
    _save_json,
)


def _plan(cfg: DictConfig) -> dict:
    coef = cfg.dataset.coef
    return build_local_plan(
        str(cfg.dataset.phase),
        kv_blocks=cfg.dataset.kv_blocks,
        kv_fill=float(cfg.dataset.kv_fill),
        coef=OmegaConf.to_container(coef) if coef is not None else None,
        order_seed=int(cfg.dataset.order_seed),
        max_model_len=int(cfg.engine.max_model_len),
    )


def _chunks(
    rows: list[dict], size: int, split_after_ref: bool = False
) -> list[list[dict]]:
    """Consecutive rows with one engine configuration, at most ``size`` each;
    with ``split_after_ref`` a REF ends its chunk, so its drift is seen at once."""
    chunks: list[list[dict]] = []
    for row in rows:
        last = chunks[-1] if chunks else None
        same = last and last[0].get("engine_overrides") == row.get("engine_overrides")
        if split_after_ref and last and last[-1]["series"] == "REF":
            same = False
        if same and len(last) < size:
            last.append(row)
        else:
            chunks.append([row])
    return chunks


def _run_chunk(chunk: list[dict], cfg: DictConfig, model_path: str) -> list[dict]:
    """Run a chunk in fresh processes until every attempt has a result."""
    engine = {**_engine_config(cfg), **(chunk[0].get("engine_overrides") or {})}
    results: list[dict] = []
    while len(results) < len(chunk):
        rest = chunk[len(results) :]
        done = run_isolated(
            {
                "run_specs": rest,
                "model_path": model_path,
                "engine_config": engine,
                "output_root": cfg.dataset.output_root,
                "experiment_profile": dict(cfg.hardware_profile),
                "telemetry_interval_s": float(cfg.dataset.telemetry_interval_s),
                "locked_sm_clock_mhz": cfg.dataset.locked_sm_clock_mhz,
                "logging_config": {
                    "logging": dict(cfg.get("logging", {})),
                    "clearml": dict(cfg.get("clearml", {})),
                },
            },
            timeout_s=float(cfg.dataset.attempt_timeout_s) * len(rest) + 120,
        )
        if not done:  # the worker died before its first attempt finished
            done = [{"run_id": rest[0]["run_id"], "status": "worker_crash"}]
        results.extend(done)
    return results


def _compact(root: Path) -> int:
    """Per-step KV usage into ``kv_blocks.jsonl``, then gzip ``cache_events``.

    ``cache_events`` holds per-block states (about 60% of an S4/S5 attempt) and
    the time-model loader does not read it; the used-block count after each
    allocation is all the report needs.
    """
    count = 0
    for events in sorted(root.glob("*/*/cache_events.jsonl")):
        used: dict[str, dict] = {}
        with events.open(encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                after = event.get("after")
                if not after or event.get("step_id") is None:
                    continue
                row = used.setdefault(
                    event["step_id"],
                    {
                        "step_id": event["step_id"],
                        "used_block_count": 0,
                        "capacity_blocks": after["capacity_blocks"],
                    },
                )
                row["used_block_count"] = max(
                    row["used_block_count"], after["used_block_count"]
                )
        with (events.parent / "kv_blocks.jsonl").open("w", encoding="utf-8") as f:
            for row in used.values():
                f.write(json.dumps(row) + "\n")
        with events.open("rb") as src, gzip.open(f"{events}.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        events.unlink()
        count += 1
    return count


def _machine_problem() -> str | None:
    """Another process on the GPU, battery power or a non-performance profile."""
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if apps:
        return f"other processes on the GPU: {apps}"
    adapter = Path("/sys/class/power_supply/ADP0/online")
    if adapter.exists() and adapter.read_text().strip() != "1":
        return "the laptop runs on battery"
    profile = Path("/sys/firmware/acpi/platform_profile")
    if profile.exists() and profile.read_text().strip() != "performance":
        return f"platform profile is {profile.read_text().strip()}"
    return None


def _engine_mismatch(cfg: DictConfig) -> dict:
    """Engine settings that differ from the reference plan (phase D: phase C)."""
    reference = cfg.dataset.engine_reference
    if not reference:
        return {}
    want = json.loads(Path(reference).read_text())["configuration"]["engine"]
    have = _engine_config(cfg)
    return {
        k: {"reference": want.get(k), "now": have.get(k)}
        for k in sorted(set(want) | set(have))
        if want.get(k) != have.get(k)
    }


def _done_runs(roots) -> set[str]:
    """Run ids with a successful attempt under any of ``roots`` (resume)."""
    done = set()
    for root in roots or []:
        for path in Path(root).glob("*/*/runs.json"):
            row = json.loads(path.read_text())
            row = row[0] if isinstance(row, list) else row
            if row.get("status") == "success":
                done.add(row["run_id"])
    return done


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _exact_n_share(reqs: list[dict], loop: dict) -> float:
    """Share of the measurement window with exactly N requests in the system."""
    t = np.array(
        [q["enqueue_monotonic_ns"] for q in reqs]
        + [q["finish_monotonic_ns"] for q in reqs],
        float,
    )
    d = np.concatenate([np.ones(len(reqs)), -np.ones(len(reqs))])
    order = np.lexsort((d, t))  # at equal times a finish before a submit
    t, count = t[order], np.cumsum(d[order])
    lo, hi = loop["window_start_ns"], loop["window_end_ns"]
    dur = np.diff(np.clip(np.concatenate([t, [hi]]), lo, hi))
    return float(dur[count == loop["N"]].sum() / (hi - lo))


def _block_check(results: list[dict], cfg: DictConfig, hours: float) -> dict:
    """Short check after a block: attempts, failures and reruns, O as planned,
    preemptions in single-batch series, closed loop share, REF drift."""
    final = {r["run_id"]: r for r in results}
    ok = [r for r in final.values() if r.get("status") == "success"]
    short = preempted = 0
    exact: list[float] = []
    refs = []
    for r in ok:
        path = Path(r["path"])
        if r["run_id"].split("-")[1] == "REF":
            refs.append(r["metrics"]["wall_seconds"])
            continue
        reqs = _jsonl(path / "requests.jsonl")
        short += sum(
            q["output_token_count"] != q["requested_output_tokens"] for q in reqs
        )
        loop = json.loads((path / "runs.json").read_text()).get("closed_loop")
        if loop is None:
            preempted += sum(
                s.get("preempted_count", 0) for s in _jsonl(path / "steps.jsonl")
            )
            continue
        exact.append(_exact_n_share(reqs, loop))
    base = float(cfg.dataset.ref_seconds)
    return {
        "hours": hours,
        "attempts": len(results),
        "runs": len(final),
        "success": len(ok),
        "failed": sorted(r["run_id"] for r in final.values() if r not in ok),
        "throttled_attempts": sum(r.get("status") == "throttled" for r in results),
        "requests_with_wrong_O": short,
        "preemptions_single_batch": preempted,
        "closed_loop_exact_n_min": min(exact) if exact else None,
        "ref_seconds": refs,
        "ref_drift_max": max(abs(x / base - 1) for x in refs) if refs else None,
    }


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> dict:
    return run_experiment(cfg, run)


def run(cfg: DictConfig) -> dict:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    model_path = str(Path(cfg.dataset.model_path).expanduser().resolve())
    if cfg.mode == "download":
        return {"mode": "download", **_download_model(cfg, output_dir)}
    plan = _plan(cfg)
    plan["configuration"] = {
        "engine": _engine_config(cfg),
        "model_path": model_path,
        "experiment_profile": dict(cfg.hardware_profile),
    }
    _save_json(output_dir / "plan.json", plan)
    summary = {
        "mode": str(cfg.mode),
        "phase": plan["phase"],
        "run_count": plan["run_count"],
        "series_counts": plan["series_counts"],
        "dropped": len(plan["dropped"]),
        "plan_path": str(output_dir / "plan.json"),
    }
    if cfg.mode == "plan":
        return summary
    if cfg.mode != "run":
        raise ValueError(f"unknown mode: {cfg.mode}")
    mismatch = _engine_mismatch(cfg)
    if mismatch:
        raise RuntimeError(
            f"engine differs from {cfg.dataset.engine_reference}: {mismatch}"
        )
    _prepare_model(cfg, output_dir)
    os.sched_setaffinity(0, list(cfg.dataset.cpu_affinity))  # workers inherit it
    limit = int(cfg.dataset.run_limit)
    rows = plan["matrix"][:limit] if limit > 0 else plan["matrix"]
    # Phase D runs one block at a time; budget cuts drop seeds or workloads.
    if cfg.dataset.block:
        rows = [r for r in rows if r.get("block") == cfg.dataset.block]
    if cfg.dataset.seeds:
        seeds = set(cfg.dataset.seeds)
        rows = [r for r in rows if r["series"] == "REF" or r["seed"] in seeds]
    rows = [
        r
        for r in rows
        if r["workload_id"] not in set(cfg.dataset.exclude)
        and r["run_order"] > int(cfg.dataset.resume_after)
    ]
    done = _done_runs(cfg.dataset.done_roots)
    rows = [r for r in rows if r["run_id"] not in done]
    drift_max = cfg.dataset.ref_drift_max
    started = time.monotonic()
    results: list[dict] = []
    stopped = None
    by_id = {row["run_id"]: row for row in rows}
    pending = rows
    with project_logging.experiment_progress(len(rows), "Local attempts") as bar:
        # Throttled attempts are rerun after the rest, at most ``throttle_reruns``
        # times; each try is a new attempt directory of the same run.
        for round_ in range(int(cfg.dataset.throttle_reruns) + 1):
            throttled = []
            chunks = _chunks(
                pending, int(cfg.dataset.attempts_per_process), drift_max is not None
            )
            for chunk in chunks:
                if cfg.dataset.machine_checks:
                    stopped = _machine_problem()
                    if stopped:
                        break
                bar.stage(
                    f"round {round_}: {chunk[0]['run_id']} .. {chunk[-1]['run_id']}"
                )
                for result in _run_chunk(chunk, cfg, model_path):
                    results.append(result)
                    if result.get("status") == "throttled":
                        throttled.append(by_id[result["run_id"]])
                    elif round_ == 0:
                        bar.advance(str(result.get("status", "unknown")))
                    else:
                        bar.advance(f"rerun {result.get('status')}")
                    blocks = (
                        (result.get("effective_config") or {})
                        .get("effective_config", {})
                        .get("num_kvcache_blocks")
                    )
                    if blocks is not None and blocks != cfg.dataset.kv_blocks:
                        stopped = (
                            f"{blocks} KV blocks instead of {cfg.dataset.kv_blocks}"
                        )
                    ref = result["run_id"].split("-")[1] == "REF"
                    if drift_max is not None and ref and result.get("metrics"):
                        drift = (
                            result["metrics"]["wall_seconds"] / cfg.dataset.ref_seconds
                            - 1
                        )
                        if abs(drift) > drift_max:
                            stopped = (
                                f"REF drift {drift:+.1%} at {result['run_id']} "
                                f"(run_order {by_id[result['run_id']]['run_order']})"
                            )
                if cfg.dataset.compact_cache_events:
                    _compact(Path(cfg.dataset.output_root))
                if stopped:
                    break
            if stopped or not throttled:
                break
            pending = throttled
    if stopped:
        logger.error("Collection stopped: {}", stopped)
    final = {r["run_id"]: r for r in results}
    failed = [r for r in final.values() if r.get("status") != "success"]
    summary.update(
        {
            "attempt_count": len(results),
            "throttled_attempts": sum(r.get("status") == "throttled" for r in results),
            "failed": [{"run_id": r["run_id"], "status": r["status"]} for r in failed],
            "kv_blocks": sorted(
                {
                    (r.get("effective_config") or {})
                    .get("effective_config", {})
                    .get("num_kvcache_blocks")
                    for r in results
                    if r.get("effective_config")
                }
                - {None}
            ),
        }
    )
    if cfg.dataset.block:
        summary["stopped"] = stopped
        check = _block_check(results, cfg, (time.monotonic() - started) / 3600)
        summary["block_check"] = {k: v for k, v in check.items() if k != "ref_seconds"}
        _save_json(output_dir / "block-check.json", {"stopped": stopped, **check})
    _save_json(output_dir / "dataset-summary.json", {**summary, "runs": results})
    return summary


if __name__ == "__main__":
    if maybe_launch_detached(__file__):
        raise SystemExit(0)
    main()
