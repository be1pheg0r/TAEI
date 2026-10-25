"""Optional minimal/detailed timing diagnostic, without validation or acceptance gates."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .engine import PINNED_REVISION, NanoVLLMBackend
from .metadata import collect_environment, collect_hardware, collect_model
from .workload import build_workload

PROFILES = ("minimal", "detailed")
TIMING_POLICY = {
    "clock": "host time.monotonic_ns",
    "window": "synchronize; monotonic start; one engine.generate; synchronize; end",
    "includes": ["scheduler", "model execution", "sampling", "tokenizer decode"],
    "excludes": [
        "model load",
        "backend warmup and cache reset",
        "prefix donor priming",
        "workload construction",
        "artifact serialization and disk flush",
    ],
    "wall_unit": "seconds",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _execute_profile(
    profile: str,
    requests: list[dict[str, Any]],
    model_path: str,
    engine_config: dict[str, Any],
    seed: int,
    emit: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "profile": profile,
        "status": "failed",
        "request_order": [str(r["request_id"]) for r in requests],
        "outputs": [],
        "traces": {},
        "prepare": None,
        "wall_seconds": None,
        "wall_start_monotonic_ns": None,
        "wall_end_monotonic_ns": None,
        "synchronize_count": 0,
        "error": None,
    }
    backend = None
    try:
        backend = NanoVLLMBackend(
            model_path=model_path,
            engine_config=dict(engine_config),
            prefix_cache_mode="on",
            scheduler_mode="upstream_dynamic",
            seed=seed,
            emit=emit,
            measurement_profile=profile,
        )
        record["prepare"] = backend.prepare()
        record["prefix_priming"] = backend.prime_requests(requests)
        backend.synchronize()
        record["synchronize_count"] += 1
        start_ns = time.monotonic_ns()
        record["wall_start_monotonic_ns"] = start_ns
        record["outputs"] = backend.generate(requests)
        backend.synchronize()
        record["synchronize_count"] += 1
        end_ns = time.monotonic_ns()
        record["wall_end_monotonic_ns"] = end_ns
        record["wall_seconds"] = (end_ns - start_ns) / 1e9
        record["status"] = "success"
    except Exception as exc:
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "stage": "prepare_generate_or_synchronize",
        }
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                record["error"] = record["error"] or {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "stage": "close",
                }
                record["status"] = "failed"
    return record


def measure_overhead(
    model_path: str,
    engine_config: dict[str, Any],
    output_root: str | Path,
    *,
    workload_name: str = "prefix_sweep",
    seed: int = 42,
    pair_count: int = 1,
    progress_callback: Callable[[str, str | None], None] | None = None,
) -> dict[str, Any]:
    """Persist observed timings and raw traces; do not compare or certify them."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    workload = build_workload(workload_name, seed)
    requests = workload["requests"]
    diagnostic_id = str(uuid.uuid4())
    attempt_dir = root / diagnostic_id
    attempt_dir.mkdir(exist_ok=False)
    hardware = collect_hardware()
    environment = collect_environment()
    model = collect_model(model_path)
    profile_records = []
    pairs = []
    for pair_index in range(1, pair_count + 1):
        profiles = {}
        for profile in PROFILES:
            stage = f"Timing pair {pair_index}/{pair_count}: {profile}"
            if progress_callback is not None:
                progress_callback(stage, None)
            events: dict[str, list[dict[str, Any]]] = {}

            def emit(
                table: str,
                row: dict[str, Any],
                *,
                _events: dict[str, list[dict[str, Any]]] = events,
            ) -> None:
                _events.setdefault(table, []).append(dict(row))

            record = _execute_profile(
                profile, requests, model_path, engine_config, seed, emit
            )
            record["pair_index"] = pair_index
            record["traces"] = events
            prefix = f"pair-{pair_index:02d}-{profile}"
            record["outputs_file"] = f"{prefix}-outputs.json"
            record["traces_file"] = f"{prefix}-traces.json"
            _write_json(attempt_dir / record["outputs_file"], record["outputs"])
            _write_json(attempt_dir / record["traces_file"], events)
            profiles[profile] = record
            profile_records.append(record)
            if progress_callback is not None:
                progress_callback(stage, record["status"])
        minimal = profiles["minimal"]["wall_seconds"]
        detailed = profiles["detailed"]["wall_seconds"]
        pairs.append(
            {
                "pair_index": pair_index,
                "profiles": profiles,
                "validation_status": "disabled",
                "overhead": {
                    "wall_delta_seconds": detailed - minimal
                    if minimal is not None and detailed is not None
                    else None,
                    "wall_ratio_detailed_over_minimal": detailed / minimal
                    if minimal and detailed is not None
                    else None,
                    "n_pairs": 1,
                },
            }
        )
    for name, value in (
        ("hardware.json", hardware),
        ("environment.json", environment),
        ("model.json", model),
        ("workload.json", workload),
        ("engine_config.json", engine_config),
    ):
        _write_json(attempt_dir / name, value)
    artifacts = {}
    for path in sorted(attempt_dir.iterdir()):
        raw = path.read_bytes()
        artifacts[path.name] = {"sha256": _sha_bytes(raw), "bytes": len(raw)}
    report = {
        "schema_version": "1.0.0",
        "diagnostic_id": diagnostic_id,
        "diagnostic_only": True,
        "primary_analysis": False,
        "status": "recorded",
        "validation_status": "disabled",
        "calibration_applicable": None,
        "pair_count": pair_count,
        "profile_order": list(PROFILES),
        "workload_id": workload["workload_id"],
        "workload_name": workload["workload_name"],
        "seed": seed,
        "request_count": len(requests),
        "prefix_cache_mode": "on",
        "scheduler_mode": "upstream_dynamic",
        "engine_config": dict(engine_config),
        "configured_engine_revision": PINNED_REVISION,
        "timing_policy": TIMING_POLICY,
        "pairs": pairs,
        "files": artifacts,
        "attempts": profile_records,
        "path": str(attempt_dir.resolve() / "overhead.json"),
    }
    _write_json(attempt_dir / "overhead.json", report)
    report_raw = (attempt_dir / "overhead.json").read_bytes()
    _write_json(
        attempt_dir / "overhead-manifest.json",
        {
            "diagnostic_id": diagnostic_id,
            "report_file": "overhead.json",
            "sha256": _sha_bytes(report_raw),
            "bytes": len(report_raw),
            "supporting_files": artifacts,
        },
    )
    return report
