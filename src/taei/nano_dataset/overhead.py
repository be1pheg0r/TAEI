"""Измеряет накладные расходы подробной трассировки nano-vLLM парными запусками.

Диагностика отдельно от основных измерений выполняет профили minimal и detailed
на одинаковых запросах, конфигурации и условиях, сопоставляет фактические
результаты и идентичность трасс, а затем сохраняет отчет и контрольные суммы
артефактов. Неполные пары или несовпадающие данные не признаются применимой
калибровкой. В рабочем коде используется настоящий закрепленный
``NanoVLLMBackend``; подмена разрешена тестам.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

from .engine import PINNED_REVISION, NanoVLLMBackend
from .metadata import collect_environment, collect_hardware, collect_model
from .workload import build_workload

PROFILES = ("minimal", "detailed")
TIMING_POLICY = {
    "clock": "host time.monotonic_ns",
    "window": (
        "synchronize; monotonic start; one engine.generate; synchronize; monotonic end"
    ),
    "includes": ["scheduler", "model execution", "sampling", "tokenizer decode"],
    "excludes": [
        "model load",
        "backend warmup and cache reset",
        "workload construction",
        "artifact serialization and disk flush",
    ],
    "wall_unit": "seconds",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )


def _metadata_identity(
    hardware: dict[str, Any], environment: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Use the exact stable identity field sets used by primary run_attempt."""
    stable_hardware = {
        "cpu_model": hardware.get("cpu_model"),
        "architecture": hardware.get("architecture"),
        "gpu_devices": [
            {
                "name": gpu.get("name"),
                "memory_total_bytes": gpu.get("memory_total_bytes"),
            }
            for gpu in hardware.get("gpu_devices") or []
        ],
    }
    stable_environment = {
        key: environment.get(key)
        for key in (
            "python_version",
            "torch_version",
            "transformers_version",
            "triton_version",
            "source_commit",
        )
    }
    stable_model = {
        key: model.get(key)
        for key in (
            "config_sha256",
            "tokenizer_sha256",
            "weight_bytes",
            "parameter_count",
            "weight_dtype",
        )
    }
    return {
        "hardware_id": "H-" + _sha_bytes(_canonical(stable_hardware))[:16],
        "environment_id": "E-" + _sha_bytes(_canonical(stable_environment))[:16],
        "model_id": "M-" + _sha_bytes(_canonical(stable_model))[:16],
        "hardware": stable_hardware,
        "environment": stable_environment,
        "model": stable_model,
    }


def _trace_identity(
    requests: list[dict[str, Any]], outputs: list[dict[str, Any]]
) -> tuple[bool, list[str], list[tuple[str, list[int], list[int]]]]:
    reasons: list[str] = []
    expected = [
        (str(request["request_id"]), list(request["input_token_ids"]))
        for request in requests
    ]
    normalized: list[tuple[str, list[int], list[int] | None]] = []
    for output in outputs:
        if not isinstance(output, dict) or not isinstance(
            output.get("output_token_ids"), list
        ):
            reasons.append("missing_actual_output_token_ids")
            continue
        if output.get("status") != "completed":
            reasons.append("request_generation_incomplete")
        if any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in output["output_token_ids"]
        ):
            reasons.append("invalid_actual_output_token_ids")
            continue
        if output.get("output_token_count") != len(output["output_token_ids"]):
            reasons.append("output_token_count_mismatch")
        normalized.append(
            (
                str(output.get("request_id")),
                list(output.get("input_token_ids", [])),
                list(output["output_token_ids"]),
            )
        )
    if len(outputs) != len(requests) or len(normalized) != len(requests):
        reasons.append("request_output_count_mismatch")
    if [(rid, ids) for rid, ids, _ in normalized] != expected:
        reasons.append("input_trace_or_request_order_mismatch")
    if any(ids is None for _, _, ids in normalized):
        reasons.append("missing_actual_output_token_ids")
    signature = [
        (request_id, input_ids, output_ids)
        for request_id, input_ids, output_ids in normalized
        if output_ids is not None
    ]
    return not reasons, sorted(set(reasons)), signature


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
        "request_order": [str(request["request_id"]) for request in requests],
        "outputs": [],
        "traces": [],
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
        if record["prepare"].get("measurement_profile") != profile:
            raise RuntimeError(f"backend did not confirm measurement_profile={profile}")
        backend.synchronize()
        record["synchronize_count"] += 1
        start_ns = time.monotonic_ns()
        record["wall_start_monotonic_ns"] = start_ns
        outputs = backend.generate(requests)
        backend.synchronize()
        record["synchronize_count"] += 1
        end_ns = time.monotonic_ns()
        record["wall_end_monotonic_ns"] = end_ns
        record["wall_seconds"] = (end_ns - start_ns) / 1e9
        record["outputs"] = outputs if isinstance(outputs, list) else []
        record["status"] = "success"
        if not isinstance(outputs, list):
            record["error"] = {
                "type": "TypeError",
                "message": "backend.generate must return a list",
                "stage": "generate",
            }
            record["status"] = "failed"
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
    workload_name: str = "S",
    seed: int = 42,
    pair_count: int = 1,
) -> dict[str, Any]:
    """Run paired real generation profiles and persist an isolated diagnostic.

    ``output_root`` is the explicit diagnostic directory; it must not be the
    primary run directory. Each pair runs minimal then detailed on identical
    request objects/order and the same seed/config/cache/scheduler modes.
    """
    if (
        not isinstance(pair_count, int)
        or isinstance(pair_count, bool)
        or pair_count < 1
    ):
        raise ValueError("pair_count must be a positive integer")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    workload = build_workload(workload_name, seed)
    requests = workload["requests"]
    hardware, environment, model = (
        collect_hardware(),
        collect_environment(),
        collect_model(model_path),
    )
    identity = _metadata_identity(hardware, environment, model)
    run_id = str(uuid.uuid4())
    attempt_dir = root / run_id
    attempt_dir.mkdir(parents=False, exist_ok=False)
    profile_records: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    incomplete_reasons: list[str] = []

    for pair_index in range(1, pair_count + 1):
        profile_results = {}
        event_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for profile in PROFILES:
            events: dict[str, list[dict[str, Any]]] = {}

            def emit(
                table: str,
                row: dict[str, Any],
                *,
                _events: dict[str, list[dict[str, Any]]] = events,
            ) -> None:
                _events.setdefault(table, []).append(dict(row))

            result = _execute_profile(
                profile, requests, model_path, engine_config, seed, emit
            )
            result["pair_index"] = pair_index
            result["traces"] = events
            profile_results[profile] = result
            event_rows[profile] = events
            profile_records.append(result)
        minimal, detailed = profile_results["minimal"], profile_results["detailed"]
        profile_identity_match = (
            minimal.get("prepare") is not None
            and detailed.get("prepare") is not None
            and minimal["prepare"].get("engine_revision") == PINNED_REVISION
            and detailed["prepare"].get("engine_revision") == PINNED_REVISION
            and isinstance(minimal["prepare"].get("engine_source_hashes"), dict)
            and bool(minimal["prepare"].get("engine_source_hashes"))
            and minimal["prepare"].get("engine_revision")
            == detailed["prepare"].get("engine_revision")
            and minimal["prepare"].get("engine_source_hashes")
            == detailed["prepare"].get("engine_source_hashes")
            and minimal["prepare"].get("effective_config")
            == detailed["prepare"].get("effective_config")
        )
        trace_match, trace_reasons, minimal_trace = _trace_identity(
            requests, minimal["outputs"]
        )
        detailed_match, detailed_reasons, detailed_trace = _trace_identity(
            requests, detailed["outputs"]
        )
        output_equal = minimal_trace == detailed_trace
        reasons = []
        for label, result in (("minimal", minimal), ("detailed", detailed)):
            if result["status"] != "success":
                reasons.append(f"{label}_profile_failed")
            if (
                not isinstance(result["wall_seconds"], (int, float))
                or not math.isfinite(result["wall_seconds"])
                or result["wall_seconds"] <= 0
            ):
                reasons.append("nonpositive_wall_time")
        if not profile_identity_match:
            reasons.append("profile_identity_mismatch")
        if not trace_match or not detailed_match or not output_equal:
            reasons.append("token_trace_mismatch")
            reasons.extend(trace_reasons + detailed_reasons)
        minimal_wall, detailed_wall = (
            minimal.get("wall_seconds"),
            detailed.get("wall_seconds"),
        )
        valid_walls = all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in (minimal_wall, detailed_wall)
        )
        pairs.append(
            {
                "pair_index": pair_index,
                "profiles": profile_results,
                "trace_identity_match": bool(
                    trace_match and detailed_match and output_equal
                ),
                "profile_identity_match": profile_identity_match,
                "incomplete_reasons": sorted(set(reasons)),
                "overhead": {
                    "wall_delta_seconds": detailed_wall - minimal_wall
                    if valid_walls
                    else None,
                    "wall_ratio_detailed_over_minimal": detailed_wall / minimal_wall
                    if valid_walls
                    else None,
                    "units": "seconds and dimensionless ratio",
                    "n_pairs": 1,
                },
            }
        )
        incomplete_reasons.extend(reasons)

    # Persist every supporting input, result, and event stream; hashing occurs only
    # after measured windows and serializations have completed.
    _write_json(attempt_dir / "hardware.json", hardware)
    _write_json(attempt_dir / "environment.json", environment)
    _write_json(attempt_dir / "model.json", model)
    _write_json(attempt_dir / "workload.json", workload)
    _write_json(attempt_dir / "engine_config.json", engine_config)
    artifacts: dict[str, dict[str, Any]] = {}
    for result in profile_records:
        prefix = f"pair-{result['pair_index']:02d}-{result['profile']}"
        outputs_path = f"{prefix}-outputs.json"
        traces_path = f"{prefix}-traces.json"
        _write_json(attempt_dir / outputs_path, result["outputs"])
        _write_json(attempt_dir / traces_path, result["traces"])
        result["outputs_file"] = outputs_path
        result["traces_file"] = traces_path
    for path in sorted(attempt_dir.iterdir()):
        if path.is_file() and path.name not in {
            "overhead.json",
            "overhead-manifest.json",
        }:
            raw = path.read_bytes()
            artifacts[path.name] = {"sha256": _sha_bytes(raw), "bytes": len(raw)}
    status = "complete" if not incomplete_reasons else "incomplete"
    report = {
        "schema_version": "1.0.0",
        "diagnostic_id": run_id,
        "diagnostic_only": True,
        "primary_analysis": False,
        "status": status,
        "calibration_applicable": status == "complete",
        "incomplete_reasons": sorted(set(incomplete_reasons)),
        "pair_count": pair_count,
        "n": pair_count,
        "profile_order": list(PROFILES),
        "workload_id": workload["workload_id"],
        "workload_name": workload_name,
        "seed": seed,
        "request_count": len(requests),
        "prefix_cache_mode": "on",
        "scheduler_mode": "upstream_dynamic",
        "engine_config": dict(engine_config),
        "config_id": _sha_bytes(
            _canonical(
                {
                    "engine": engine_config,
                    "model_path": str(Path(model_path).resolve()),
                    "workload_id": workload["workload_id"],
                    "seed": seed,
                    "prefix_cache_mode": "on",
                    "scheduler_mode": "upstream_dynamic",
                }
            )
        ),
        "identity": {
            **identity,
            "engine_revision": next(
                (
                    p["prepare"].get("engine_revision")
                    for p in profile_records
                    if p.get("prepare")
                ),
                None,
            ),
            "engine_source_hashes": next(
                (
                    p["prepare"].get("engine_source_hashes")
                    for p in profile_records
                    if p.get("prepare")
                ),
                None,
            ),
            "model_path": str(Path(model_path).resolve()),
            "measurement_profiles": [
                ((p.get("prepare") or {}).get("measurement_profile"))
                for p in profile_records
            ],
            "model_details": model,
            "matching_primary_identity": {
                "hardware_id": identity["hardware_id"],
                "environment_id": identity["environment_id"],
                "model_id": identity["model_id"],
                "engine_config_sha256": _sha_bytes(_canonical(engine_config)),
                "scope": (
                    "hardware/environment/model IDs use orchestrator stable field "
                    "sets; config hash is diagnostic common config, not a primary "
                    "run config_id"
                ),
            },
        },
        "version_policy": {
            "expected_engine_revision": PINNED_REVISION,
            "requires_exact_revision_and_source_hash_match": True,
            "status": "matched"
            if all(pair["profile_identity_match"] for pair in pairs)
            else "mismatch_or_missing",
            "evidence": (
                "each prepared profile reports the pinned revision, nonempty "
                "source hashes, and the same effective config and source hashes"
            ),
        },
        "timing_policy": TIMING_POLICY,
        "interpretation_policy": {
            "threshold": None,
            "threshold_reason": (
                "No universal overhead threshold is specified; review magnitude "
                "and variability against the intended condition-specific "
                "analysis limits."
            ),
            "trace_identity_required": True,
            "units": {
                "wall_seconds": "seconds",
                "wall_delta_seconds": "seconds",
                "wall_ratio_detailed_over_minimal": "dimensionless ratio",
            },
            "primary_gate": (
                "Do not claim calibration success or accept primary timing solely "
                "from this diagnostic; require complete pairs, identical actual "
                "token traces, matching identities, and explicit "
                "condition-specific review."
            ),
        },
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
            "diagnostic_id": run_id,
            "report_file": "overhead.json",
            "sha256": _sha_bytes(report_raw),
            "bytes": len(report_raw),
            "supporting_files": artifacts,
            "hash_policy": (
                "SHA-256 of exact persisted supporting-file bytes and final "
                "report bytes; manifest excludes itself"
            ),
        },
    )
    return report
