"""Plan and persist paired cache experiments without automatic acceptance checks."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from .metadata import collect_environment, collect_hardware, collect_model
from .request_scaling import PROTOCOL as SCALING_PROTOCOL
from .request_scaling import build_scaling_plan
from .schema import DATASET_SCHEMA, FIELD_DICTIONARY, SCHEMA_VERSION
from .workload import (
    LONG_CONTEXT_SETS,
    WORKLOAD_SETS,
    build_long_context_plan,
    build_run_matrix,
    build_workload,
)

COLLECTOR_VERSION = "0.3.0"
TRACE_TABLES = (
    "steps",
    "step_requests",
    "cache_events",
    "scheduler_events",
    "token_events",
    "telemetry",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


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
                    row, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                )
                + "\n"
            )


def build_plan(
    order_seed: int = 20261023,
    *,
    workload_protocol: str = "six_set_paired_prefix_sweep_v1",
    context_cap: int = 32768,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 16384,
) -> dict[str, Any]:
    if workload_protocol == SCALING_PROTOCOL:
        plan = build_scaling_plan(
            order_seed, context_cap, max_num_seqs, max_num_batched_tokens
        )
        plan["schema_version"] = SCHEMA_VERSION
        return plan
    if workload_protocol == "qwen3_8b_long_context_32k_v1":
        plan = build_long_context_plan(
            context_cap=context_cap,
            order_seed=order_seed,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        prompt_total = output_total = 0
        workload_summaries: dict[str, dict[str, Any]] = {}
        for run in plan["matrix"]:
            workload = build_workload(run["workload_id"], run["seed"])
            workload_summaries[workload["workload_id"]] = workload["summary"]
            prompt_total += workload["summary"]["logical_prompt_tokens"]
            output_total += workload["summary"]["requested_output_token_total"]
        plan.update(
            {
                "schema_version": SCHEMA_VERSION,
                "condition_count": len({row["condition_id"] for row in plan["matrix"]}),
                "workload_sets": list(LONG_CONTEXT_SETS),
                "workload_summaries": workload_summaries,
                "seed_count": 2,
                "seeds": [42, 43],
                "treatments_per_set_seed": 4,
                "logical_prompt_tokens": prompt_total,
                "requested_output_tokens": output_total,
                "validation_status": "disabled",
            }
        )
        return plan
    if workload_protocol != "six_set_paired_prefix_sweep_v1":
        raise ValueError(f"unsupported workload protocol: {workload_protocol}")
    matrix = build_run_matrix(order_seed)
    prompt_total = output_total = request_total = 0
    workload_summaries: dict[str, dict[str, Any]] = {}
    for run in matrix:
        workload = build_workload(run["workload_id"], run["seed"])
        summary = workload["summary"]
        workload_summaries[workload["workload_id"]] = summary
        prompt_total += summary["logical_prompt_tokens"]
        output_total += summary["requested_output_token_total"]
        request_total += summary["request_count"]
    return {
        "schema_version": SCHEMA_VERSION,
        "requirements_source": {
            "document_id": "b0d87b1f-2006-4d92-8fbf-2efc72375b3c",
            "revision": 16,
            "source_updated_at": "2026-10-23T13:24:06.215Z",
            "mapping_version": "six_workload_sets_v1",
            "source_snapshot_sha256": None,
            "source_snapshot_note": (
                "historical source reference; active protocol is six independent "
                "workload sets"
            ),
        },
        "matrix": matrix,
        "run_count": len(matrix),
        "condition_count": len({r["condition_id"] for r in matrix}),
        "request_count": request_total,
        "workload_protocol": "six_set_paired_prefix_sweep_v1",
        "workload_sets": list(WORKLOAD_SETS),
        "workload_summaries": workload_summaries,
        "seed_count": 2,
        "seeds": [42, 43],
        "treatments_per_set_seed": 4,
        "requests_per_run": 16,
        "prompt_lengths_vary_by_set_and_seed": True,
        "validation_status": "disabled",
        "logical_prompt_tokens": prompt_total,
        "requested_output_tokens": output_total,
        "adapter_request_counts_per_engine_launch": {
            "measured": 16,
            "untimed_donors": 8,
            "warmup": 1,
        },
        "overhead_policy": {
            "diagnostic": "paired minimal-vs-detailed trace calibration is separate",
            "primary_analysis_gate": "disabled",
        },
        "scheduler_policy": {
            "max_num_seqs": 8,
            "dynamic_max_num_seqs_semantics": (
                "per-schedule selected batch cap; upstream running set may exceed 8"
            ),
            "fixed_batch": "admit waves <=8 without replacement until wave completes",
        },
        "timing_policy": {
            "window": (
                "synchronize; host monotonic around one engine.generate call; "
                "synchronize"
            ),
            "includes": [
                "scheduler",
                "model execution",
                "sampling",
                "tokenizer decode",
            ],
            "excludes": [
                "model loading",
                "warmup",
                "prefix donor priming",
                "workload tokenization",
                "artifact disk flush",
            ],
            "gpu_per_token_timestamps": (
                "unsupported unless measured by engine instrumentation"
            ),
        },
        "split_policy": (
            "pair_id groups four treatments per workload/seed; split_group_id keeps "
            "all treatments and both seeds of a workload set together"
        ),
    }


def preflight(model_path: str) -> dict[str, Any]:
    hardware, environment, model = (
        collect_hardware(),
        collect_environment(),
        collect_model(model_path),
    )
    blocking = []
    if model.get("model_path") is None:
        blocking.append("model_path_missing")
    if not model.get("config_sha256"):
        blocking.append("model_config_missing")
    if not isinstance(model.get("weight_bytes"), int) or model["weight_bytes"] <= 0:
        blocking.append("model_weights_missing")
    if not model.get("tokenizer_sha256"):
        blocking.append("tokenizer_assets_missing")
    installed = _installed_distributions()
    if "nano-vllm" not in installed:
        blocking.append("inference_extra_missing")
    if "torch" not in installed:
        blocking.append("torch_missing")
    cuda_available = _cuda_available()
    if not cuda_available:
        blocking.append("torch_cuda_unavailable")
    pinned_revision = None
    try:
        from .engine import PINNED_REVISION, _upstream_root, verify_pinned_source

        pinned_revision = PINNED_REVISION
        pinned_hashes = verify_pinned_source(_upstream_root())
        pin_error = None
    except (ImportError, OSError, RuntimeError) as exc:
        pinned_hashes = None
        pin_error = f"{type(exc).__name__}: {exc}"
        blocking.append("pinned_engine_unverified")
    return {
        "ready": not blocking,
        "blocking_reasons": blocking,
        "gpu_inference_attempted": False,
        "hardware": hardware,
        "environment": environment,
        "model": model,
        "inference_extra_installed": "nano-vllm" in installed,
        "torch_cuda_available": cuda_available,
        "pinned_engine_revision": pinned_revision if pinned_hashes else None,
        "pinned_engine_source_hashes": pinned_hashes,
        "pinned_engine_error": pin_error,
        "note": (
            "Preflight never initializes CUDA or loads the model; "
            "no inference launched."
        ),
    }


def _installed_distributions() -> set[str]:
    from importlib.metadata import distributions

    return {
        str(d.metadata.get("Name", "")).lower().replace("_", "-")
        for d in distributions()
    }


def _cuda_available() -> bool:
    """Probe CUDA visibility without loading the model or constructing a context."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError, OSError):
        return False


def evaluate_calibration_scope(
    run_spec: dict[str, Any],
    workload_id: str,
    identity: dict[str, str],
    engine_config_sha256: str,
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Match actual paired overhead evidence to one narrowly applicable run scope."""
    evidence = {
        "status": "pending_calibration",
        "diagnostic_id": None,
        "scope_matches": False,
        "review_required": True,
        "reason": "no_overhead_report_supplied",
        "overhead": None,
    }
    if not report:
        return evidence
    matching = (report.get("identity") or {}).get("matching_primary_identity") or {}
    expected_modes = (run_spec.get("prefix_cache_mode"), run_spec.get("scheduler_mode"))
    pairs = report.get("pairs") or []
    valid_pairs = bool(pairs) and all(
        pair.get("trace_identity_match") is True
        and pair.get("profile_identity_match") is True
        and not pair.get("incomplete_reasons")
        for pair in pairs
    )
    same_identity = all(
        matching.get(key) == identity.get(key)
        for key in ("hardware_id", "environment_id", "model_id")
    )
    same_config = matching.get("engine_config_sha256") == engine_config_sha256
    same_scope = (
        report.get("status") == "complete"
        and report.get("calibration_applicable") is True
        and report.get("workload_id") == workload_id
        and report.get("seed") == run_spec.get("seed")
        and report.get("request_count") == 32
        and report.get("prefix_cache_mode") == expected_modes[0] == "on"
        and report.get("scheduler_mode") == expected_modes[1] == "upstream_dynamic"
        and report.get("profile_order") == ["minimal", "detailed"]
        and valid_pairs
        and same_identity
        and same_config
    )
    if not same_scope:
        return {
            **evidence,
            "status": "out_of_scope",
            "diagnostic_id": report.get("diagnostic_id"),
            "reason": (
                "incomplete_or_nonmatching_diagnostic_identity_scope_or_token_trace"
            ),
        }
    return {
        "status": "review_required",
        "diagnostic_id": report.get("diagnostic_id"),
        "scope_matches": True,
        "review_required": True,
        "reason": (
            "condition-specific magnitude and variability require explicit review; "
            "no threshold is specified"
        ),
        "overhead": [pair.get("overhead") for pair in pairs],
    }


def _metadata_id(value: dict[str, Any], prefix: str) -> str:
    return prefix + "-" + _sha(value)[:16]


def run_attempt(
    run_spec: dict[str, Any],
    model_path: str,
    engine_config: dict[str, Any],
    output_root: str | Path,
    *,
    experiment_profile: dict[str, Any] | None = None,
    telemetry_interval_s: float = 1.0,
    calibration_report: dict[str, Any] | None = None,
    calibration_report_path: str | None = None,
) -> dict[str, Any]:
    """Execute one measured run into a unique immutable attempt directory."""
    attempt_id = str(uuid.uuid4())
    run_dir = Path(output_root) / run_spec["run_id"] / attempt_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events: dict[str, list[dict[str, Any]]] = {table: [] for table in TRACE_TABLES}
    phase_by_step: dict[str, str] = {}

    def emit(table: str, row: dict[str, Any]) -> None:
        normalized = dict(row)
        for key in ("step_id", "block_id", "wave_id", "request_id"):
            if normalized.get(key) is not None:
                normalized[key] = str(normalized[key])
        if isinstance(normalized.get("request_ids"), list):
            normalized["request_ids"] = [
                str(value) for value in normalized["request_ids"]
            ]
        if table == "steps":
            original_step_id = row.get("step_id")
            if original_step_id is not None:
                normalized["step_id"] = str(original_step_id)
                if isinstance(original_step_id, int) and not isinstance(
                    original_step_id, bool
                ):
                    normalized.setdefault("step_index", original_step_id)
            normalized["run_id"] = run_spec["run_id"]
            normalized.setdefault("batch_size", row.get("actual_batch_size"))
            normalized.setdefault("queue_before", row.get("queue_count_before"))
            normalized.setdefault("running_before", row.get("running_count_before"))
            if original_step_id is not None:
                phase_by_step[str(original_step_id)] = str(row.get("phase", "unknown"))
        if table == "step_requests":
            normalized["run_id"] = run_spec["run_id"]
            phase = phase_by_step.get(str(row.get("step_id")), "unknown")
            normalized.setdefault(
                "relation",
                "prefill" if phase in {"prefill", "chunked_prefill"} else "decode",
            )
        if table == "scheduler_events":
            event_name = row.get("event_type", row.get("event", "unknown"))
            normalized["event_type"] = (
                "admit" if event_name == "enqueue" else event_name
            )
            normalized["run_id"] = run_spec["run_id"]
            normalized.setdefault(
                "request_ids",
                row.get(
                    "admitted_request_ids",
                    [row["request_id"]] if row.get("request_id") else [],
                ),
            )
            normalized.setdefault("request_count", len(normalized["request_ids"]))
        if table == "cache_events":
            normalized["run_id"] = run_spec["run_id"]
        if table == "token_events" and normalized.get("token_index") is None:
            normalized.setdefault("field_status", {})["token_index"] = {
                "status": "unsupported",
                "reason": (
                    "pinned engine exposes sampled tokens without a stable "
                    "request-local index at this hook"
                ),
            }
        status = normalized.setdefault("field_status", {})
        for name, value in normalized.items():
            if value is None and name not in status:
                status[name] = {
                    "status": "not_collected",
                    "reason": "source hook did not supply this nullable field",
                }
        events[table].append(
            {
                **normalized,
                "dataset_clock_domain": "host_process_monotonic_ns",
                "dataset_process_id": __import__("os").getpid(),
                "recorded_monotonic_ns": time.monotonic_ns(),
            }
        )

    hardware, environment, model = (
        collect_hardware(),
        collect_environment(),
        collect_model(model_path),
    )
    workload = build_workload(run_spec["workload_id"], int(run_spec["seed"]))
    stable_hardware = {
        "cpu_model": hardware.get("cpu_model"),
        "architecture": hardware.get("architecture"),
        "gpu_devices": [
            {"name": g.get("name"), "memory_total_bytes": g.get("memory_total_bytes")}
            for g in hardware.get("gpu_devices") or []
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
    ids = {
        "hardware_id": _metadata_id(stable_hardware, "H"),
        "environment_id": _metadata_id(stable_environment, "E"),
        "model_id": _metadata_id(stable_model, "M"),
        "workload_id": workload["workload_id"],
    }
    for payload, identifier_key in (
        (hardware, "hardware_id"),
        (environment, "environment_id"),
        (model, "model_id"),
    ):
        payload[identifier_key] = ids[identifier_key]
        payload.setdefault("field_status", {})[identifier_key] = {
            "status": "supported",
            "reason": (
                "stable identity hash over named non-secret hardware/software/model "
                "fields"
            ),
        }
    config_id = _sha(
        {
            "engine": engine_config,
            "model_path": str(Path(model_path).resolve()),
            "hardware_id": ids["hardware_id"],
            "environment_id": ids["environment_id"],
            "model_id": ids["model_id"],
            "condition_id": run_spec["condition_id"],
            "prefix_cache_mode": run_spec["prefix_cache_mode"],
            "scheduler_mode": run_spec["scheduler_mode"],
            "seed": run_spec["seed"],
            "workload_id": workload["workload_id"],
            "timing_policy": build_plan()["timing_policy"],
        }
    )
    calibration_gate = {"status": "disabled", "review_required": False}
    run_row = {
        **run_spec,
        **ids,
        "config_id": config_id,
        "attempt_id": attempt_id,
        "status": "running",
        "primary_status": "pending",
        "error": None,
        "metrics": {
            "wall_seconds": None,
            "output_tokens": None,
            "output_tokens_per_second": None,
            "requests_per_second": None,
            "request_count": len(workload["requests"]),
            "prefix_reused_tokens": None,
            "cache_hit_tokens": None,
            "cache_hit_ratio": None,
            "requested_output_tokens": workload["summary"][
                "requested_output_token_total"
            ],
        },
        "effective_config": dict(engine_config),
        "experiment_profile": experiment_profile,
        "timing_policy": {
            **build_plan()["timing_policy"],
            "primary_analysis_gate": {
                **calibration_gate,
                "report_path": calibration_report_path,
            },
        },
    }
    records: list[dict[str, Any]] = []
    status, error = "failed", None
    traceback_text: str | None = None
    backend = sampler = None
    generate_completed = False
    try:
        from .engine import NanoVLLMBackend
        from .metadata import TelemetrySampler

        backend = NanoVLLMBackend(
            model_path,
            engine_config,
            run_spec["prefix_cache_mode"],
            run_spec["scheduler_mode"],
            int(run_spec["seed"]),
            emit,
        )
        prepared = backend.prepare()
        run_row["effective_config"] = prepared
        prime_requests = workload["requests"]
        if (
            run_spec["workload_id"].startswith("qwen3_8b_request_scaling_n")
            and run_spec["prefix_cache_mode"] == "off"
        ):
            prime_requests = []
        run_row["prefix_priming"] = backend.prime_requests(prime_requests)
        sampler = TelemetrySampler(telemetry_interval_s, emit)
        emit(
            "telemetry",
            {**sampler.snapshot(), "sample_scope": "before_measurement"},
        )
        sampler.start()
        backend.synchronize()
        start_ns = time.monotonic_ns()
        records = backend.generate(workload["requests"])
        generate_completed = True
        backend.synchronize()
        end_ns = time.monotonic_ns()
        status = "success"
        elapsed = (end_ns - start_ns) / 1e9
        output_count = sum(row.get("output_token_count") or 0 for row in records)
        prompt_count = sum(row["input_token_count"] for row in records)
        hit_count = sum(row["cache_hit_tokens"] for row in records)
        run_row["metrics"] = {
            "wall_seconds": elapsed,
            "output_tokens": output_count,
            "output_tokens_per_second": output_count / elapsed if elapsed > 0 else None,
            "requests_per_second": len(records) / elapsed if elapsed > 0 else None,
            "request_count": len(records),
            "input_tokens": prompt_count,
            "prefix_reused_tokens": hit_count,
            "cache_hit_tokens": hit_count,
            "cache_hit_ratio": hit_count / prompt_count if prompt_count else None,
            "cache_hit_ratio_definition": "sum(cache_hit_tokens) / sum(input_tokens)",
            "requested_output_tokens": workload["summary"][
                "requested_output_token_total"
            ],
        }
        run_row["generate_start_monotonic_ns"], run_row["generate_end_monotonic_ns"] = (
            start_ns,
            end_ns,
        )
    except Exception as exc:
        traceback_text = traceback.format_exc()
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "stage": "prepare_or_generate",
            "traceback": traceback_text,
        }
        run_row["error"] = error
        run_row["metrics"] = {
            "wall_seconds": None,
            "output_tokens": None,
            "output_tokens_per_second": None,
            "requests_per_second": None,
            "input_tokens": None,
            "prefix_reused_tokens": None,
            "cache_hit_tokens": None,
            "cache_hit_ratio": None,
        }
    finally:
        if sampler is not None:
            sampler.stop()
            emit(
                "telemetry",
                {**sampler.snapshot(), "sample_scope": "after_measurement"},
            )
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                error = error or {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "stage": "close",
                }
                run_row["error"] = error
                status = "failed"
        by_id = {row["request_id"]: row for row in records}
        requests = []
        for request in workload["requests"]:
            result = by_id.get(request["request_id"], {})
            requests.append(
                {
                    **request,
                    **result,
                    **ids,
                    "run_id": run_spec["run_id"],
                    "attempt_id": attempt_id,
                    "condition_id": run_spec["condition_id"],
                    "pair_id": run_spec["pair_id"],
                    "set_id": run_spec.get("set_id", run_spec["workload_id"]),
                    "split_group_id": run_spec["split_group_id"],
                    "repeat_id": run_spec["repeat_id"],
                    "run_order": run_spec["run_order"],
                    "config_id": config_id,
                    "status": result.get("status", "failed"),
                    "output_token_ids": result.get("output_token_ids"),
                    "actual_output_tokens": result.get("output_token_count"),
                    "output_token_count": result.get("output_token_count"),
                    "input_token_count": len(request["input_token_ids"]),
                    "prefix_reused_tokens": result.get("prefix_reused_tokens"),
                    "cache_hit_tokens": result.get("cache_hit_tokens"),
                    "cache_hit_ratio": result.get("cache_hit_ratio"),
                    "error_stage": None
                    if result.get("status")
                    else (error or {}).get("stage"),
                    "error_type": None
                    if result.get("status")
                    else (error or {}).get("type"),
                    "error_reason": None
                    if result.get("status")
                    else (error or {}).get("message", "no result"),
                    "step_refs": result.get("step_refs", []),
                    "field_status": {
                        **result.get("field_status", {}),
                        "output_token_ids": {
                            "status": "supported"
                            if result.get("output_token_ids") is not None
                            else "failed",
                            "reason": "actual engine result"
                            if result.get("output_token_ids") is not None
                            else (error or {}).get("message", "no result"),
                        },
                    },
                }
            )
        for request_record in requests:
            field_status = request_record.setdefault("field_status", {})
            for field_name, field_value in request_record.items():
                if field_value is None and field_name not in field_status:
                    field_status[field_name] = {
                        "status": "failed" if status == "failed" else "not_collected",
                        "reason": (error or {}).get(
                            "message", "source hook did not supply this nullable field"
                        ),
                    }
        _write_jsonl(run_dir / "requests.jsonl", requests)
        for table in TRACE_TABLES:
            _write_jsonl(run_dir / f"{table}.jsonl", events[table])
        for name, value in (
            ("hardware.json", hardware),
            ("environment.json", environment),
            ("model.json", model),
            ("workload.json", workload),
        ):
            _write_json(run_dir / name, value)
        if traceback_text is not None:
            (run_dir / "traceback.txt").write_text(traceback_text, encoding="utf-8")
        run_row["status"] = status
        missing_hooks = [
            table
            for table in (
                "steps",
                "step_requests",
                "cache_events",
                "scheduler_events",
                "token_events",
            )
            if not events[table]
        ]
        run_row["primary_status"] = "not_evaluated"
        run_row["validation_status"] = "disabled"
        run_row.update(ids)
        _write_json(run_dir / "runs.json", run_row)
        _write_json(run_dir / "schema.json", DATASET_SCHEMA)
        _write_json(run_dir / "field_dictionary.json", FIELD_DICTIONARY)
        files = {}
        for path in sorted(run_dir.iterdir()):
            if path.name != "manifest.json":
                raw = path.read_bytes()
                files[path.name] = {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                }
        _write_json(
            run_dir / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "collector_version": COLLECTOR_VERSION,
                "run_id": run_spec["run_id"],
                "attempt_id": attempt_id,
                "condition_id": run_spec["condition_id"],
                "pair_id": run_spec["pair_id"],
                "set_id": run_spec.get("set_id", run_spec["workload_id"]),
                "split_group_id": run_spec["split_group_id"],
                "repeat_id": run_spec["repeat_id"],
                "run_order": run_spec["run_order"],
                **ids,
                "config_id": config_id,
                "files": files,
                "hash_policy": "SHA-256 of exact persisted bytes; manifest excluded",
                "coverage": {
                    "primary_complete": None,
                    "primary_analysis_accepted": False,
                    "validation_status": "disabled",
                    "missing_hooks": missing_hooks,
                    "capabilities": {"gpu_inference_attempted": backend is not None},
                    "calibration_gate": {
                        **calibration_gate,
                        "report_path": calibration_report_path,
                    },
                },
                "requirements_source": {
                    "document_id": "b0d87b1f-2006-4d92-8fbf-2efc72375b3c",
                    "revision": 16,
                    "source_updated_at": "2026-10-23T13:24:06.215Z",
                    "mapping_version": "paired_prefix_sweep_v1",
                },
            },
        )

    return {
        "run_id": run_spec["run_id"],
        "attempt_id": attempt_id,
        "path": str(run_dir),
        "status": status,
        "hardware_id": ids["hardware_id"],
        "environment_id": ids["environment_id"],
        "model_id": ids["model_id"],
        "workload_id": ids["workload_id"],
        "config_id": config_id,
        "effective_config": run_row["effective_config"],
        "backend_constructed": backend is not None,
        "real_generate_completed": generate_completed,
        "primary_status": run_row["primary_status"],
        "metrics": run_row["metrics"],
        "qc": {"status": "disabled", "valid": None},
    }
