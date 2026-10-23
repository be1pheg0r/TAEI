"""Версионированная JSON Schema и словарь полей датасета запусков nano-vLLM.

Модуль задает перечень артефактов, строгие типы и правила их описания,
метаданные полей и проверку экземпляров датасета по Draft 2020-12. Проверка
возвращает стабильные сообщения с путями к ошибочным полям и поддерживает
явное описание nullable-значений и расширений контракта.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0.0"
ARTIFACTS = (
    "manifest",
    "hardware",
    "environment",
    "model",
    "workload",
    "runs",
    "requests",
    "steps",
    "step_requests",
    "cache_events",
    "scheduler_events",
    "token_events",
    "telemetry",
)

STR = {"type": "string", "minLength": 1}
ID = {"type": "string", "minLength": 1}
INT = {"type": "integer"}
NONNEG_INT = {"type": "integer", "minimum": 0}
NUM = {"type": "number"}
BOOL = {"type": "boolean"}
TOKEN_IDS = {"type": "array", "items": {"type": "integer", "minimum": 0}}
STRING_LIST = {"type": "array", "items": STR}
INTEGER_LIST = {"type": "array", "items": NONNEG_INT}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def _object(
    properties: dict[str, Any], required: tuple[str, ...] = ()
) -> dict[str, Any]:
    # Extra fields are deliberate: raw snapshots and future metadata must round-trip.
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": True,
    }


def _status_map() -> dict[str, Any]:
    status = {
        "type": "object",
        "properties": {
            "status": {
                "enum": [
                    "supported",
                    "unsupported",
                    "not_enabled",
                    "permission_denied",
                    "not_collected",
                    "failed",
                    "not_applicable",
                ]
            },
            "reason": {"type": "string", "minLength": 1},
        },
        "required": ["status", "reason"],
        "additionalProperties": True,
    }
    return {"type": "object", "additionalProperties": status}


def _stamped(
    properties: dict[str, Any], required: tuple[str, ...] = ()
) -> dict[str, Any]:
    props = {
        "event_time_utc": {
            "type": "string",
            "format": "date-time",
            "pattern": "(?:Z|\\+00:00)$",
        },
        "timestamp_utc": {
            "type": "string",
            "format": "date-time",
            "pattern": "(?:Z|\\+00:00)$",
        },
        "monotonic_ns": NONNEG_INT,
        "clock_id": STR,
        "process_id": NONNEG_INT,
        **properties,
    }
    props["field_status"] = _status_map()
    # If nullable data is present, require an explicit status and nonempty reason.
    conditionals = []
    for name, spec in props.items():
        if (
            name == "field_status"
            or "anyOf" not in spec
            or not any(s.get("type") == "null" for s in spec["anyOf"])
        ):
            continue
        conditionals.append(
            {
                "if": {"required": [name], "properties": {name: {"type": "null"}}},
                "then": {
                    "required": ["field_status"],
                    "properties": {
                        "field_status": {
                            "required": [name],
                            "properties": {name: {"required": ["status", "reason"]}},
                        }
                    },
                },
            }
        )
    result = _object(props, required)
    if conditionals:
        result["allOf"] = conditionals
    return result


FILE_RECORD = _object(
    {"sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}, "bytes": NONNEG_INT},
    ("sha256", "bytes"),
)
MANIFEST = _object(
    {
        "schema_version": {"const": SCHEMA_VERSION},
        "collector_version": STR,
        "run_id": ID,
        "attempt_id": ID,
        "condition_id": ID,
        "pair_id": ID,
        "split_group_id": ID,
        "repeat_id": ID,
        "run_order": {"type": "integer", "minimum": 1},
        "hardware_id": ID,
        "environment_id": ID,
        "model_id": ID,
        "workload_id": ID,
        "config_id": ID,
        "files": {"type": "object", "additionalProperties": FILE_RECORD},
        "hash_policy": STR,
        "coverage": _object(
            {
                "primary_complete": BOOL,
                "missing_hooks": STRING_LIST,
                "capabilities": {"type": "object"},
            },
            ("primary_complete",),
        ),
        "requirements_source": _object(
            {
                "document_id": ID,
                "revision": NONNEG_INT,
                "source_updated_at": {"type": "string", "format": "date-time"},
                "mapping_version": ID,
            }
        ),
        "field_status": _status_map(),
    },
    ("schema_version", "collector_version", "run_id", "files", "coverage"),
)

HARDWARE_PROPERTIES: dict[str, Any] = {
    "hardware_id": ID,
    "hostname_pseudonym": _nullable(STR),
    "os": _nullable(STR),
    "os_release": _nullable(STR),
    "kernel_version": _nullable(STR),
    "architecture": _nullable(STR),
    "cpu_model": _nullable(STR),
    "cpu_logical_cores": _nullable(NONNEG_INT),
    "cpu_affinity_count": _nullable(NONNEG_INT),
    "cpu_affinity": _nullable(INTEGER_LIST),
    "cpu_sockets": _nullable(NONNEG_INT),
    "cpu_physical_cores": _nullable(NONNEG_INT),
    "cpu_cache_bytes": _nullable(NONNEG_INT),
    "cpu_governor": _nullable(STR),
    "cpu_quota_cores": _nullable(NUM),
    "cpu_cpuset_quota_bytes": _nullable(NONNEG_INT),
    "cpu_cpuset": _nullable(STRING_LIST),
    "numa_nodes": _nullable(NONNEG_INT),
    "memory_total_bytes": _nullable(NONNEG_INT),
    "memory_available_bytes": _nullable(NONNEG_INT),
    "swap_total_bytes": _nullable(NONNEG_INT),
    "gpu_devices": _nullable(
        {
            "type": "array",
            "items": _object(
                {
                    "index": NONNEG_INT,
                    "name": _nullable(STR),
                    "uuid": _nullable(STR),
                    "pci_bus_id": _nullable(STR),
                    "memory_total_bytes": _nullable(NONNEG_INT),
                    "memory_free_bytes": _nullable(NONNEG_INT),
                    "memory_used_bytes": _nullable(NONNEG_INT),
                    "utilization_gpu_percent": _nullable(NUM),
                    "utilization_memory_percent": _nullable(NUM),
                    "temperature_c": _nullable(NUM),
                    "power_w": _nullable(NUM),
                    "power_limit_w": _nullable(NUM),
                    "sm_clock_mhz": _nullable(NONNEG_INT),
                    "memory_clock_mhz": _nullable(NONNEG_INT),
                    "graphics_clock_mhz": _nullable(NONNEG_INT),
                    "application_sm_clock_mhz": _nullable(NONNEG_INT),
                    "pci_link_gen": _nullable(NONNEG_INT),
                    "pci_link_width": _nullable(NONNEG_INT),
                    "pci_max_link_gen": _nullable(NONNEG_INT),
                    "pci_max_link_width": _nullable(NONNEG_INT),
                    "compute_capability": _nullable(STR),
                    "ecc_mode": _nullable(STR),
                    "mig_mode": _nullable(STR),
                    "field_status": _status_map(),
                },
                ("index",),
            ),
        }
    ),
    "gpu_visible_rank_map": _nullable({"type": "object"}),
    "gpu_compute_capability": _nullable(STR),
    "gpu_pcie_link": _nullable(STR),
    "gpu_nvlink_topology": _nullable(STR),
    "gpu_ecc_mode": _nullable(STR),
    "gpu_power_limits": _nullable({"type": "object"}),
    "gpu_clocks_configured": _nullable({"type": "object"}),
    "gpu_datasheet_flops": _nullable({"type": "object"}),
    "gpu_memory_type": _nullable(STR),
    "gpu_memory_bus_width_bits": _nullable(NONNEG_INT),
    "gpu_memory_bandwidth_bytes_per_second": _nullable(NONNEG_INT),
    "gpu_l2_bytes": _nullable(NONNEG_INT),
    "gpu_mps_mode": _nullable(STR),
    "gpu_exclusive_process_mode": _nullable(BOOL),
    "virtualization": _nullable(STR),
    "cloud_instance_type": _nullable(STR),
    "container_runtime": _nullable(STR),
}
ENVIRONMENT_PROPERTIES = {
    name: _nullable(STR)
    for name in (
        "python_version",
        "torch_version",
        "transformers_version",
        "triton_version",
        "flash_attn_version",
        "nvidia_nccl_version",
        "source_commit",
        "source_diff_sha256",
        "engine_git_pin",
        "engine_version",
        "cuda_driver_version",
        "cuda_toolkit_version",
        "torch_linked_cuda_version",
        "cuda_visible_devices",
        "omp_settings",
        "container_runtime",
    )
}
ENVIRONMENT_PROPERTIES.update(
    {
        "environment_id": ID,
        "source_dirty": _nullable(BOOL),
        "environment_variables": {
            "type": "object",
            "additionalProperties": {"type": ["string", "number", "boolean", "null"]},
        },
        "cli_arguments": _nullable(STRING_LIST),
        "config_values": _nullable({"type": "object"}),
        "backend_compile_flags": _nullable({"type": "object"}),
        "tf32_enabled": _nullable(BOOL),
        "determinism_enabled": _nullable(BOOL),
        "tensor_parallel_size": _nullable(NONNEG_INT),
        "pipeline_parallel_size": _nullable(NONNEG_INT),
        "data_parallel_size": _nullable(NONNEG_INT),
        "kv_budget_bytes_per_rank": _nullable(NONNEG_INT),
        "field_status": _status_map(),
    }
)
MODEL_PROPERTIES: dict[str, Any] = {
    name: _nullable(schema)
    for name, schema in {
        "model_path": STR,
        "model_type": STR,
        "architectures": STRING_LIST,
        "model_revision": STR,
        "download_provenance": {
            "type": "object",
            "properties": {
                "repo_id": STR,
                "requested_revision": STR,
                "resolved_revision": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "config_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "source": STR,
            },
            "required": ["repo_id", "resolved_revision", "config_sha256", "source"],
            "additionalProperties": False,
        },
        "tokenizer_revision": STR,
        "config_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "tokenizer_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "chat_template_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "layer_count": NONNEG_INT,
        "hidden_size": NONNEG_INT,
        "intermediate_size": NONNEG_INT,
        "attention_head_count": NONNEG_INT,
        "key_value_head_count": NONNEG_INT,
        "head_dim": NONNEG_INT,
        "vocab_size": NONNEG_INT,
        "max_position_embeddings": NONNEG_INT,
        "rope_theta": NUM,
        "attention_type": STR,
        "sliding_window": NONNEG_INT,
        "normalization_type": NUM,
        "rms_norm_epsilon": NUM,
        "layer_norm_epsilon": NUM,
        "qk_layer_norm": BOOL,
        "activation": STR,
        "tied_weights": BOOL,
        "weight_dtype": STR,
        "compute_dtype": STR,
        "kv_dtype": STR,
        "quantization_config": {"type": "object"},
        "parameter_count": NONNEG_INT,
        "weight_bytes": NONNEG_INT,
        "tensor_count": NONNEG_INT,
        "dense_moe_config": {"type": "object"},
        "raw_config": {"type": "object"},
        "chat_template_revision": STR,
        "parameter_count_estimate": NONNEG_INT,
        "parameter_estimate_method": STR,
        "compute_dtype_assumptions": STR,
        "kv_dtype_assumptions": STR,
    }.items()
}
MODEL_PROPERTIES["model_id"] = ID
MODEL_PROPERTIES["field_status"] = _status_map()

WORKLOAD = _object(
    {
        "workload_id": ID,
        "workload_name": {"enum": ["S", "L", "M"]},
        "seed": NONNEG_INT,
        "requests": {
            "type": "array",
            "items": _object(
                {
                    "request_id": ID,
                    "arrival_index": NONNEG_INT,
                    "planned_arrival_offset_ns": NONNEG_INT,
                    "workload_class": {"enum": ["short", "long"]},
                    "prefix_family_id": _nullable(STR),
                    "prefix_group_id": _nullable(STR),
                    "prefix_token_count": NONNEG_INT,
                    "input_token_ids": TOKEN_IDS,
                    "requested_output_tokens": NONNEG_INT,
                    "temperature": NUM,
                    "ignore_eos": BOOL,
                },
                (
                    "request_id",
                    "input_token_ids",
                    "requested_output_tokens",
                    "temperature",
                    "ignore_eos",
                ),
            ),
        },
        "summary": _object(
            {
                "request_count": NONNEG_INT,
                "short_count": NONNEG_INT,
                "long_count": NONNEG_INT,
                "input_token_counts": {"type": "array", "items": NONNEG_INT},
                "requested_output_tokens_each": NONNEG_INT,
                "requested_output_token_counts": {"type": "array", "items": NONNEG_INT},
                "logical_prompt_tokens": NONNEG_INT,
                "requested_output_token_total": NONNEG_INT,
                "token_ids_reused_across_modes": BOOL,
            }
        ),
    },
    ("workload_id", "requests"),
)

RUN = _object(
    {
        "run_id": ID,
        "attempt_id": ID,
        "hardware_id": ID,
        "environment_id": ID,
        "model_id": ID,
        "workload_id": ID,
        "config_id": ID,
        "condition_id": ID,
        "pair_id": ID,
        "repeat_id": ID,
        "run_order": {"type": "integer", "minimum": 1},
        "status": {
            "enum": [
                "planned",
                "running",
                "success",
                "failed",
                "incomplete",
                "completed",
                "failed_or_no_output",
            ]
        },
        "primary_status": {"enum": ["accepted", "rejected", "pending"]},
        "error": _nullable(
            _object({"type": STR, "message": {"type": "string"}, "stage": STR})
        ),
        "primary_gate_reason": {"type": "string"},
        "metrics": _object(
            {
                "wall_seconds": _nullable(NUM),
                "output_tokens": _nullable(NONNEG_INT),
                "output_tokens_per_second": _nullable(NUM),
                "requests_per_second": _nullable(NUM),
                "request_count": NONNEG_INT,
                "requested_output_tokens": NONNEG_INT,
            }
        ),
        "effective_config": {"type": "object"},
        "timing_policy": {"type": "object"},
        "generate_start_monotonic_ns": NONNEG_INT,
        "generate_end_monotonic_ns": NONNEG_INT,
        "field_status": _status_map(),
    },
    (
        "run_id",
        "status",
        "primary_status",
        "config_id",
        "condition_id",
        "pair_id",
        "repeat_id",
        "run_order",
        "attempt_id",
    ),
)

REQUEST = _stamped(
    {
        "request_id": ID,
        "run_id": ID,
        "attempt_id": ID,
        "condition_id": ID,
        "pair_id": ID,
        "split_group_id": ID,
        "repeat_id": ID,
        "run_order": {"type": "integer", "minimum": 1},
        "hardware_id": ID,
        "environment_id": ID,
        "model_id": ID,
        "workload_id": ID,
        "config_id": ID,
        "arrival_index": NONNEG_INT,
        "planned_arrival_offset_ns": NONNEG_INT,
        "workload_class": {"enum": ["short", "long"]},
        "prefix_family_id": _nullable(STR),
        "prefix_group_id": _nullable(STR),
        "prefix_token_count": NONNEG_INT,
        "input_token_ids": TOKEN_IDS,
        "output_token_ids": _nullable(TOKEN_IDS),
        "requested_output_tokens": NONNEG_INT,
        "actual_output_tokens": _nullable(NONNEG_INT),
        "output_token_count": _nullable(NONNEG_INT),
        "temperature": NUM,
        "ignore_eos": BOOL,
        "status": {
            "enum": [
                "planned",
                "running",
                "success",
                "failed",
                "incomplete",
                "completed",
                "failed_or_no_output",
            ]
        },
        "error_stage": _nullable(STR),
        "error_type": _nullable(STR),
        "error_reason": _nullable(STR),
        "finish_reason": _nullable(STR),
        **{
            name: _nullable(NONNEG_INT)
            for name in (
                "enqueue_monotonic_ns",
                "first_scheduled_monotonic_ns",
                "first_generated_monotonic_ns",
                "first_host_observed_monotonic_ns",
                "last_generated_monotonic_ns",
                "finish_monotonic_ns",
            )
        },
        "step_refs": STRING_LIST,
    },
    (
        "request_id",
        "input_token_ids",
        "requested_output_tokens",
        "temperature",
        "ignore_eos",
        "status",
    ),
)
STEP = _stamped(
    {
        "step_id": ID,
        "run_id": ID,
        "step_index": NONNEG_INT,
        "phase": {
            "enum": [
                "prefill",
                "chunked_prefill",
                "recompute",
                "decode",
                "mixed",
                "unknown",
            ]
        },
        "request_ids": STRING_LIST,
        "contexts_before": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
        "contexts_after": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
        **{
            name: _nullable(NONNEG_INT)
            for name in (
                "scheduled_tokens",
                "produced_tokens",
                "fresh_prefill_tokens",
                "recompute_tokens",
                "prefix_reused_tokens",
                "batch_size",
                "graph_padded_batch_size",
                "queue_before",
                "queue_after",
                "running_before",
                "running_after",
                "schedule_start_monotonic_ns",
                "schedule_end_monotonic_ns",
                "model_call_start_monotonic_ns",
                "model_call_end_monotonic_ns",
                "postprocess_start_monotonic_ns",
                "postprocess_end_monotonic_ns",
                "gpu_event_duration_ns",
            )
        },
    },
    ("step_id", "step_index", "phase", "request_ids"),
)
STEP_REQUEST = _stamped(
    {
        "step_id": ID,
        "request_id": ID,
        "relation": {
            "enum": ["scheduled", "produced", "prefill", "decode", "recompute"]
        },
        "token_count": NONNEG_INT,
        "sequence_index": NONNEG_INT,
    },
    ("step_id", "request_id", "relation"),
)
CACHE_EVENT = _stamped(
    {
        "event_id": ID,
        "operation": {
            "enum": [
                "can_allocate",
                "hash_blocks",
                "allocate",
                "deallocate",
                "may_append",
            ]
        },
        "event_type": {
            "enum": [
                "lookup",
                "hit",
                "miss",
                "allocate",
                "free",
                "reuse",
                "overwrite",
                "invalidate",
                "recompute",
                "unknown",
            ]
        },
        "run_id": ID,
        "request_id": _nullable(ID),
        "step_id": _nullable(ID),
        "block_id": _nullable(ID),
        "block_generation": _nullable(NONNEG_INT),
        "lifetime_id": _nullable(ID),
        "refcount": _nullable(NONNEG_INT),
        "eligible_blocks": _nullable(NONNEG_INT),
        "hit_blocks": _nullable(NONNEG_INT),
        "miss_blocks": _nullable(NONNEG_INT),
        "logical_kv_tokens": _nullable(NONNEG_INT),
        "physical_kv_bytes": _nullable(NONNEG_INT),
        "physical_blocks": _nullable(NONNEG_INT),
        "resident_reusable_blocks": _nullable(NONNEG_INT),
        "occupancy_bytes": _nullable(NONNEG_INT),
        "recompute_reason": _nullable(STR),
    },
    ("operation",),
)
SCHED_EVENT = _stamped(
    {
        "event_id": ID,
        "event_type": {
            "enum": [
                "admit",
                "preempt",
                "requeue",
                "complete",
                "schedule",
                "wave_start",
                "wave_end",
            ]
        },
        "run_id": ID,
        "request_ids": STRING_LIST,
        "request_count": NONNEG_INT,
        "reason": _nullable(STR),
        "wave_id": _nullable(ID),
        "queue_wait_ns": _nullable(NONNEG_INT),
    },
    ("event_type",),
)
TOKEN_EVENT = _stamped(
    {
        "request_id": ID,
        "token_index": _nullable(NONNEG_INT),
        "token_id": NONNEG_INT,
        "step_id": _nullable(ID),
        "token_type": {
            "enum": [
                "input",
                "generated",
                "observed",
                "unknown",
                "sampled_output",
                "discarded_prefill_sample",
            ]
        },
    },
    ("request_id", "token_index", "token_id", "token_type"),
)
TELEMETRY = _stamped(
    {
        "timestamp_utc": {
            "type": "string",
            "format": "date-time",
            "pattern": "(?:Z|\\+00:00)$",
        },
        "monotonic_ns": NONNEG_INT,
        "pid": NONNEG_INT,
        "source": STR,
        "interval_s": {"type": "number", "exclusiveMinimum": 0},
        "sample_duration_s": _nullable(NUM),
        "sample_scope": {"enum": ["before_measurement", "after_measurement"]},
        "cpu_count": _nullable(NONNEG_INT),
        "memory_total_bytes": _nullable(NONNEG_INT),
        "memory_available_bytes": _nullable(NONNEG_INT),
        "process_rss_bytes": _nullable(NONNEG_INT),
        "process_cpu_percent_since_previous_sample": _nullable(NUM),
        "gpu_devices": _nullable(
            {
                "type": "array",
                "items": HARDWARE_PROPERTIES["gpu_devices"]["anyOf"][0]["items"],
            }
        ),
        "gpu_error": _nullable(STR),
        "torch_cuda_allocator": _nullable(
            {
                "type": "array",
                "items": _object(
                    {
                        "device_index": NONNEG_INT,
                        "allocated_bytes": NONNEG_INT,
                        "reserved_bytes": NONNEG_INT,
                        "peak_allocated_bytes": NONNEG_INT,
                    },
                    (
                        "device_index",
                        "allocated_bytes",
                        "reserved_bytes",
                        "peak_allocated_bytes",
                    ),
                ),
            }
        ),
        "gpu_energy_j": _nullable(NUM),
        "pcie_tx_bytes": _nullable(NONNEG_INT),
        "pcie_rx_bytes": _nullable(NONNEG_INT),
        "nvlink_tx_bytes": _nullable(NONNEG_INT),
        "nvlink_rx_bytes": _nullable(NONNEG_INT),
        "external_process_gpu_memory_bytes": _nullable(NONNEG_INT),
        "torch_cuda_allocator_reason": STR,
        "sample_count": NONNEG_INT,
        "running": BOOL,
        "last_error": _nullable(STR),
        "field_status": _status_map(),
    },
    ("timestamp_utc", "monotonic_ns", "pid", "source", "interval_s"),
)

# Trace event rows require correlation clocks/identity; empty trace arrays remain valid
# for failed runs whose engine never initialized. Telemetry uses pid instead.
for _event_schema in (STEP, STEP_REQUEST, CACHE_EVENT, SCHED_EVENT, TOKEN_EVENT):
    _event_schema["required"].extend(
        ["timestamp_utc", "monotonic_ns", "clock_id", "process_id"]
    )

DATASET_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": f"https://taei.example/schemas/nano-vllm-dataset/{SCHEMA_VERSION}/schema.json",
    "title": "TAEI nano-vLLM experiment dataset",
    "type": "object",
    "additionalProperties": False,
    "required": list(ARTIFACTS),
    "properties": {
        "manifest": {"$ref": "#/$defs/manifest"},
        "hardware": {"$ref": "#/$defs/hardware"},
        "environment": {"$ref": "#/$defs/environment"},
        "model": {"$ref": "#/$defs/model"},
        "workload": {"$ref": "#/$defs/workload"},
        "runs": {"$ref": "#/$defs/run"},
        **{
            name: {"type": "array", "items": {"$ref": f"#/$defs/{name}"}}
            for name in ARTIFACTS[6:]
        },
    },
    "$defs": {
        "manifest": MANIFEST,
        "hardware": _stamped(HARDWARE_PROPERTIES),
        "environment": _stamped(ENVIRONMENT_PROPERTIES),
        "model": _stamped(MODEL_PROPERTIES),
        "workload": WORKLOAD,
        "run": RUN,
        "requests": REQUEST,
        "steps": STEP,
        "step_requests": STEP_REQUEST,
        "cache_events": CACHE_EVENT,
        "scheduler_events": SCHED_EVENT,
        "token_events": TOKEN_EVENT,
        "telemetry": TELEMETRY,
    },
}

# The dictionary uses full artifact paths. It deliberately documents artifact-path
# entries too, so downstream columnar extraction can retain provenance.
_FIELD_META: dict[str, tuple[str, str | None, bool, str, str, str, list[str]]] = {}


def _field(
    path: str,
    typ: str,
    unit: str | None,
    required: bool,
    role: str,
    source: str,
    availability: str = "always",
    null_reasons: tuple[str, ...] = (),
) -> None:
    _FIELD_META[path] = (
        typ,
        unit,
        required,
        role,
        source,
        availability,
        list(null_reasons),
    )


for artifact in ARTIFACTS:
    _field(
        f"{artifact}.{'json' if artifact in ARTIFACTS[:6] else 'jsonl'}",
        "object" if artifact in ARTIFACTS[:6] else "array[object]",
        None,
        True,
        "diagnostic",
        "collector artifact writer",
    )
for path, typ, unit, role, source, avail in [
    ("manifest.schema_version", "string", None, "label", "schema", "always"),
    ("manifest.collector_version", "string", None, "diagnostic", "collector", "always"),
    ("manifest.run_id", "string", None, "label", "orchestrator", "always"),
    (
        "manifest.attempt_id",
        "string",
        None,
        "label",
        "orchestrator",
        "after run directory creation",
    ),
    (
        "manifest.files",
        "object",
        None,
        "diagnostic",
        "artifact writer SHA-256",
        "after artifact flush",
    ),
    (
        "manifest.coverage.primary_complete",
        "boolean",
        None,
        "diagnostic",
        "trace coverage audit",
        "after execution",
    ),
]:
    _field(path, typ, unit, True, role, source, avail)


def _add_group(
    artifact: str,
    fields: dict[str, Any],
    role: str,
    source: str,
    unit_overrides: dict[str, str] | None = None,
) -> None:
    unit_overrides = unit_overrides or {}

    def walk(path: str, spec: dict[str, Any], required: bool = False) -> None:
        branches = spec.get("anyOf", [])
        nullable = any(branch.get("type") == "null" for branch in branches)
        base = next(
            (branch for branch in branches if branch.get("type") != "null"), spec
        )
        kind = base.get("type", "object")
        if kind == "array":
            item = base.get("items", {})
            item_type = item.get("type", "object")
            kind = "array[object]" if item_type == "object" else f"array[{item_type}]"
        _field(
            path,
            kind,
            unit_overrides.get(path.rsplit(".", 1)[-1]),
            required,
            role,
            source,
            "when capability is available" if nullable else "always",
            (
                "unsupported",
                "not_enabled",
                "permission_denied",
                "not_collected",
                "failed",
                "not_applicable",
            )
            if nullable
            else (),
        )
        if base.get("type") == "object":
            required_names = set(base.get("required", []))
            for child, child_spec in base.get("properties", {}).items():
                walk(f"{path}.{child}", child_spec, child in required_names)
        elif (
            base.get("type") == "array"
            and base.get("items", {}).get("type") == "object"
        ):
            item = base["items"]
            required_names = set(item.get("required", []))
            for child, child_spec in item.get("properties", {}).items():
                walk(f"{path}[].{child}", child_spec, child in required_names)

    for name, spec in fields.items():
        walk(
            f"{artifact}.{name}",
            spec,
            name in DATASET_SCHEMA["$defs"].get(artifact, {}).get("required", []),
        )


for _group, _role, _source in (
    ("hardware", "diagnostic", "read-only host/NVML metadata collector"),
    (
        "environment",
        "diagnostic",
        "allowlisted environment and package metadata collector",
    ),
    ("model", "diagnostic", "local model/config/tokenizer metadata collector"),
):
    _add_group(_group, DATASET_SCHEMA["$defs"][_group]["properties"], _role, _source)
for _group, _role, _source in (
    ("run", "label", "orchestrator/run summary"),
    ("requests", "input", "workload and engine request result"),
    ("steps", "diagnostic", "nano-vLLM scheduler/model/cache hooks"),
    ("step_requests", "diagnostic", "scheduler/request join hook"),
    ("cache_events", "diagnostic", "prefix cache/block manager hook"),
    ("scheduler_events", "diagnostic", "scheduler hook"),
    ("token_events", "label", "actual token observation hook"),
    ("telemetry", "diagnostic", "periodic read-only host/NVML sampler"),
):
    _add_group(_group, DATASET_SCHEMA["$defs"][_group]["properties"], _role, _source)
# Add important workload columns absent from the six metadata groups.
_add_group(
    "workload",
    DATASET_SCHEMA["$defs"]["workload"]["properties"],
    "input",
    "deterministic workload builder",
)
FIELD_DICTIONARY = [
    {
        "name": name,
        "type": meta[0],
        "unit": meta[1],
        "required": meta[2],
        "role": meta[3],
        "source": meta[4],
        "availability": meta[5],
        "null_reasons": meta[6],
    }
    for name, meta in _FIELD_META.items()
]


def validate_instance(value: Any) -> list[str]:
    """Return stable, path-qualified strict Draft 2020-12 validation errors."""
    import math

    from jsonschema import Draft202012Validator, FormatChecker, validators

    # JSON numbers are finite by definition, while Python's jsonschema type check
    # otherwise accepts NaN/Infinity. Reject those explicitly and keep bool != int.
    type_checker = Draft202012Validator.TYPE_CHECKER.redefine(
        "number",
        lambda checker, instance: (
            isinstance(instance, (int, float))
            and not isinstance(instance, bool)
            and (not isinstance(instance, float) or math.isfinite(instance))
        ),
    ).redefine(
        "integer",
        lambda checker, instance: (
            isinstance(instance, int) and not isinstance(instance, bool)
        ),
    )
    StrictValidator = validators.extend(Draft202012Validator, type_checker=type_checker)
    validator = StrictValidator(DATASET_SCHEMA, format_checker=FormatChecker())
    result = []
    for error in validator.iter_errors(value):
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        result.append(f"{path}: {error.message}")
    return sorted(result)
