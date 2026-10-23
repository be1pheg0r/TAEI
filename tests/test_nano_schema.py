"""Проверяет версионированную JSON Schema и словарь полей датасета.

Тесты проверяют Draft 2020-12 и обязательные верхнеуровневые артефакты,
допустимость корректно описанных неудачных/неполных запусков, обязательные
статусы null-полей и строгую проверку временных меток, идентификаторов, чисел
и строк трасс. Также проверяются полнота словаря, поддерживаемая терминология
движка и сохранение объявленных расширений при строгой типизации известных полей.
"""

from jsonschema import Draft202012Validator

from taei.nano_dataset.schema import (
    ARTIFACTS,
    DATASET_SCHEMA,
    FIELD_DICTIONARY,
    SCHEMA_VERSION,
    validate_instance,
)


def valid_dataset():
    # Minimal persisted failure artifact: success-only traces and labels may be absent.
    return {
        "manifest": {
            "schema_version": SCHEMA_VERSION,
            "collector_version": "0.1.0",
            "run_id": "R01",
            "attempt_id": "attempt-1",
            "files": {"requests.jsonl": {"sha256": "a" * 64, "bytes": 2}},
            "coverage": {"primary_complete": False, "missing_hooks": ["steps"]},
            "hash_policy": "exact persisted bytes",
        },
        "hardware": {
            "field_status": {
                "gpu_name": {"status": "unsupported", "reason": "no device"},
                "gpu_devices": {"status": "unsupported", "reason": "no device"},
                "gpu_compute_capability": {
                    "status": "unsupported",
                    "reason": "no CUDA",
                },
            },
            "gpu_devices": None,
            "cpu_model": "CPU",
            "cpu_logical_cores": 8,
            "cpu_affinity": [0, 1],
            "gpu_compute_capability": None,
        },
        "environment": {
            "field_status": {
                "cuda_driver_version": {"status": "not_collected", "reason": "no CUDA"}
            },
            "python_version": "3.12.0",
            "cuda_driver_version": None,
            "environment_variables": {"OMP_NUM_THREADS": "1"},
        },
        "model": {
            "field_status": {
                "model_path": {"status": "not_collected", "reason": "no local model"},
                "layer_count": {"status": "not_collected", "reason": "no config"},
                "raw_config": {
                    "status": "not_collected",
                    "reason": "sensitive raw config omitted",
                },
            },
            "model_path": None,
            "layer_count": None,
            "raw_config": None,
        },
        "workload": {"workload_id": "S-seed-42", "requests": []},
        "runs": {
            "run_id": "R01",
            "attempt_id": "attempt-1",
            "status": "failed",
            "primary_status": "rejected",
            "config_id": "cfg",
            "condition_id": "C01",
            "pair_id": "S-seed-42",
            "repeat_id": "01",
            "run_order": 1,
            "hardware_id": "H1",
            "environment_id": "E1",
            "model_id": "M1",
            "workload_id": "S-seed-42",
            "error": {"type": "RuntimeError", "stage": "prepare", "message": "no GPU"},
            "metrics": {"wall_seconds": None, "output_tokens_per_second": None},
            "timing_policy": {"window": "one generate"},
            "effective_config": {},
        },
        "requests": [
            {
                "request_id": "q1",
                "run_id": "R01",
                "attempt_id": "attempt-1",
                "input_token_ids": [1, 2],
                "output_token_ids": None,
                "requested_output_tokens": 2,
                "actual_output_tokens": None,
                "temperature": 0.6,
                "ignore_eos": True,
                "event_time_utc": "2026-10-23T10:00:00+00:00",
                "monotonic_ns": 123,
                "status": "failed",
                "field_status": {
                    "output_token_ids": {"status": "failed", "reason": "no GPU"},
                    "actual_output_tokens": {"status": "failed", "reason": "no GPU"},
                },
            }
        ],
        "steps": [],
        "step_requests": [],
        "cache_events": [],
        "scheduler_events": [],
        "token_events": [],
        "telemetry": [],
    }


def errors(value):
    return list(Draft202012Validator(DATASET_SCHEMA).iter_errors(value))


def test_schema_is_valid_draft_2020_12_and_accepts_failed_incomplete_artifact():
    Draft202012Validator.check_schema(DATASET_SCHEMA)
    assert DATASET_SCHEMA["$id"].endswith(f"/{SCHEMA_VERSION}/schema.json")
    assert not errors(valid_dataset())
    assert validate_instance(valid_dataset()) == []


def test_run_envelope_is_one_object_and_top_level_artifacts_are_required():
    data = valid_dataset()
    data["runs"] = [data["runs"]]
    assert errors(data)
    assert any("runs" in message for message in validate_instance(data))
    del data["runs"]
    assert errors(data)


def test_null_values_require_explicit_per_field_status_and_reason():
    data = valid_dataset()
    data["requests"][0]["field_status"] = {}
    assert errors(data)
    data = valid_dataset()
    data["requests"][0]["field_status"]["output_token_ids"]["reason"] = ""
    assert errors(data)


def test_strict_tokens_timestamp_identifiers_and_finite_numbers():
    data = valid_dataset()
    data["requests"][0]["input_token_ids"] = [1, True]
    assert errors(data)
    data = valid_dataset()
    data["requests"][0]["temperature"] = float("inf")
    assert validate_instance(data)
    data = valid_dataset()
    data["requests"][0]["event_time_utc"] = "2026-10-23 10:00:00"
    assert errors(data)
    data = valid_dataset()
    data["requests"][0]["request_id"] = ""
    assert errors(data)
    data = valid_dataset()
    data["requests"][0]["monotonic_ns"] = True
    assert errors(data)


def test_malformed_detailed_trace_rows_are_rejected():
    mutations = [
        ("steps", {"step_id": "s1", "phase": "mystery", "step_index": True}),
        ("step_requests", {"step_id": "s1", "request_id": "q1", "relation": "maybe"}),
        (
            "cache_events",
            {"event_type": "invented", "timestamp_utc": "bad", "monotonic_ns": -1},
        ),
        ("scheduler_events", {"event_type": "admit", "request_ids": [""]}),
        (
            "telemetry",
            {
                "timestamp_utc": "bad",
                "monotonic_ns": 2,
                "gpu_utilization_percent": float("nan"),
            },
        ),
    ]
    for artifact, row in mutations:
        data = valid_dataset()
        data[artifact] = [row]
        assert errors(data), artifact


def test_dictionary_is_full_name_unique_and_covers_artifact_and_domain_fields():
    names = [entry["name"] for entry in FIELD_DICTIONARY]
    assert len(names) == len(set(names))
    assert {f"{artifact}.json" for artifact in ARTIFACTS[:6]} <= set(names)
    assert {f"{artifact}.jsonl" for artifact in ARTIFACTS[6:]} <= set(names)
    for domain_fields in (
        {
            "hardware.gpu_devices",
            "hardware.gpu_compute_capability",
            "hardware.cpu_model",
            "hardware.memory_total_bytes",
        },
        {
            "environment.python_version",
            "environment.cuda_driver_version",
            "environment.torch_linked_cuda_version",
            "environment.environment_variables",
        },
        {
            "model.layer_count",
            "model.attention_head_count",
            "model.kv_dtype",
            "model.weight_bytes",
            "model.rope_theta",
        },
        {
            "telemetry.timestamp_utc",
            "telemetry.gpu_devices",
            "telemetry.gpu_devices[].utilization_gpu_percent",
            "telemetry.torch_cuda_allocator[].allocated_bytes",
            "telemetry.process_rss_bytes",
        },
        {
            "steps.scheduled_tokens",
            "steps.fresh_prefill_tokens",
            "cache_events.eligible_blocks",
            "scheduler_events.wave_id",
            "requests.output_token_ids",
        },
    ):
        assert domain_fields <= set(names)
    assert all(
        {"type", "unit", "required", "role", "source", "availability", "null_reasons"}
        <= set(item)
        for item in FIELD_DICTIONARY
    )


def test_valid_stamped_trace_rows_accept_contract_engine_vocabulary():
    data = valid_dataset()
    stamp = {
        "timestamp_utc": "2026-10-23T10:00:00+00:00",
        "monotonic_ns": 123,
        "clock_id": "clock-1",
        "process_id": 99,
    }
    data["steps"] = [
        {
            **stamp,
            "step_id": "s1",
            "step_index": 0,
            "phase": "prefill",
            "request_ids": ["q1"],
        }
    ]
    data["cache_events"] = [
        {**stamp, "operation": "can_allocate", "eligible_blocks": 2}
    ]
    data["scheduler_events"] = [
        {**stamp, "event_type": "schedule", "request_ids": ["q1"], "request_count": 1}
    ]
    data["token_events"] = [
        {
            **stamp,
            "request_id": "q1",
            "token_index": 0,
            "token_id": 7,
            "token_type": "sampled_output",
        }
    ]
    assert not errors(data)
    assert validate_instance(data) == []


def test_declared_extensions_are_preserved_but_known_fields_remain_typed():
    data = valid_dataset()
    data["model"]["raw_snapshot_extension"] = {"vendor_field": [1, 2]}
    assert not errors(data)
    data["hardware"]["cpu_logical_cores"] = "eight"
    assert errors(data)
