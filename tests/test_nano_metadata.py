"""Проверяет безопасный сбор метаданных и телеметрии запусков датасета.

Проверяются CPU/память и необязательные GPU-возможности, обозначение
недоступных данных, фильтрация переменных окружения и вложенных секретов,
чистота репозитория, анализ метаданных модели и заголовков весов без загрузки
весов, корректность происхождения числа параметров, варианты успеха/ошибок/
тайм-аутов NVML и конечные периодические сэмплы с безопасной остановкой.
"""

import hashlib
import json
import math
import sys
import time
import types

from taei.nano_dataset import metadata


def assert_status_complete(payload):
    assert isinstance(payload["field_status"], dict)
    for key, value in payload.items():
        if key == "field_status":
            continue
        assert key in payload["field_status"]
        assert (
            value is not None or payload["field_status"][key]["status"] != "supported"
        )
        assert payload["field_status"][key]["status"] in {
            "supported",
            "unsupported",
            "not_enabled",
            "permission_denied",
            "not_collected",
            "failed",
            "not_applicable",
        }
        assert isinstance(payload["field_status"][key]["reason"], str)


def test_collectors_report_live_cpu_and_explicit_optional_capabilities():
    hardware = metadata.collect_hardware()
    environment = metadata.collect_environment()
    assert_status_complete(hardware)
    assert_status_complete(environment)
    assert hardware["cpu_model"]
    assert isinstance(hardware["cpu_logical_cores"], int)
    assert environment["python_version"]
    assert "gpu_devices" in hardware and "gpu_devices" in hardware["field_status"]
    assert "cuda_visible_devices" in environment
    assert isinstance(environment["environment_variables"], dict)
    assert not any(
        "SECRET" in k or "TOKEN" in k or "PASSWORD" in k
        for k in environment["environment_variables"]
    )


def test_environment_whitelist_never_serializes_secret_values(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "do-not-leak-token")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-leak-aws")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    result = metadata.collect_environment()
    serialized = json.dumps(result)
    assert "do-not-leak-token" not in serialized
    assert "do-not-leak-aws" not in serialized
    assert result["cuda_visible_devices"] == "0"
    assert "--api-token=cli-do-not-leak" not in json.dumps(result)


def test_collect_model_missing_path_has_named_reasons(tmp_path):
    result = metadata.collect_model(str(tmp_path / "not-here"))
    assert_status_complete(result)
    assert result["model_path"] is None
    assert result["weight_bytes"] is None
    assert result["parameter_count"] is None
    assert result["field_status"]["weight_bytes"]["status"] == "not_collected"


def test_collect_model_counts_real_safetensors_header_without_loading_weights(tmp_path):
    path = tmp_path / "model.safetensors"
    header = {
        "__metadata__": {"format": "pt"},
        "a": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, 12]},
        "b": {"dtype": "F16", "shape": [1], "data_offsets": [12, 14]},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"x" * 14)
    result = metadata.collect_model(str(tmp_path))
    assert result["weight_bytes"] == path.stat().st_size
    assert result["parameter_count"] == 7
    assert result["tensor_count"] == 2
    assert result["weight_dtype"] == "F16"
    assert_status_complete(result)


def test_pytorch_checkpoint_bytes_are_counted_without_estimating_parameters(tmp_path):
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"real")
    result = metadata.collect_model(str(tmp_path))
    assert result["weight_bytes"] == 4
    assert result["parameter_count"] is None


def test_collect_model_huggingface_config_fixture(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fixture",
                "architectures": ["FixtureLM"],
                "num_hidden_layers": 4,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 100,
                "max_position_embeddings": 128,
                "torch_dtype": "float16",
                "tie_word_embeddings": True,
            }
        )
    )
    (tmp_path / "tokenizer.json").write_text('{"version":"1"}')
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {"chat_template": "{{ messages }}", "tokenizer_revision": "fixture-rev"}
        )
    )
    result = metadata.collect_model(str(tmp_path))
    assert result["model_type"] == "fixture"
    assert result["layer_count"] == 4
    assert result["head_dim"] == 4
    assert result["weight_dtype"] == "float16"
    assert result["tied_weights"] is True
    assert result["config_sha256"]
    assert result["tokenizer_sha256"]
    assert result["chat_template_sha256"]
    assert_status_complete(result)


def test_declared_parameter_count_is_not_labeled_as_actual(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_parameters": 1234}))
    result = metadata.collect_model(str(tmp_path))
    assert result["parameter_count"] is None
    assert result["declared_parameter_count"] == 1234
    assert result["field_status"]["parameter_count"]["status"] == "not_collected"


def test_model_head_dim_prefers_explicit_config_value(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "head_dim": 128,
            }
        )
    )
    result = metadata.collect_model(str(tmp_path))
    assert result["head_dim"] == 128


def test_huggingface_snapshot_revision_is_recognized_from_known_layout(tmp_path):
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    snapshot = tmp_path / "models--Qwen--Qwen3-0.6B" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    result = metadata.collect_model(str(snapshot))
    assert result["model_revision"] == revision
    assert result["tokenizer_revision"] == revision


def test_download_receipt_revision_is_bound_to_local_config(tmp_path):
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    config = b'{"model_type":"qwen3"}'
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / ".taei-model-download.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "repo_id": "Qwen/Qwen3-0.6B",
                "requested_revision": revision,
                "resolved_revision": revision,
                "files": {
                    "config.json": {"sha256": hashlib.sha256(config).hexdigest()}
                },
                "token": "never-serialize-receipt-secret",
            }
        )
    )
    result = metadata.collect_model(str(tmp_path))
    assert result["model_revision"] == revision
    assert result["tokenizer_revision"] == revision
    assert result["download_provenance"]["repo_id"] == "Qwen/Qwen3-0.6B"
    assert "never-serialize-receipt-secret" not in json.dumps(result)
    assert_status_complete(result)


def test_stale_download_receipt_never_labels_different_model_revision(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"changed"}')
    (tmp_path / ".taei-model-download.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "repo_id": "Qwen/Qwen3-0.6B",
                "requested_revision": "main",
                "resolved_revision": "a" * 40,
                "files": {"config.json": {"sha256": "0" * 64}},
            }
        )
    )
    result = metadata.collect_model(str(tmp_path))
    assert result["model_revision"] is None
    assert result["download_provenance"] is None
    assert result["field_status"]["download_provenance"]["status"] != "supported"


def test_clean_repository_is_not_reported_dirty(monkeypatch):
    monkeypatch.setattr(
        metadata, "_git", lambda args: "" if args[0] == "status" else "abc123"
    )
    result = metadata.collect_environment()
    assert result["source_dirty"] is False
    assert result["field_status"]["source_dirty"]["status"] == "supported"


def test_model_config_arbitrary_nested_secret_is_not_returned(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fixture",
                "quantization_config": {"token": "model-config-secret", "bits": 4},
            }
        )
    )
    result = metadata.collect_model(str(tmp_path))
    assert "model-config-secret" not in json.dumps(result)
    assert result["raw_config"]["quantization_config"] == {"bits": 4}


def test_nvml_optional_fixture_and_unavailable_reason(monkeypatch):
    class Device:
        name = b"Fixture GPU"
        uuid = b"GPU-fixture"
        pci = types.SimpleNamespace(busId=b"0000:01:00.0")
        memory = types.SimpleNamespace(total=1024, free=768, used=256)
        temperature = 61
        power = 45000
        clock = 1500
        utilization = types.SimpleNamespace(gpu=42, memory=12)

    class Nvml:
        NVML_TEMPERATURE_GPU = 0
        NVML_CLOCK_SM = 0
        NVML_POWER_LIMIT = 0

        def nvmlInit(self):
            pass

        def nvmlShutdown(self):
            pass

        def nvmlDeviceGetCount(self):
            return 1

        def nvmlDeviceGetHandleByIndex(self, i):
            return Device()

        def nvmlDeviceGetName(self, d):
            return d.name

        def nvmlDeviceGetUUID(self, d):
            return d.uuid

        def nvmlDeviceGetPciInfo(self, d):
            return d.pci

        def nvmlDeviceGetMemoryInfo(self, d):
            return d.memory

        def nvmlDeviceGetTemperature(self, d, s):
            return d.temperature

        def nvmlDeviceGetPowerUsage(self, d):
            return d.power

        def nvmlDeviceGetClockInfo(self, d, s):
            return d.clock

        def nvmlDeviceGetUtilizationRates(self, d):
            return d.utilization

    monkeypatch.setitem(sys.modules, "pynvml", Nvml())
    result = metadata.collect_hardware()
    assert result["gpu_devices"][0]["name"] == "Fixture GPU"
    assert result["gpu_devices"][0]["memory_total_bytes"] == 1024
    assert result["gpu_devices"][0]["pci_bus_id"] == "0000:01:00.0"
    json.dumps(result)
    assert math.isfinite(result["gpu_devices"][0]["power_w"])
    assert_status_complete(result)


def test_nvml_permission_denial_is_explicit(monkeypatch):
    class Device:
        memory = types.SimpleNamespace(total=10, free=5, used=5)
        utilization = types.SimpleNamespace(gpu=0, memory=0)
        pci = types.SimpleNamespace(busId=b"0000:00:01.0")

    class Nvml:
        def nvmlInit(self):
            pass

        def nvmlShutdown(self):
            pass

        def nvmlDeviceGetCount(self):
            return 1

        def nvmlDeviceGetHandleByIndex(self, index):
            return Device()

        def nvmlDeviceGetName(self, device):
            return b"Fixture"

        def nvmlDeviceGetUUID(self, device):
            return b"GPU-fixture"

        def nvmlDeviceGetPciInfo(self, device):
            return device.pci

        def nvmlDeviceGetMemoryInfo(self, device):
            return device.memory

        def nvmlDeviceGetUtilizationRates(self, device):
            return device.utilization

        def nvmlDeviceGetTemperature(self, device, sensor):
            return 30

        def nvmlDeviceGetPowerUsage(self, device):
            raise PermissionError("denied")

        def nvmlDeviceGetClockInfo(self, device, kind):
            return 100

    monkeypatch.setitem(sys.modules, "pynvml", Nvml())
    result = metadata.collect_hardware()
    assert (
        result["gpu_devices"][0]["field_status"]["power_w"]["status"]
        == "permission_denied"
    )


def test_nvml_missing_device_provider_is_explicit(monkeypatch):
    monkeypatch.setattr(
        metadata,
        "_query_nvml_devices",
        lambda: (None, "NVML Python binding is not installed"),
    )
    result = metadata.collect_hardware()
    assert result["gpu_devices"] is None
    assert result["field_status"]["gpu_devices"]["status"] == "unsupported"


def test_nvml_timeout_does_not_spawn_additional_stuck_workers(monkeypatch):
    monkeypatch.setattr(
        metadata,
        "_query_nvml_devices",
        lambda: time.sleep(0.05) or ([], None),
    )
    result = metadata._nvml_devices(timeout_s=0.001)
    assert result[0] is None
    assert result[1] and "timed out" in result[1]
    second = metadata._nvml_devices(timeout_s=0.001)
    assert second[0] is None
    assert second[1] and "in progress" in second[1]
    time.sleep(0.06)


def test_telemetry_samples_finite_rows_and_stops():
    rows = []
    sampler = metadata.TelemetrySampler(
        0.02, lambda table, row: rows.append((table, row))
    )
    sampler.start()
    time.sleep(0.07)
    snapshot = sampler.snapshot()
    sampler.stop()
    assert rows
    assert all(table == "telemetry" for table, _ in rows)
    assert snapshot["sample_count"] >= 1
    assert snapshot["process_rss_bytes"] > 0
    assert snapshot["monotonic_ns"] > 0
    assert "torch_cuda_allocator" in snapshot
    assert "field_status" in snapshot
    assert "process_rss_bytes" in snapshot["field_status"]
    for _, row in rows:
        assert row["timestamp_utc"]
        assert isinstance(row["monotonic_ns"], int)
        assert all(
            not isinstance(value, float) or math.isfinite(value)
            for value in row.values()
        )


def test_telemetry_stop_is_idempotent_and_no_rows_after_stop():
    rows = []
    sampler = metadata.TelemetrySampler(0.01, lambda table, row: rows.append(row))
    sampler.stop()
    sampler.start()
    sampler.stop()
    count = len(rows)
    time.sleep(0.03)
    assert len(rows) == count
