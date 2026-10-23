"""Собирает метаданные среды и телеметрию только для чтения.

Функции модуля по возможности получают сведения о процессоре, памяти, GPU,
версиях программного окружения, состоянии исходного репозитория и локальной
модели, сопровождая поля статусами доступности и происхождения. Секреты и
полное окружение не сериализуются, веса модели не загружаются; необязательные
системные интерфейсы обрабатываются как best-effort. ``TelemetrySampler``
периодически записывает ограниченные сэмплы метрик хоста и доступных GPU.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _status(
    payload: dict[str, Any], key: str, value: Any, reason: str = "collected"
) -> None:
    payload[key] = value
    payload.setdefault("field_status", {})[key] = {
        "status": "supported" if value is not None else "not_collected",
        "reason": reason if value is not None else reason,
    }


def _put(
    payload: dict[str, Any],
    key: str,
    value: Any,
    status: str | None = None,
    reason: str = "",
) -> None:
    payload[key] = value
    payload.setdefault("field_status", {})[key] = {
        "status": status or ("supported" if value is not None else "unsupported"),
        "reason": reason
        or (
            "collected"
            if value is not None
            else "capability unavailable or not exposed by this host"
        ),
    }


def _finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _read(path: str, default: str = "") -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, PermissionError):
        return default


def _cpu_model() -> str | None:
    text = _read("/proc/cpuinfo")
    for line in text.splitlines():
        if line.lower().startswith(("model name", "hardware")) and ":" in line:
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _memory_bytes() -> tuple[int | None, int | None]:
    values: dict[str, int] = {}
    for line in _read("/proc/meminfo").splitlines():
        match = re.match(
            r"(MemTotal|MemAvailable|SwapTotal|SwapFree):\s+(\d+)\s+kB", line
        )
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    return total, available


def _cpu_topology() -> tuple[int | None, int | None]:
    processors: set[tuple[str, str]] = set()
    sockets: set[str] = set()
    physical = "0"
    core: str | None = None
    for line in _read("/proc/cpuinfo").splitlines() + [""]:
        if not line.strip():
            if core is not None:
                processors.add((physical, core))
            physical, core = "0", None
        elif ":" in line:
            key, value = (part.strip() for part in line.split(":", 1))
            if key == "physical id":
                physical = value
                sockets.add(value)
            elif key == "core id":
                core = value
    return len(sockets) or None, len(processors) or None


def _cpu_cache_bytes() -> int | None:
    caches: dict[tuple[str, str], int] = {}
    for size_path in Path("/sys/devices/system/cpu").glob(
        "cpu[0-9]*/cache/index*/size"
    ):
        try:
            raw = size_path.read_text().strip().upper()
            match = re.fullmatch(r"(\d+)([KMG]?)", raw)
            level = (size_path.parent / "level").read_text().strip()
            cache_id = (size_path.parent / "id").read_text().strip()
            if match:
                factor = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
                caches[(level, cache_id)] = int(match.group(1)) * factor
        except OSError:
            continue
    return sum(caches.values()) if caches else None


def _read_cgroup_limit(filename: str) -> int | None:
    raw = _read(filename)
    if not raw or raw == "max":
        return None
    try:
        value = int(raw)
        return value if value >= 0 else None
    except ValueError:
        return None


def _query_nvml_devices() -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        try:
            import pynvml as nvml
        except ImportError:
            from nvidia_ml_py import pynvml as nvml  # type: ignore[no-redef]
    except ImportError:
        return None, "NVML Python binding is not installed"
    initialized = False
    try:
        nvml.nvmlInit()
        initialized = True
        devices = []
        count = nvml.nvmlDeviceGetCount()
        for index in range(count):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)

            call_status: dict[str, dict[str, str]] = {}

            def call(
                name: str,
                *args: Any,
                _handle: Any = handle,
                _call_status: dict[str, dict[str, str]] = call_status,
            ) -> Any:
                method = getattr(nvml, name, None)
                if not callable(method):
                    _call_status[name] = {
                        "status": "unsupported",
                        "reason": "NVML binding does not expose this query",
                    }
                    return None
                try:
                    value = method(_handle, *args)
                    _call_status[name] = {
                        "status": "supported" if value is not None else "not_collected",
                        "reason": "queried through NVML"
                        if value is not None
                        else "NVML returned no value",
                    }
                    return value
                except Exception as exc:
                    name = type(exc).__name__.lower()
                    status = (
                        "permission_denied"
                        if "permission" in name or isinstance(exc, PermissionError)
                        else "unsupported"
                        if "notsupported" in name or "not_supported" in name
                        else "failed"
                    )
                    _call_status[method.__name__] = {
                        "status": status,
                        "reason": f"NVML query raised {type(exc).__name__}",
                    }
                    return None

            def text(name: str) -> str | None:
                value = call(name)
                if isinstance(value, bytes):
                    return value.decode("utf-8", "replace")
                return str(value) if value is not None else None

            mem = call("nvmlDeviceGetMemoryInfo")
            util = call("nvmlDeviceGetUtilizationRates")
            try:
                temp = call(
                    "nvmlDeviceGetTemperature", getattr(nvml, "NVML_TEMPERATURE_GPU", 0)
                )
            except Exception:
                temp = None
            power = call("nvmlDeviceGetPowerUsage")
            clock = call("nvmlDeviceGetClockInfo", getattr(nvml, "NVML_CLOCK_SM", 0))
            pci = call("nvmlDeviceGetPciInfo")

            def attr(obj: Any, name: str) -> Any:
                return getattr(obj, name, None) if obj is not None else None

            devices.append(
                _finite(
                    {
                        "index": index,
                        "name": text("nvmlDeviceGetName"),
                        "uuid": text("nvmlDeviceGetUUID"),
                        "pci_bus_id": (
                            attr(pci, "busId").decode("utf-8", "replace")
                            if isinstance(attr(pci, "busId"), bytes)
                            else attr(pci, "busId")
                        ),
                        "memory_total_bytes": attr(mem, "total"),
                        "memory_free_bytes": attr(mem, "free"),
                        "memory_used_bytes": attr(mem, "used"),
                        "utilization_gpu_percent": attr(util, "gpu"),
                        "utilization_memory_percent": attr(util, "memory"),
                        "temperature_c": temp,
                        "power_w": power / 1000
                        if isinstance(power, (int, float))
                        else None,
                        "sm_clock_mhz": clock,
                        "memory_clock_mhz": call(
                            "nvmlDeviceGetClockInfo",
                            getattr(nvml, "NVML_CLOCK_MEM", 2),
                        ),
                        "graphics_clock_mhz": call(
                            "nvmlDeviceGetClockInfo",
                            getattr(nvml, "NVML_CLOCK_GRAPHICS", 0),
                        ),
                        "application_sm_clock_mhz": call(
                            "nvmlDeviceGetApplicationsClock",
                            getattr(nvml, "NVML_CLOCK_SM", 0),
                        ),
                        "power_limit_w": (
                            call("nvmlDeviceGetPowerManagementLimit") / 1000
                            if isinstance(
                                call("nvmlDeviceGetPowerManagementLimit"), (int, float)
                            )
                            else None
                        ),
                        "persistence_mode": call("nvmlDeviceGetPersistenceMode"),
                        "compute_mode": call("nvmlDeviceGetComputeMode"),
                        "pci_link_gen": call("nvmlDeviceGetCurrPcieLinkGeneration"),
                        "pci_link_width": call("nvmlDeviceGetCurrPcieLinkWidth"),
                        "pci_max_link_gen": call("nvmlDeviceGetMaxPcieLinkGeneration"),
                        "pci_max_link_width": call("nvmlDeviceGetMaxPcieLinkWidth"),
                    }
                )
            )
            device = devices[-1]
            method_by_field = {
                "name": "nvmlDeviceGetName",
                "uuid": "nvmlDeviceGetUUID",
                "pci_bus_id": "nvmlDeviceGetPciInfo",
                "memory_total_bytes": "nvmlDeviceGetMemoryInfo",
                "memory_free_bytes": "nvmlDeviceGetMemoryInfo",
                "memory_used_bytes": "nvmlDeviceGetMemoryInfo",
                "utilization_gpu_percent": "nvmlDeviceGetUtilizationRates",
                "utilization_memory_percent": "nvmlDeviceGetUtilizationRates",
                "temperature_c": "nvmlDeviceGetTemperature",
                "power_w": "nvmlDeviceGetPowerUsage",
                "sm_clock_mhz": "nvmlDeviceGetClockInfo",
                "memory_clock_mhz": "nvmlDeviceGetClockInfo",
                "graphics_clock_mhz": "nvmlDeviceGetClockInfo",
                "application_sm_clock_mhz": "nvmlDeviceGetApplicationsClock",
                "power_limit_w": "nvmlDeviceGetPowerManagementLimit",
                "persistence_mode": "nvmlDeviceGetPersistenceMode",
                "compute_mode": "nvmlDeviceGetComputeMode",
                "pci_link_gen": "nvmlDeviceGetCurrPcieLinkGeneration",
                "pci_link_width": "nvmlDeviceGetCurrPcieLinkWidth",
                "pci_max_link_gen": "nvmlDeviceGetMaxPcieLinkGeneration",
                "pci_max_link_width": "nvmlDeviceGetMaxPcieLinkWidth",
            }
            device["field_status"] = {
                key: call_status.get(
                    method_by_field.get(key, ""),
                    {
                        "status": "supported" if value is not None else "unsupported",
                        "reason": "queried using NVML"
                        if value is not None
                        else "NVML field not exposed",
                    },
                )
                for key, value in device.items()
                if key not in {"field_status", "index"}
            }
        return devices, None
    except PermissionError as exc:
        return None, f"NVML permission denied: {type(exc).__name__}"
    except Exception as exc:
        return None, f"NVML query failed: {type(exc).__name__}"
    finally:
        if initialized:
            with contextlib.suppress(Exception):
                nvml.nvmlShutdown()


_NVML_QUERY_ACTIVE = threading.Lock()


def _nvml_devices(
    timeout_s: float = 3.0,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Bound driver stalls and permit at most one in-flight native query."""
    if not _NVML_QUERY_ACTIVE.acquire(blocking=False):
        return None, "NVML query still in progress after a prior timeout"
    result: list[tuple[list[dict[str, Any]] | None, str | None]] = []
    finished = threading.Event()

    def query() -> None:
        try:
            result.append(_query_nvml_devices())
        except Exception as exc:
            result.append((None, f"NVML query failed: {type(exc).__name__}"))
        finally:
            _NVML_QUERY_ACTIVE.release()
            finished.set()

    threading.Thread(target=query, name="taei-nvml-query", daemon=True).start()
    if not finished.wait(timeout_s):
        return None, f"NVML query timed out after {timeout_s:g}s"
    return result[0]


def collect_hardware() -> dict[str, Any]:
    """Collect host CPU/memory and accessible NVIDIA device metadata read-only."""
    result: dict[str, Any] = {"field_status": {}}
    cpu_count = os.cpu_count()
    total, available = _memory_bytes()
    sockets, physical_cores = _cpu_topology()
    meminfo = _read("/proc/meminfo")
    swap_match = re.search(r"^SwapTotal:\s+(\d+)\s+kB", meminfo, re.MULTILINE)
    swap_total = int(swap_match.group(1)) * 1024 if swap_match else None
    cpu_max = _read("/sys/fs/cgroup/cpu.max").split()
    quota_cores = None
    if len(cpu_max) == 2 and cpu_max[0] != "max":
        with contextlib.suppress(ValueError, ZeroDivisionError):
            quota_cores = float(cpu_max[0]) / int(cpu_max[1])
    governors = list(
        Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")
    )
    governor = _read(str(governors[0])) if governors else None
    numa_nodes = len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))) or None
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        with contextlib.suppress(OSError):
            affinity = sorted(os.sched_getaffinity(0))
    fields = {
        "hostname_pseudonym": hashlib.sha256(platform.node().encode()).hexdigest()[:16]
        if platform.node()
        else None,
        "os": platform.system() or None,
        "os_release": platform.release() or None,
        "kernel_version": platform.version() or None,
        "architecture": platform.machine() or None,
        "cpu_model": _cpu_model(),
        "cpu_logical_cores": cpu_count,
        "cpu_affinity_count": len(affinity) if affinity is not None else cpu_count,
        "cpu_affinity": affinity,
        "cpu_sockets": sockets,
        "cpu_physical_cores": physical_cores,
        "cpu_cache_bytes": _cpu_cache_bytes(),
        "cpu_governor": governor or None,
        "cpu_quota_cores": quota_cores,
        "cpu_cgroup_memory_limit_bytes": _read_cgroup_limit(
            "/sys/fs/cgroup/memory.max"
        ),
        "numa_nodes": numa_nodes,
        "memory_total_bytes": total,
        "memory_available_bytes": available,
        "swap_total_bytes": swap_total,
    }
    for key, value in fields.items():
        _put(
            result,
            key,
            value,
            reason="read from host OS"
            if value is not None
            else "host did not expose this field",
        )
    devices, error = _nvml_devices()
    _put(
        result, "gpu_devices", devices, reason=error or "queried through read-only NVML"
    )
    result["field_status"]["gpu_devices"]["status"] = (
        "supported"
        if devices is not None
        else ("permission_denied" if error and "permission" in error else "unsupported")
    )
    for key, reason in {
        "cpu_cpuset": "cpuset membership not collected",
        "gpu_visible_rank_map": "logical rank mapping is supplied by runtime config",
        "gpu_compute_capability": "requires CUDA runtime/device inspection",
        "gpu_pcie_link": "NVML link properties not queried",
        "gpu_nvlink_topology": "NVLink topology not queried",
        "gpu_ecc_mode": "ECC mode not queried",
        "gpu_power_limits": "power limits are not queried",
        "gpu_clocks_configured": "configured clock policy not queried",
        "gpu_datasheet_flops": "requires external SKU database",
    }.items():
        _put(result, key, None, "unsupported", reason)
    return _finite(result)


_ENV_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NVIDIA_VISIBLE_DEVICES",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "NODE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCH_CUDA_ARCH_LIST",
)
_PACKAGE_NAMES = {
    "torch": "torch",
    "transformers": "transformers",
    "triton": "triton",
    "flash_attn": "flash-attn",
    "nvidia_nccl": "nvidia-nccl-cu12",
}


def _git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[3]), *args],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _command_output(args: list[str], timeout_s: float = 2.0) -> str | None:
    try:
        completed = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout_s, check=True
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment() -> dict[str, Any]:
    """Collect version and sanitized environment context; never dump env or CLI."""
    result: dict[str, Any] = {"field_status": {}}
    env = {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}
    _put(
        result,
        "environment_variables",
        env,
        reason="explicit non-secret allowlist only",
    )
    _put(
        result,
        "cuda_visible_devices",
        env.get("CUDA_VISIBLE_DEVICES"),
        reason="allowlisted variable absent"
        if "CUDA_VISIBLE_DEVICES" not in env
        else "allowlisted value",
    )
    _put(result, "python_version", platform.python_version())
    for output, package in _PACKAGE_NAMES.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        _put(
            result,
            f"{output}_version",
            version,
            "supported" if version else "unsupported",
            f"{package} distribution not installed"
            if not version
            else "installed distribution metadata",
        )
    root = Path(__file__).resolve().parents[3]
    _put(
        result,
        "source_commit",
        _git(["rev-parse", "HEAD"]),
        reason="git unavailable or not a repository",
    )
    dirty = _git(["status", "--porcelain"])
    _put(
        result,
        "source_dirty",
        bool(dirty) if dirty is not None else None,
        "supported" if dirty is not None else "not_collected",
        "git status only; no diff content persisted"
        if dirty is not None
        else "git status unavailable",
    )
    diff_hash = None
    try:
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "HEAD"],
            capture_output=True,
            timeout=3,
            check=True,
        ).stdout
        digest = hashlib.sha256(diff)
        untracked = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            capture_output=True,
            timeout=3,
            check=True,
        ).stdout.split(b"\0")
        safe_suffixes = {".py", ".toml", ".yaml", ".yml", ".md"}
        for raw_path in untracked:
            if not raw_path:
                continue
            relative = Path(os.fsdecode(raw_path))
            if relative.suffix not in safe_suffixes or any(
                part.lower() in {".ssh", "secrets", "private_specs"}
                for part in relative.parts
            ):
                continue
            source = root / relative
            if source.is_file() and not source.is_symlink():
                digest.update(raw_path)
                with source.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        diff_hash = digest.hexdigest()
    except (OSError, subprocess.SubprocessError):
        pass
    _put(
        result,
        "source_diff_sha256",
        diff_hash,
        reason="hash of diff bytes; diff contents never persisted"
        if diff_hash
        else "no diff hash available",
    )
    for key, reason in {
        "engine_git_pin": "recorded by engine collector",
        "cuda_driver_version": "driver version not exposed by nvidia-smi",
        "cuda_toolkit_version": "CUDA toolkit not exposed by nvcc",
        "torch_linked_cuda_version": "torch runtime not loaded in this process",
        "omp_settings": "only allowlisted OMP_NUM_THREADS is captured",
        "cli_arguments": "intentionally not collected to prevent secret leakage",
        "config_values": "raw configuration intentionally not collected",
        "container_runtime": "container runtime metadata not collected",
    }.items():
        _put(result, key, None, "not_collected", reason)
    smi_output = _command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    driver = smi_output.splitlines()[0].strip() if smi_output else None
    if driver and re.fullmatch(r"[0-9A-Za-z._-]{1,64}", driver):
        _put(result, "cuda_driver_version", driver, reason="read-only nvidia-smi query")
    nvcc_output = _command_output(["nvcc", "--version"])
    toolkit_match = re.search(r"release\s+([0-9.]+)", nvcc_output or "")
    if toolkit_match:
        _put(
            result,
            "cuda_toolkit_version",
            toolkit_match.group(1),
            reason="nvcc version output",
        )
    torch = sys.modules.get("torch")
    linked_cuda = getattr(getattr(torch, "version", None), "cuda", None)
    if linked_cuda:
        _put(
            result,
            "torch_linked_cuda_version",
            str(linked_cuda),
            reason="loaded torch.version.cuda",
        )
    return result


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _model_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.name.endswith(".safetensors") else []
    try:
        return sorted(item for item in path.rglob("*.safetensors") if item.is_file())
    except OSError:
        return []


def _weight_files(path: Path) -> list[Path]:
    """Return recognized local checkpoint files without loading their contents."""
    files = _model_files(path)
    if path.is_dir():
        for pattern in ("pytorch_model*.bin", "model*.pt", "model*.pth"):
            files.extend(item for item in path.glob(pattern) if item.is_file())
    return sorted(set(files))


def _safetensors_info(
    path: Path,
) -> tuple[int | None, int | None, str | None, str | None]:
    """Read safetensors headers; return params, tensor count, dtype, error."""
    count = 0
    params = 0
    dtypes: set[str] = set()
    try:
        with path.open("rb") as stream:
            raw_len = stream.read(8)
            if len(raw_len) != 8:
                return None, None, None, "invalid or truncated safetensors header"
            header_len = int.from_bytes(raw_len, "little")
            if header_len > 100 * 1024 * 1024:
                return None, None, None, "safetensors header exceeds safe read limit"
            header = json.loads(stream.read(header_len))
        for key, entry in header.items():
            if key == "__metadata__" or not isinstance(entry, dict):
                continue
            shape = entry.get("shape")
            dtype = entry.get("dtype")
            if not isinstance(shape, list) or not all(
                isinstance(dim, int) and dim >= 0 for dim in shape
            ):
                return None, None, None, "tensor shape missing or malformed"
            params += math.prod(shape)
            count += 1
            if isinstance(dtype, str):
                dtypes.add(dtype)
        return params, count, ",".join(sorted(dtypes)) or None, None
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return None, None, None, f"header unreadable: {type(exc).__name__}"


def _download_provenance(path: Path, config_hash: str | None) -> dict[str, Any] | None:
    """Read an allowlisted download receipt bound to the current config bytes."""
    marker = path / ".taei-model-download.json"
    try:
        if marker.is_symlink() or marker.stat().st_size > 1024 * 1024:
            return None
        receipt = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict) or receipt.get("status") != "complete":
            return None
        repo_id = receipt.get("repo_id")
        revision = receipt.get("resolved_revision")
        requested = receipt.get("requested_revision")
        files = receipt.get("files")
        if (
            not isinstance(repo_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id)
            or not isinstance(revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or not isinstance(requested, str)
            or not re.fullmatch(r"[A-Za-z0-9._/-]{1,128}", requested)
            or not isinstance(files, dict)
            or not isinstance(files.get("config.json"), dict)
            or config_hash is None
            or files["config.json"].get("sha256") != config_hash
        ):
            return None
        return {
            "repo_id": repo_id,
            "requested_revision": requested,
            "resolved_revision": revision,
            "config_sha256": config_hash,
            "source": "local_download_receipt_bound_to_config_sha256",
        }
    except (OSError, ValueError, UnicodeError, TypeError):
        return None


def collect_model(model_path: str) -> dict[str, Any]:
    """Inspect local HF-style metadata and exact safetensors file/header facts."""
    result: dict[str, Any] = {"field_status": {}}
    path = Path(model_path).expanduser()
    exists = path.exists()
    _put(
        result,
        "model_path",
        str(path.resolve()) if exists else None,
        "supported" if exists else "not_collected",
        "local model path exists" if exists else "local path does not exist",
    )
    config_path = (
        path / "config.json" if path.is_dir() else path.with_name("config.json")
    )
    config: dict[str, Any] = {}
    try:
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
    except (OSError, ValueError):
        config = {}
    config_hash = _sha256(config_path) if config_path.is_file() else None
    _put(
        result,
        "config_sha256",
        config_hash,
        reason="SHA-256 of local config.json"
        if config_hash
        else "config.json absent or unreadable",
    )
    index_path = (
        path / "model.safetensors.index.json"
        if path.is_dir()
        else path.with_name("model.safetensors.index.json")
    )
    index_hash = _sha256(index_path) if index_path.is_file() else None
    _put(
        result,
        "safetensors_index_sha256",
        index_hash,
        "supported" if index_hash else "not_collected",
        "SHA-256 of local safetensors index"
        if index_hash
        else "index absent or unreadable",
    )
    fields = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "layer_count": config.get("num_hidden_layers", config.get("n_layer")),
        "hidden_size": config.get("hidden_size", config.get("n_embd")),
        "intermediate_size": config.get("intermediate_size"),
        "attention_head_count": config.get("num_attention_heads", config.get("n_head")),
        "key_value_head_count": config.get("num_key_value_heads"),
        "vocab_size": config.get("vocab_size"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "head_dim": config.get("head_dim"),
        "rope_theta": config.get("rope_theta"),
        "attention_type": config.get("_attn_implementation"),
        "sliding_window": config.get("sliding_window"),
        "normalization_type": config.get("rms_norm_eps"),
        "rms_norm_epsilon": config.get("rms_norm_eps"),
        "layer_norm_epsilon": config.get("layer_norm_eps"),
        "qk_layer_norm": config.get("qk_layernorm", config.get("qk_norm")),
        "activation": config.get("hidden_act"),
        "tied_weights": config.get("tie_word_embeddings"),
        "weight_dtype": config.get("torch_dtype", config.get("dtype")),
        "quantization_config": {
            key: value
            for key, value in (config.get("quantization_config") or {}).items()
            if key
            in {
                "bits",
                "quant_method",
                "group_size",
                "desc_act",
                "sym",
                "zero_point",
                "use_dq",
            }
            and isinstance(value, (str, int, float, bool, type(None)))
        }
        if isinstance(config.get("quantization_config"), dict)
        else None,
        "declared_parameter_count": config.get("num_parameters"),
    }
    h, heads = fields["hidden_size"], fields["attention_head_count"]
    fields["head_dim"] = fields["head_dim"] or (
        h // heads
        if isinstance(h, int)
        and isinstance(heads, int)
        and heads > 0
        and h % heads == 0
        else None
    )
    for key, value in fields.items():
        _put(
            result,
            key,
            value,
            reason="read from local config.json"
            if value is not None
            else "not present in local config.json",
        )
    _put(
        result,
        "parameter_count",
        None,
        "not_collected",
        "actual count requires complete readable safetensors headers",
    )
    files = _model_files(path) if exists else []
    weight_files = _weight_files(path) if exists else []
    weight_bytes = None
    if weight_files:
        try:
            weight_bytes = sum(p.stat().st_size for p in weight_files)
        except OSError:
            weight_files = []
            files = []
    _put(
        result,
        "weight_bytes",
        weight_bytes,
        "supported" if weight_bytes is not None else "not_collected",
        "sum of actual local checkpoint file sizes"
        if weight_bytes is not None
        else "no readable recognized local checkpoint weights",
    )
    params_list = []
    dtypes = set()
    header_errors = []
    tensor_count = 0
    for file in files:
        params, count, dtype, error = _safetensors_info(file)
        if params is None or count is None:
            header_errors.append(error or "header unavailable")
        else:
            params_list.append(params)
            tensor_count += count
        if dtype:
            dtypes.update(dtype.split(","))
    actual_params = (
        sum(params_list) if files and len(params_list) == len(files) else None
    )
    if actual_params is not None:
        result["parameter_count"] = actual_params
        result["field_status"]["parameter_count"] = {
            "status": "supported",
            "reason": "exact sum of tensor shape products from safetensors headers",
        }
    _put(
        result,
        "tensor_count",
        tensor_count if files and not header_errors else None,
        reason="tensor entries counted from safetensors headers"
        if files and not header_errors
        else "safetensors headers unavailable",
    )
    if dtypes:
        result["weight_dtype"] = ",".join(sorted(dtypes))
        result["field_status"]["weight_dtype"] = {
            "status": "supported",
            "reason": "dtype from safetensors headers",
        }
    tokenizer_files = []
    if path.is_dir():
        for pattern in (
            "tokenizer*.json",
            "vocab*",
            "merges*",
            "added_tokens*.json",
            "special_tokens_map.json",
            "spiece.model",
            "*.tiktoken",
        ):
            tokenizer_files.extend(
                p for p in path.glob(pattern) if p.is_file() and p.name != "config.json"
            )
    tokenizer_digest = hashlib.sha256()
    tokenizer_assets = sorted(set(tokenizer_files))
    if tokenizer_assets:
        for asset in tokenizer_assets:
            tokenizer_digest.update(asset.name.encode())
            tokenizer_digest.update(bytes.fromhex(_sha256(asset) or ""))
        tokenizer_hash = tokenizer_digest.hexdigest()
    else:
        tokenizer_hash = None
    _put(
        result,
        "tokenizer_sha256",
        tokenizer_hash,
        "supported" if tokenizer_hash else "not_collected",
        "hash of local tokenizer asset hashes"
        if tokenizer_hash
        else "tokenizer assets absent",
    )
    tokenizer_config_path = (
        path / "tokenizer_config.json"
        if path.is_dir()
        else path.with_name("tokenizer_config.json")
    )
    tokenizer_config = {}
    try:
        if tokenizer_config_path.is_file():
            tokenizer_config = json.loads(
                tokenizer_config_path.read_text(encoding="utf-8")
            )
    except (OSError, ValueError):
        pass
    chat_template = tokenizer_config.get("chat_template")
    chat_template_hash = (
        hashlib.sha256(
            json.dumps(chat_template, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        if chat_template is not None
        else None
    )
    _put(
        result,
        "chat_template_sha256",
        chat_template_hash,
        "supported" if chat_template_hash else "not_collected",
        "hash of local chat template; template content omitted"
        if chat_template_hash
        else "chat template absent",
    )
    provenance = _download_provenance(path, config_hash)
    _put(
        result,
        "download_provenance",
        provenance,
        "supported" if provenance else "not_collected",
        "completed local download receipt matches current config SHA-256"
        if provenance
        else "no valid completed download receipt bound to current config",
    )
    tokenizer_revision = tokenizer_config.get(
        "tokenizer_revision", tokenizer_config.get("_commit_hash")
    )
    snapshot_revision = (
        path.name
        if path.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", path.name)
        else None
    )
    if not isinstance(tokenizer_revision, str) or not re.fullmatch(
        r"[A-Za-z0-9._-]{1,128}", tokenizer_revision
    ):
        tokenizer_revision = snapshot_revision or (
            provenance["resolved_revision"] if provenance else None
        )
    _put(
        result,
        "tokenizer_revision",
        tokenizer_revision,
        "supported" if tokenizer_revision else "not_collected",
        "local tokenizer metadata, HF snapshot path, or config-bound download receipt"
        if tokenizer_revision
        else "revision not recorded in a recognized local snapshot",
    )
    model_revision = config.get("_commit_hash")
    if not isinstance(model_revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}", model_revision
    ):
        model_revision = snapshot_revision or (
            provenance["resolved_revision"] if provenance else None
        )
    _put(
        result,
        "model_revision",
        model_revision,
        "supported" if model_revision else "not_collected",
        "local model metadata, HF snapshot path, or config-bound download receipt"
        if model_revision
        else "remote revisions are never fetched; local revision is unknown",
    )
    safe_config: dict[str, Any] = {}
    safe_config_keys = {
        "model_type",
        "architectures",
        "num_hidden_layers",
        "n_layer",
        "hidden_size",
        "n_embd",
        "intermediate_size",
        "num_attention_heads",
        "n_head",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "head_dim",
        "rope_theta",
        "sliding_window",
        "rms_norm_eps",
        "layer_norm_eps",
        "hidden_act",
        "tie_word_embeddings",
        "torch_dtype",
        "dtype",
        "num_parameters",
        "num_experts",
        "num_local_experts",
        "num_experts_per_tok",
        "attention_bias",
        "qkv_bias",
        "qk_layernorm",
        "qk_norm",
    }
    safe_string = re.compile(r"[A-Za-z0-9_.:/+-]{1,128}")
    for key in safe_config_keys:
        value = config.get(key)
        safe_scalar = (
            isinstance(value, (str, int, float, bool))
            and (not isinstance(value, str) or bool(safe_string.fullmatch(value)))
            and (not isinstance(value, float) or math.isfinite(value))
        )
        if safe_scalar or (
            isinstance(value, list)
            and all(
                isinstance(item, str) and safe_string.fullmatch(item) for item in value
            )
        ):
            safe_config[key] = value
    if fields["quantization_config"] is not None:
        safe_config["quantization_config"] = fields["quantization_config"]
    commit_hash = config.get("_commit_hash")
    if isinstance(commit_hash, str) and re.fullmatch(r"[0-9a-f]{40}", commit_hash):
        safe_config["_commit_hash"] = commit_hash
    _put(
        result,
        "raw_config",
        safe_config if config else None,
        "supported" if config else "not_collected",
        "sensitive keys and arbitrary nested values removed from allowlisted HF config"
        if config
        else "local config.json unavailable",
    )
    return _finite(result)


class TelemetrySampler:
    """Periodic best-effort host and NVML snapshots, emitted as telemetry rows."""

    def __init__(self, interval_s: float, emit: Callable[[str, dict[str, Any]], Any]):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("interval_s must be finite and greater than zero")
        self.interval_s = float(interval_s)
        self.emit = emit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sample_count = 0
        self._last_error: str | None = None
        self._last_sample: dict[str, Any] | None = None
        self._previous_cpu_time: float | None = None
        self._previous_cpu_wall: float | None = None

    def _sample(self) -> dict[str, Any]:
        devices, error = _nvml_devices()
        total, available = _memory_bytes()
        statm = _read("/proc/self/statm").split()
        rss_bytes = None
        if len(statm) > 1:
            with contextlib.suppress(ValueError, OSError, AttributeError):
                rss_bytes = int(statm[1]) * os.sysconf("SC_PAGE_SIZE")
        times = os.times()
        cpu_time = times.user + times.system
        wall_time = time.monotonic()
        cpu_percent = None
        if self._previous_cpu_time is not None and self._previous_cpu_wall is not None:
            elapsed = wall_time - self._previous_cpu_wall
            if elapsed > 0:
                cpu_percent = 100 * (cpu_time - self._previous_cpu_time) / elapsed
        self._previous_cpu_time = cpu_time
        self._previous_cpu_wall = wall_time
        torch_memory = None
        torch_reason = "PyTorch CUDA allocator not queried"
        torch = sys.modules.get("torch")
        if torch is None:
            torch_reason = "PyTorch is not already loaded in this process"
        else:
            try:
                if torch.cuda.is_available():
                    torch_memory = [
                        {
                            "device_index": index,
                            "allocated_bytes": torch.cuda.memory_allocated(index),
                            "reserved_bytes": torch.cuda.memory_reserved(index),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                                index
                            ),
                        }
                        for index in range(torch.cuda.device_count())
                    ]
                    torch_reason = (
                        "read-only current-process PyTorch allocator counters"
                    )
                else:
                    torch_reason = "PyTorch loaded but CUDA unavailable in this process"
            except Exception as exc:
                torch_reason = f"PyTorch allocator query failed: {type(exc).__name__}"
        row = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "source": "host+NVML best effort",
            "interval_s": self.interval_s,
            "cpu_count": os.cpu_count(),
            "memory_total_bytes": total,
            "memory_available_bytes": available,
            "process_rss_bytes": rss_bytes,
            "process_cpu_percent_since_previous_sample": cpu_percent,
            "gpu_devices": devices,
            "gpu_error": error,
            "torch_cuda_allocator": torch_memory,
            "torch_cuda_allocator_reason": torch_reason,
            "field_status": {
                "gpu_devices": {
                    "status": "supported" if devices is not None else "unsupported",
                    "reason": error or "read-only NVML sampling",
                },
                "torch_cuda_allocator": {
                    "status": "supported"
                    if torch_memory is not None
                    else "unsupported",
                    "reason": torch_reason,
                },
                "process_rss_bytes": {
                    "status": "supported" if rss_bytes is not None else "not_collected",
                    "reason": "current process statm"
                    if rss_bytes is not None
                    else "proc statm unavailable",
                },
                "process_cpu_percent_since_previous_sample": {
                    "status": "supported"
                    if cpu_percent is not None
                    else "not_collected",
                    "reason": "derived from process CPU-time delta"
                    if cpu_percent is not None
                    else "requires a prior sample",
                },
                "memory_total_bytes": {
                    "status": "supported" if total is not None else "not_collected",
                    "reason": "host /proc/meminfo"
                    if total is not None
                    else "host memory total unavailable",
                },
                "memory_available_bytes": {
                    "status": "supported" if available is not None else "not_collected",
                    "reason": "host /proc/meminfo"
                    if available is not None
                    else "host available memory unavailable",
                },
            },
        }
        return _finite(row)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            row = self._sample()
            try:
                self.emit("telemetry", row)
                with self._lock:
                    self._sample_count += 1
                    self._last_error = None
                    self._last_sample = row
            except Exception as exc:
                with self._lock:
                    self._last_error = type(exc).__name__
            self._stop.wait(max(0.0, self.interval_s - (time.monotonic() - started)))

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="taei-telemetry", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, min(self.interval_s + 0.5, 5.0)))
        self._thread = None

    def snapshot(self) -> dict[str, Any]:
        sample = self._sample()
        with self._lock:
            return {
                **sample,
                "sample_count": self._sample_count,
                "last_error": self._last_error,
                "running": bool(self._thread and self._thread.is_alive()),
            }
