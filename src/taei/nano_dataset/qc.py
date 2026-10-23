"""Выполняет fail-closed контроль каталогов попыток nano-vLLM.

Аудитор читает строгие JSON/JSONL-артефакты, отклоняет небезопасные пути и
символические ссылки, нечисловые значения, пропущенные файлы и несовпадающие
хэши/размеры, затем проверяет версионированную схему и межфайловые идентификаторы.
Также сверяются запросы с нагрузкой, связи шагов и запросов, монотонность часов,
события токенов, режимы планировщика/кэша, полнота трасс, калибровочный барьер и
сохранение объема запланированной и выполненной работы. Публичная функция
преобразует ошибки разбора в отчет об отказе: приемка требует полного внутренне
согласованного свидетельства, а не только заявленных счетчиков или статусов.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from .schema import SCHEMA_VERSION, validate_instance

TRACE_FILES = (
    "requests.jsonl",
    "steps.jsonl",
    "step_requests.jsonl",
    "cache_events.jsonl",
    "scheduler_events.jsonl",
    "token_events.jsonl",
    "telemetry.jsonl",
)
REQUIRED_PRIMARY = TRACE_FILES[:-1]
JSON_FILES = (
    "hardware.json",
    "environment.json",
    "model.json",
    "workload.json",
    "runs.json",
)
REQUIRED_FILES = ("manifest.json", *JSON_FILES, *TRACE_FILES)
ID_FIELDS = (
    "run_id",
    "attempt_id",
    "hardware_id",
    "environment_id",
    "model_id",
    "workload_id",
    "config_id",
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line, parse_constant=_reject_constant)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{number}: event must be an object")
        rows.append(value)
    return rows


def _safe_member(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"unsafe manifest path: {name!r}")
    posix = PurePosixPath(name)
    if posix.is_absolute() or any(part in ("..", "") for part in posix.parts):
        raise ValueError(f"unsafe manifest path: {name!r}")
    path = root.joinpath(*posix.parts)
    if path.is_symlink():
        raise ValueError(f"unsafe symlink in run directory: {name}")
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError(f"unsafe path escapes run directory: {name}") from exc
    return path


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _id_joins(
    row: dict[str, Any], canonical: dict[str, Any], label: str, errors: list[str]
) -> None:
    for field, expected in canonical.items():
        if field not in row or row[field] != expected:
            errors.append(
                f"{label} {field} does not join manifest/run "
                f"({row.get(field)!r} != {expected!r})"
            )


def _check_ids(row: dict[str, Any], label: str, errors: list[str]) -> None:
    for key in ID_FIELDS:
        if not isinstance(row.get(key), str) or not row[key]:
            errors.append(f"{label} missing or invalid {key}")


def _audit_run_directory(directory: str | Path) -> dict[str, Any]:
    """Audit persisted nano-vLLM attempt evidence; primary admission is fail-closed."""
    root = Path(directory)
    errors: list[str] = []
    missing: list[str] = []
    data: dict[str, Any] = {}
    if not root.is_dir():
        errors.append("run directory does not exist or is not a directory")
        return {
            "accepted": False,
            "primary_status": "rejected",
            "errors": errors,
            "counts": {},
            "missing_primary_files": list(REQUIRED_PRIMARY),
        }
    root = root.resolve()
    for name in REQUIRED_FILES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            if name in REQUIRED_PRIMARY:
                missing.append(name)
            errors.append(f"missing or unsafe required file: {name}")
    manifest: dict[str, Any] = {}
    try:
        manifest = _read_json(root / "manifest.json")
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid manifest: {exc}")
    records = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(records, dict):
        records = {}
        errors.append("manifest.files must be an object")
    for name, record in records.items():
        try:
            path = _safe_member(root, name)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(record, dict):
            errors.append(f"invalid file record for {name}: expected object")
            continue
        if not path.is_file():
            errors.append(f"manifest file missing: {name}")
            continue
        try:
            raw = path.read_bytes()
            size = path.stat().st_size
        except OSError as exc:
            errors.append(f"cannot read manifest file {name}: {exc}")
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if record.get("sha256") != digest:
            errors.append(f"hash mismatch: {name}")
        byte_count = record.get("bytes")
        if not _int(byte_count) or byte_count != size:
            errors.append(f"byte-size mismatch or invalid bytes: {name}")
    for name in REQUIRED_FILES[1:]:
        if name not in records:
            errors.append(f"manifest has no hash record for required file: {name}")
    for filename in JSON_FILES:
        path = root / filename
        if path.is_file() and not path.is_symlink():
            try:
                data[filename] = _read_json(path)
                if not isinstance(data[filename], dict):
                    errors.append(f"{filename} must contain one JSON object")
                    data[filename] = {}
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{filename}: invalid strict JSON: {exc}")
                data[filename] = {}
    for filename in TRACE_FILES:
        path = root / filename
        if path.is_file() and not path.is_symlink():
            try:
                data[filename] = _jsonl(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{filename}: invalid strict JSONL: {exc}")
                data[filename] = []
        else:
            data[filename] = []

    run = data.get("runs.json", {})
    workload = data.get("workload.json", {})
    coverage = manifest.get("coverage", {}) if isinstance(manifest, dict) else {}
    if not isinstance(coverage, dict):
        coverage = {}
        errors.append("manifest.coverage must be an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"manifest.schema_version must equal {SCHEMA_VERSION}")
    # Validate the schema worker's stable dataset envelope as well as persisted files.
    envelope = {
        "manifest": manifest,
        "hardware": data.get("hardware.json", {}),
        "environment": data.get("environment.json", {}),
        "model": data.get("model.json", {}),
        "workload": workload,
        "runs": run,
        **{name.removesuffix(".jsonl"): data.get(name, []) for name in TRACE_FILES},
    }
    errors.extend(f"schema: {message}" for message in validate_instance(envelope))

    canonical = {key: manifest.get(key) for key in ID_FIELDS}
    for key in ID_FIELDS:
        if not isinstance(canonical[key], str) or not canonical[key]:
            errors.append(f"manifest missing or invalid {key}")
    _id_joins(run, canonical, "runs.json", errors)
    for filename, field in (
        ("hardware.json", "hardware_id"),
        ("environment.json", "environment_id"),
        ("model.json", "model_id"),
    ):
        row = data.get(filename, {})
        if row.get(field) != canonical.get(field):
            errors.append(f"{filename} {field} does not join manifest")
    if workload.get("workload_id") != canonical.get("workload_id"):
        errors.append("workload.json workload_id does not join manifest")
    for name in ("run_id", "attempt_id", "config_id"):
        if run.get(name) != manifest.get(name):
            errors.append(f"runs.json {name} does not join manifest")

    requests = data.get("requests.jsonl", [])
    workload_requests = workload.get("requests")
    if not isinstance(workload_requests, list) or len(workload_requests) != 32:
        errors.append("workload must define exactly 32 requests")
        workload_requests = []
    req_by_id: dict[str, dict[str, Any]] = {}
    expected_ids = []
    for position, request in enumerate(workload_requests):
        if not isinstance(request, dict):
            errors.append(f"workload request {position} must be an object")
            continue
        request_id = request.get("request_id")
        expected_ids.append(request_id)
        if not isinstance(request_id, str) or not request_id or request_id in req_by_id:
            errors.append(
                f"workload has invalid or duplicate request_id at index {position}"
            )
        else:
            req_by_id[request_id] = request
        if request.get("arrival_index") != position:
            errors.append(
                f"workload request order/arrival_index mismatch at {position}"
            )
    seen: set[str] = set()
    for position, row in enumerate(requests):
        rid = row.get("request_id")
        if not isinstance(rid, str) or rid in seen:
            errors.append(
                f"requests.jsonl has invalid or duplicate request_id at row {position}"
            )
            continue
        seen.add(rid)
        source = req_by_id.get(rid)
        if source is None:
            errors.append(f"unknown request {rid} in requests.jsonl")
            continue
        for key in (
            "input_token_ids",
            "requested_output_tokens",
            "temperature",
            "ignore_eos",
            "arrival_index",
            "workload_class",
            "prefix_family_id",
            "prefix_group_id",
            "prefix_token_count",
        ):
            if row.get(key) != source.get(key):
                errors.append(f"request {rid} {key} differs from workload")
        _id_joins(row, canonical, f"request {rid}", errors)
        for key in ("run_id", "attempt_id", "config_id"):
            if row.get(key) != manifest.get(key):
                errors.append(f"request {rid} {key} does not join manifest")
        if run.get("status") == "success":
            out = row.get("output_token_ids")
            count = row.get("actual_output_tokens", row.get("output_token_count"))
            if row.get("status") not in ("completed", "success"):
                errors.append(f"successful run has noncompleted request {rid}")
            if not isinstance(out, list) or any(not _int(token) for token in out):
                errors.append(
                    f"successful request {rid} lacks integer output_token_ids"
                )
            elif len(out) != source.get("requested_output_tokens"):
                errors.append(
                    f"request {rid} output length differs from requested_output_tokens"
                )
            if count != len(out) if isinstance(out, list) else True:
                errors.append(
                    f"request {rid} actual output count disagrees with output IDs"
                )
    if run.get("status") == "success" and (
        len(requests) != 32 or seen != set(req_by_id)
    ):
        errors.append(
            "successful run must contain each of the 32 workload requests exactly once"
        )

    steps = data.get("steps.jsonl", [])
    prefix_mode = run.get("prefix_cache_mode")
    scheduler_mode = run.get("scheduler_mode")
    if prefix_mode not in ("on", "off"):
        errors.append("runs.json prefix_cache_mode is missing or invalid")
    if scheduler_mode not in ("upstream_dynamic", "fixed_batch"):
        errors.append("runs.json scheduler_mode is missing or invalid")
    step_ids: set[Any] = set()
    step_map = {}
    previous_clock: dict[Any, int] = {}
    for index, row in enumerate(steps):
        sid = row.get("step_id")
        if not isinstance(sid, (str, int)) or isinstance(sid, bool):
            errors.append(f"steps.jsonl has invalid step_id at row {index}")
            continue
        if sid in step_ids:
            errors.append(
                f"steps.jsonl has missing or duplicate step_id at row {index}"
            )
        step_ids.add(sid)
        step_map[sid] = row
        req_ids = row.get("request_ids")
        if (
            not isinstance(req_ids, list)
            or not req_ids
            or any(not isinstance(rid, str) for rid in req_ids)
            or len(set(req_ids)) != len(req_ids)
        ):
            errors.append(f"step {sid} has missing/invalid request_ids")
            req_ids = []
        for rid in req_ids:
            if rid not in req_by_id:
                errors.append(f"unknown request {rid} referenced by step {sid}")
        for clock_key in (
            "monotonic_ns",
            "step_host_start_ns",
            "schedule_start_ns",
            "modelcall_start_ns",
            "modelcall_end_ns",
            "step_host_end_ns",
        ):
            val = row.get(clock_key)
            if val is not None:
                if not _int(val):
                    errors.append(f"step {sid} {clock_key} must be integer nanoseconds")
                else:
                    clock = row.get("clock_id", "default")
                    if not isinstance(clock, str) or not clock:
                        errors.append(f"step {sid} clock_id must be a nonempty string")
                        continue
                    if clock_key == "monotonic_ns":
                        if clock in previous_clock and val < previous_clock[clock]:
                            errors.append(f"step monotonic clock inversion for {clock}")
                        previous_clock[clock] = val
        for start, end in (
            ("step_host_start_ns", "step_host_end_ns"),
            ("modelcall_start_ns", "modelcall_end_ns"),
        ):
            if _int(row.get(start)) and _int(row.get(end)) and row[end] < row[start]:
                errors.append(f"step {sid} {end} precedes {start}")
        batch = row.get("actual_batch_size", row.get("batch_size"))
        if batch is not None and (not _int(batch) or batch != len(req_ids)):
            errors.append(f"step {sid} actual batch size disagrees with request_ids")
        reused = row.get("prefix_reused_tokens")
        if reused is not None and (not _int(reused) or reused < 0):
            errors.append(
                f"step {sid} prefix_reused_tokens must be a nonnegative integer"
            )
        if prefix_mode == "off" and reused not in (None, 0):
            errors.append(f"step {sid} reports prefix reuse while prefix cache is off")

    links: set[tuple[Any, Any]] = set()
    for row in data.get("step_requests.jsonl", []):
        rid, sid = row.get("request_id"), row.get("step_id")
        if rid not in req_by_id:
            errors.append(f"unknown request {rid} in step_requests")
        if sid not in step_ids:
            errors.append(f"unknown step {sid} in step_requests")
        pair = (sid, rid)
        if pair in links:
            errors.append(f"duplicate step_requests join {pair}")
        links.add(pair)
    for sid, step in step_map.items():
        if set(step.get("request_ids", [])) != {
            rid for linked_sid, rid in links if linked_sid == sid
        }:
            errors.append(
                f"step {sid} request_ids do not exactly match step_requests joins"
            )

    token_indexes: dict[str, list[tuple[int, int]]] = {}
    for row in data.get("token_events.jsonl", []):
        rid, sid = row.get("request_id"), row.get("step_id")
        if rid not in req_by_id:
            errors.append(f"unknown request {rid} in token_events")
        if sid not in step_ids:
            errors.append(f"unknown step {sid} in token_events")
        index, token = row.get("token_index"), row.get("token_id")
        if row.get("token_type") == "sampled_output":
            if not _int(index) or index < 0 or not _int(token):
                errors.append(
                    f"token event for {rid} needs nonnegative index "
                    "and integer token_id"
                )
            else:
                token_indexes.setdefault(rid, []).append((index, token))
    output_by_id = {
        row.get("request_id"): row.get("output_token_ids") for row in requests
    }
    for rid in req_by_id:
        if run.get("status") == "success":
            ordered = sorted(token_indexes.get(rid, []))
            actual = [token for _, token in ordered]
            expected = output_by_id.get(rid)
            indexes = [i for i, _ in ordered]
            if (
                not isinstance(expected, list)
                or actual != expected
                or indexes != list(range(len(actual)))
            ):
                errors.append(
                    f"request {rid} token events do not exactly match "
                    "ordered output token IDs"
                )
    for table in ("cache_events.jsonl", "scheduler_events.jsonl"):
        for row in data.get(table, []):
            rid, sid = row.get("request_id"), row.get("step_id")
            if rid is not None and rid not in req_by_id:
                errors.append(f"unknown request {rid} in {table}")
            if sid is not None and sid not in step_ids:
                errors.append(f"unknown step {sid} in {table}")
    schedule_events = [
        row
        for row in data.get("scheduler_events.jsonl", [])
        if row.get("event") == "schedule"
    ]
    if not schedule_events:
        errors.append("scheduler trace has no schedule decisions")
    for event in schedule_events:
        admitted = event.get("admitted_request_ids")
        if not isinstance(admitted, list) or any(
            rid not in req_by_id for rid in admitted
        ):
            errors.append("scheduler schedule event has invalid admitted_request_ids")
            continue
        if scheduler_mode == "fixed_batch" and (
            len(admitted) > 8 or not isinstance(event.get("wave_id"), (str, int))
        ):
            errors.append("fixed_batch schedule must use a wave_id and admit at most 8")
    for event in data.get("cache_events.jsonl", []):
        mode = event.get("prefix_cache_mode")
        if mode is not None and mode != prefix_mode:
            errors.append("cache event prefix_cache_mode disagrees with run config")
        if prefix_mode == "off" and event.get("operation") in (
            "hash_blocks",
            "can_allocate",
        ):
            errors.append(
                "prefix-cache-off trace includes prefix lookup/hash operation"
            )
    if not steps:
        errors.append("missing primary step trace records")
    if not data.get("step_requests.jsonl"):
        errors.append("missing primary request-step joins")
    if not data.get("token_events.jsonl"):
        errors.append("missing primary token events")
    if not data.get("scheduler_events.jsonl"):
        errors.append("missing primary scheduler events")
    if not data.get("cache_events.jsonl"):
        errors.append("missing primary cache events")
    if not any(error.startswith("schema:") for error in errors):
        _check_work_conservation(data, run, req_by_id, errors)
    if coverage.get("primary_complete") is not True:
        errors.append("manifest marks primary trace coverage incomplete or unknown")
    missing_hooks = coverage.get("missing_hooks")
    if not isinstance(missing_hooks, list) or missing_hooks:
        errors.append("manifest missing_hooks is absent, invalid, or nonempty")
    # Gate primary analysis until paired instrumentation calibration.
    timing = run.get("timing_policy")
    if (
        not isinstance(timing, dict)
        or timing.get("primary_analysis_gate") != "calibrated"
    ):
        errors.append(
            "overhead calibration is pending or unverified; primary acceptance is gated"
        )
    if run.get("primary_status") != "accepted":
        errors.append("runs.json does not declare primary_status accepted")
    if run.get("status") == "success":
        metrics = run.get("metrics")
        if not isinstance(metrics, dict):
            errors.append("successful runs.json metrics must be an object")
        else:
            if metrics.get("request_count") != 32:
                errors.append("successful run metrics request_count must equal 32")
            actual_outputs = sum(
                len(row.get("output_token_ids", []))
                for row in requests
                if isinstance(row.get("output_token_ids"), list)
            )
            if metrics.get("output_tokens") != actual_outputs:
                errors.append(
                    "successful run output_tokens metric disagrees with request outputs"
                )
    if run.get("status") != "success":
        errors.append(f"attempt status is {run.get('status')!r}, not success")
    counts = {
        filename.removesuffix(".jsonl"): len(data.get(filename, []))
        for filename in TRACE_FILES
    }
    return {
        "accepted": not errors,
        "primary_status": "accepted" if not errors else "rejected",
        "errors": errors,
        "counts": counts,
        "missing_primary_files": missing,
    }


def _check_work_conservation(data, run, requests, errors):
    """Reconcile independent traces instead of trusting their scalar counters."""
    produced = Counter()
    for token in data.get("token_events.jsonl", []):
        if token.get("token_type") == "sampled_output":
            produced[(token.get("step_id"), token.get("request_id"))] += 1
    links = defaultdict(list)
    for link in data.get("step_requests.jsonl", []):
        links[link.get("step_id")].append(link)
    decisions = {
        event.get("step_id"): event
        for event in data.get("scheduler_events.jsonl", [])
        if event.get("event") == "schedule"
    }
    generated = Counter()
    current_wave = None
    wave_members = set()
    finished_waves = set()
    for step in data.get("steps.jsonl", []):
        sid = step.get("step_id")
        request_ids = step.get("request_ids", [])
        step_links = links[sid]
        observed_outputs = sum(produced[(sid, rid)] for rid in request_ids)
        if step.get("produced_tokens") != observed_outputs:
            errors.append(f"step {sid} produced_tokens disagrees with token events")
        scheduled = [link.get("scheduled_tokens") for link in step_links]
        if not all(_int(value) and value >= 0 for value in scheduled):
            errors.append(f"step {sid} lacks per-request scheduled_tokens")
        elif step.get("scheduled_tokens") != sum(scheduled):
            errors.append(f"step {sid} scheduled_tokens disagrees with request work")
        if run.get("scheduler_mode") == "fixed_batch":
            wave = step.get("wave_id", decisions.get(sid, {}).get("wave_id"))
            if wave is None:
                errors.append(f"fixed step {sid} has no observable wave_id")
            elif wave != current_wave:
                if current_wave is not None:
                    if any(
                        generated[rid] != requests[rid]["requested_output_tokens"]
                        for rid in wave_members
                    ):
                        errors.append(f"wave {wave} starts before prior wave completes")
                    finished_waves.add(current_wave)
                if wave in finished_waves:
                    errors.append(f"fixed wave {wave} is reopened")
                current_wave, wave_members = wave, set()
            wave_members.update(request_ids)
            if len(wave_members) > 8:
                errors.append(f"fixed wave {wave} contains more than 8 requests")
        for link in step_links:
            rid = link.get("request_id")
            if rid not in requests:
                continue
            count = produced[(sid, rid)]
            before, after = (
                link.get("context_tokens_before"),
                link.get("context_tokens_after"),
            )
            expected_before = len(requests[rid]["input_token_ids"]) + generated[rid]
            if (
                not _int(before)
                or not _int(after)
                or before != expected_before
                or after != before + count
            ):
                errors.append(f"step {sid} request {rid} context/token mismatch")
            if link.get("produced_tokens") != count:
                errors.append(f"step {sid} request {rid} produced_tokens mismatch")
            generated[rid] += count


def audit_run_directory(directory: str | Path) -> dict[str, Any]:
    """Return a rejected report rather than raising on malformed artifacts."""
    try:
        return _audit_run_directory(directory)
    except Exception as exc:  # malformed nested JSON types can break set/dict joins
        return {
            "accepted": False,
            "primary_status": "rejected",
            "errors": [
                "malformed artifact prevented complete audit: "
                f"{type(exc).__name__}: {exc}"
            ],
            "counts": {},
            "missing_primary_files": list(REQUIRED_PRIMARY),
        }
