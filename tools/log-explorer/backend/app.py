from __future__ import annotations

import ast
import csv
import io
import json
import math
import os
import re
import stat
import tempfile
import threading
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.formparsers import MultiPartException

BASE = Path(__file__).resolve().parent
FRONTEND = BASE.parent / "frontend" / "dist"
# Large archives are spooled to disk. Parsing retains bounded previews; archive
# size/member count are not correctness limits. Complexity/ratio guards remain.
MAX_UPLOAD = MAX_REQUEST = MAX_EXPANDED = MAX_FILES = MAX_MEMBER = None
MAX_ROWS = 500
MAX_FIELDS = 100_000
MAX_PREVIEW_TABLE_ROWS = 500
MAX_DEPTH = 64
MAX_LINE = 2 * 1024 * 1024
MAX_LITERAL = 1024 * 1024
MAX_RATIO = 200
PARSE_GATE = threading.BoundedSemaphore(2)
TABLES = (
    "steps",
    "requests",
    "step_requests",
    "cache_events",
    "scheduler_events",
    "token_events",
    "telemetry",
    "runs",
)
LEVEL = re.compile(
    r"^\s*(TRACE|DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\s*[|:]\s*(.*)$", re.I
)
FIELD_DESCRIPTIONS = {
    "experiment": "Имя экспериментального протокола из исходного журнала.",
    "mode": "Режим процедуры: планирование, проверка либо выполнение запусков.",
    "attempt_count": "Количество попыток выполнения, объявленное процедурой.",
    "accepted_primary_count": (
        "Число результатов, принятых первичным контролем качества."
    ),
    "hardware_id": "Стабильный идентификатор аппаратной конфигурации.",
    "environment_id": "Стабильный идентификатор программного окружения.",
    "model_id": (
        "Стабильный идентификатор модели; не название модели и не хеш всех весов."
    ),
    "workload_id": "Идентификатор входной нагрузки и её происхождения.",
    "config_id": "Идентификатор эффективной конфигурации запуска.",
    "path": "Путь к артефактам в исходном окружении; сервис не открывает этот путь.",
    "engine": "Название движка генерации, объявленное источником.",
    "engine_revision": "Зафиксированная ревизия исходного кода движка.",
    "tensor_parallel_size": "Число устройств в группе тензорного параллелизма.",
    "gpu_memory_utilization": "Доля памяти GPU, разрешённая движку; от 0 до 1.",
    "max_model_len": "Настроенная максимальная длина контекста, токены.",
    "effective_max_model_len": "Фактически применённый предел контекста, токены.",
    "max_num_batched_tokens": "Предел числа токенов, планируемых за шаг.",
    "max_num_seqs": "Настроенный предел последовательностей в планировании.",
    "scheduler_max_num_seqs": "Предел последовательностей, переданный планировщику.",
    "enforce_eager": "Принудительный eager-режим вместо CUDA Graphs.",
    "kvcache_block_size": "Вместимость одного блока KV-кэша, токены.",
    "num_kvcache_blocks": "Количество выделенных блоков KV-кэша.",
    "kv_cache_capacity_bytes": "Объявленная вместимость KV-кэша, байты.",
    "model_dtype": "Тип чисел для вычислений модели.",
    "kv_cache_dtype": "Тип чисел при хранении KV-кэша.",
    "cuda_graphs_configured": "Признак настройки CUDA Graphs; не измерение ускорения.",
    "cuda_graph_capture_batch_sizes": (
        "Размеры пакетов для захвата CUDA Graphs, последовательности."
    ),
    "sampling_rng": "Описание генераторов случайных чисел и момента установки seed.",
    "prefix_cache_mode": "Режим переиспользования префиксов KV-кэша.",
    "scheduler_mode": "Режим допуска запросов планировщиком.",
    "upstream_dynamic_admission_limit": (
        "Ограничения семантики допуска в исходном движке."
    ),
    "measurement_profile": "Профиль полноты инструментирования измерений.",
    "primary_complete": "Заявленная полнота основного набора измерений; не приёмка QC.",
    "backend_constructed": "Признак успешного создания реального backend.",
    "real_nano_vllm": (
        "Признак использования реального nano-vLLM, заявленный адаптером."
    ),
    "detailed_trace_hooks": "Доступность подробных обработчиков трассировки.",
    "primary_trace_hooks": "Классы основных обработчиков трассировки.",
    "prefix_cache_toggle": "Поддержка переключения переиспользования префиксов.",
    "fixed_wave_admission": "Поддержка допуска фиксированными волнами запросов.",
    "per_step_gpu_timestamps": (
        "Доступность GPU-времени каждого шага; false не означает ноль."
    ),
    "gpu_generated_token_timestamps": "Доступность GPU-времени каждого токена.",
    "gpu_generated_token_timestamp": (
        "Статус доступности GPU-меток токенов и причина ограничения."
    ),
    "gpu_step_duration": (
        "Статус доступности длительностей GPU-шагов и причина ограничения."
    ),
    "accepted": "Результат приёмки набора данных контролем качества.",
    "errors": "Причины отклонения или ограничений, записанные QC.",
    "missing_primary_files": "Обязательные основные файлы, отсутствующие согласно QC.",
    "requests": "Количество записей запросов, заявленное QC; не сами запросы.",
    "steps": "Количество записей шагов движка, заявленное QC.",
    "step_requests": "Количество записей связи между шагами и запросами.",
    "cache_events": "Количество зарегистрированных событий KV-кэша.",
    "scheduler_events": "Количество зарегистрированных событий планировщика.",
    "token_events": "Количество событий токенов; не независимое измерение выхода.",
    "telemetry": "Количество снимков телеметрии; не длительность измерения.",
    "all_stable_ids_constant": (
        "Совпадение стабильных аппаратных, программных и модельных ID."
    ),
    "backend_constructed_count": "Число попыток с созданным backend.",
    "real_generate_completed_count": "Число попыток с завершённым реальным generate.",
    "gpu_validation_claimed": "Заявляет ли сводка независимую проверку GPU-исполнения.",
    "runs": (
        "Сводка попыток: конфигурация, происхождение, результат генерации "
        "и независимая приёмка QC."
    ),
    "homogeneity": (
        "Проверка постоянства аппаратной, программной и модельной "
        "идентичности запусков."
    ),
    "execution_evidence": (
        "Сводка свидетельств создания backend и завершения реальной генерации."
    ),
    "run_id": "Неизменяемый идентификатор запуска; не является оценкой качества.",
    "attempt_id": "Идентификатор конкретной попытки выполнения.",
    "condition_id": (
        "Идентификатор экспериментального условия для сопоставления запусков."
    ),
    "pair_id": "Идентификатор парного сравнения условий.",
    "split_group_id": (
        "Группа разбиения для предотвращения утечки повторных наблюдений."
    ),
    "repeat_id": "Идентификатор повтора в экспериментальном дизайне.",
    "sha256": "SHA-256 содержимого артефакта в шестнадцатеричном формате.",
    "field_status": (
        "Статус доступности измерений с причиной; null не означает нулевое измерение."
    ),
    "reason": "Записанная источником причина статуса или недоступности.",
    "wall_seconds": "Длительность окна измерения, секунды.",
    "output_tokens_per_second": (
        "Скорость вывода, токены в секунду; значение источника без пересчета."
    ),
    "requests_per_second": "Пропускная способность запросов, запросы в секунду.",
    "temperature_c": "Температура устройства, градусы Цельсия.",
    "power_w": "Мощность устройства, ватт.",
    "power_limit_w": "Ограничение мощности, ватт.",
    "memory_total_bytes": "Объем памяти, байты.",
    "memory_free_bytes": "Свободная память устройства, байты.",
    "memory_used_bytes": "Используемая память устройства, байты.",
    "interval_s": "Интервал между замерами телеметрии, секунды.",
    "sample_duration_s": "Продолжительность сбора выборки, секунды.",
    "monotonic_ns": "Монотонное время процесса, наносекунды; не UTC-время.",
    "timestamp_utc": "Метка времени UTC, записанная источником.",
    "event_time_utc": "Метка времени события UTC, записанная источником.",
    "token_id": "Идентификатор токена без декодирования или изменения.",
    "input_token_ids": "Упорядоченные идентификаторы входных токенов из источника.",
    "output_token_ids": (
        "Упорядоченные идентификаторы выходных токенов; null сохраняется "
        "как отсутствие наблюдения."
    ),
    "requested_output_tokens": "Число запрошенных выходных токенов.",
    "actual_output_tokens": "Число фактически записанных выходных токенов.",
    "produced_tokens": "Число произведенных токенов на шаге, по данным источника.",
    "scheduled_tokens": "Число токенов, запланированных на шаге.",
    "prefix_reused_tokens": "Число переиспользованных токенов префикса, если измерено.",
    "gpu_energy_j": "Энергия GPU, джоули.",
    "process_rss_bytes": "Резидентная память процесса, байты.",
    "source_commit": "Идентификатор коммита исходного кода.",
    "engine_git_pin": "Зафиксированная ревизия движка.",
    "config_sha256": "SHA-256 конфигурации; не подтверждает целостность весов модели.",
    "tokenizer_sha256": "SHA-256 файлов токенизатора.",
    "weight_bytes": "Размер весовых данных модели, байты.",
    "status": "Статус, записанный производителем; отдельно от статуса приемки данных.",
    "primary_status": "Статус первичной приемки, записанный производителем.",
    "real_generate_completed": (
        "Признак завершения вызова generate, записанный производителем."
    ),
}
app = FastAPI(title="Research Log Explorer")


class AnalyzeRequestGuard:
    def __init__(self, application):
        self.application = application
        self.gate = threading.BoundedSemaphore(2)

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != "/api/analyze"
        ):
            await self.application(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = 0
        if MAX_REQUEST is not None and declared > MAX_REQUEST:
            await self._respond(send, 413, b"Request body exceeds 26 MiB limit.")
            return
        # Atomic admission; excess requests fail fast without joining a queue.
        if not self.gate.acquire(blocking=False):
            await self._respond(send, 503, b"Service busy; retry later.")
            return
        count = 0
        oversized = False

        async def limited_receive():
            nonlocal count, oversized
            message = await receive()
            if message["type"] == "http.request":
                count += len(message.get("body", b""))
                if MAX_REQUEST is not None and count > MAX_REQUEST:
                    oversized = True
                    # Starlette closes spooled multipart files on this exception.
                    raise _BodyTooLarge("Request body exceeds 26 MiB limit.")
            return message

        sent = False

        async def guarded_send(message):
            nonlocal sent
            if oversized:
                return
            if message["type"] == "http.response.start":
                sent = True
            await send(message)

        try:
            await self.application(scope, limited_receive, guarded_send)
        except _BodyTooLarge:
            pass
        finally:
            self.gate.release()
        if oversized and not sent:
            await self._respond(send, 413, b"Request body exceeds 26 MiB limit.")

    @staticmethod
    async def _respond(send, status, body):
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"text/plain"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class _BodyTooLarge(MultiPartException):
    pass


class PreviewRecords(list):
    """Bound API record previews while counting every successfully parsed row."""

    total = 0

    def append(self, item):
        self.total += 1
        if len(self) < MAX_ROWS:
            super().append(item)


app.add_middleware(AnalyzeRequestGuard)


def _kind(name: str) -> str:
    return {
        ".log": "log",
        ".json": "json",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".csv": "csv",
    }.get(Path(name).suffix.lower(), "text")


def _flatten(value: Any, path: str = "", depth: int = 0):
    if depth > MAX_DEPTH:
        raise ValueError("JSON nesting exceeds parser limit.")
    if isinstance(value, dict):
        if not value:
            yield path or "$", value
        for key, child in value.items():
            yield from _flatten(child, f"{path}.{key}" if path else str(key), depth + 1)
    elif isinstance(value, list):
        if not value:
            yield path or "$", value
        for index, child in enumerate(value):
            yield from _flatten(child, f"{path}[{index}]", depth + 1)
    else:
        yield path or "$", value


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"Duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _json_loads(text):
    try:
        result = json.loads(
            text, object_pairs_hook=_pairs, parse_constant=_reject_constant
        )
        # Bound nesting and field expansion before response construction.
        for index, (_, value) in enumerate(_flatten(result)):
            if index >= MAX_FIELDS:
                raise ValueError("JSON exceeds field expansion limit.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Non-finite JSON number.")
        return result
    except (RecursionError, MemoryError) as exc:
        raise ValueError("JSON nesting exceeds parser limit.") from exc


def _parse_literal(text):
    # Rich soft wraps are joined; raw_message retains the original source.
    if len(text) > MAX_LITERAL:
        raise ValueError("Python metric exceeds literal size limit.")
    try:
        tree = ast.parse(" ".join(text.splitlines()), mode="eval")
    except (SyntaxError, MemoryError, RecursionError):
        return None
    pending: list[tuple[ast.AST, int]] = [(tree, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > 100_000 or depth > MAX_DEPTH:
            raise ValueError("Python metric exceeds complexity limit.")
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    try:
        result = ast.literal_eval(tree)
    except (ValueError, MemoryError, RecursionError):
        return None
    for _, value in _flatten(result):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite metric number.")
    return result


def _append(records, rec):
    data = rec.get("data")
    if isinstance(data, dict) and rec.get("kind") == "metric":
        metric = data.get("metric", "")
        explanation = FIELD_DESCRIPTIONS.get(
            metric, f"Метрика {metric} из журнала; семантика не определена схемой."
        )
    elif isinstance(data, dict) and data.get("event") == "experiment":
        explanation = (
            "Событие жизненного цикла эксперимента; finished не означает "
            "принятие датасета QC."
        )
    elif isinstance(data, dict) and data.get("event") == "dataset":
        explanation = (
            "Сообщение о выбранном запуске датасета, порядке выполнения и условии."
        )
    elif "UserWarning" in rec.get("message", ""):
        explanation = (
            "Предупреждение библиотеки из stdout/stderr; не является "
            "ошибкой QC само по себе."
        )
    else:
        explanation = "Исходная запись журнала или строка структурированных данных."
    rec["description"] = (
        f"{explanation} Источник: {rec['source']}, строка {rec['line']}."
    )
    records.append(rec)


def _parse_log(content, name, records, warnings):
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        match = LEVEL.match(line)
        if not match:
            _append(
                records,
                {
                    "id": f"{name}:{i + 1}",
                    "source": name,
                    "line": i + 1,
                    "level": None,
                    "message": line,
                    "raw_message": line,
                    "kind": "text",
                },
            )
            i += 1
            continue
        level, msg = match.group(1).upper(), match.group(2)
        start, rawmsg = i, msg
        data, kind = None, "text"
        if msg.startswith("metric ") and "=" in msg:
            key, value = msg[len("metric ") :].split("=", 1)
            fragments = [value]
            depth = (
                value.count("{")
                + value.count("[")
                - value.count("}")
                - value.count("]")
            )
            if "{" in value or "[" in value:
                depth = max(depth, 1) if depth > 0 else depth
            while depth > 0 and i + 1 < len(lines) and not LEVEL.match(lines[i + 1]):
                i += 1
                fragments.append(lines[i])
                depth += (
                    lines[i].count("{")
                    + lines[i].count("[")
                    - lines[i].count("}")
                    - lines[i].count("]")
                )
                if sum(map(len, fragments)) > MAX_LINE:
                    raise ValueError("Wrapped metric exceeds line limit.")
            rawmsg = msg + (
                "\n" + "\n".join(fragments[1:]) if len(fragments) > 1 else ""
            )
            text = " ".join(fragments)
            parsed = _parse_literal(text)
            data, kind = (
                {
                    "metric": key.strip(),
                    "value": parsed if parsed is not None else text,
                },
                "metric",
            )
        elif "=" in msg:
            left, value = msg.split("=", 1)
            event, _, suffix = left.strip().partition(" ")
            status_match = re.search(r"\bstatus=([^\s]+)", value)
            data = {"event": event, "value": value.strip()}
            if status_match:
                data["status"] = status_match.group(1)
            elif event == "experiment" and " status=" in value:
                data["status"] = value.rsplit(" status=", 1)[1].split()[0]
                data["value"] = value.rsplit(" status=", 1)[0]
            if suffix:
                data["details"] = suffix
            kind = "event"
        _append(
            records,
            {
                "id": f"{name}:{start + 1}",
                "source": name,
                "line": start + 1,
                "level": level,
                "message": rawmsg,
                "raw_message": rawmsg,
                "kind": kind,
                **({"data": data} if data is not None else {}),
            },
        )
        i += 1


def _number(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        if value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        try:
            number = float(value.strip())
        except ValueError:
            return value
        if not math.isfinite(number):
            raise ValueError("Non-finite CSV number.")
        if re.fullmatch(r"[+-]?\d+", value.strip()):
            return int(value.strip())
        return number
    return value


def _record_row(name, row, records, tables, line, kind):
    stem = Path(name).stem
    table = next((t for t in TABLES if stem == t or stem.endswith("_" + t)), None)
    if table:
        tables["_counts"][table] += 1
        if len(tables[table]) < MAX_PREVIEW_TABLE_ROWS:
            tables[table].append(
                {"source": name, **row}
                if isinstance(row, dict)
                else {"source": name, "value": row}
            )
    _append(
        records,
        {
            "id": f"{name}:{line}",
            "source": name,
            "line": line,
            "level": None,
            "message": f"{kind} row {line}",
            "raw_message": None,
            "kind": kind,
            "data": row,
        },
    )


def _parse_member(name, raw, records, tables, warnings, json_runs):
    if MAX_EXPANDED is not None and len(raw) > MAX_EXPANDED:
        raise ValueError("Member exceeds parser limit.")
    text = raw.decode("utf-8-sig", errors="replace")
    kind = _kind(name)
    if kind == "log":
        _parse_log(text, name, records, warnings)
    elif kind == "json":
        obj = _json_loads(text) if text.strip() else None
        if obj is not None:
            if (
                Path(name).stem == "runs"
                and isinstance(obj, dict)
                and isinstance(obj.get("runs"), list)
            ):
                obj = obj["runs"]
            if isinstance(obj, list) and Path(name).stem in TABLES:
                for n, row in enumerate(obj, 1):
                    _record_row(name, row, records, tables, n, "json")
                if Path(name).stem == "runs":
                    json_runs.extend(obj)
            elif Path(name).stem == "runs" and isinstance(obj, dict):
                json_runs.append(obj)
                _record_row(name, obj, records, tables, 1, "json")
            else:
                _record_row(name, obj, records, tables, 1, "json")
    elif kind == "jsonl":
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                _record_row(name, _json_loads(line), records, tables, n, "json")
    elif kind == "csv":
        reader = csv.DictReader(io.StringIO(text))
        for n, row in enumerate(reader, 2):
            clean = {
                key: value
                if key and (key.endswith("_id") or key.endswith("_hash"))
                else _number(value)
                for key, value in row.items()
            }
            _record_row(name, clean, records, tables, n, "csv")


def _parse_jsonl_blocks(name, blocks, records, tables):
    pending = bytearray()
    row = 0
    for block in blocks:
        pending.extend(block)
        while (newline := pending.find(b"\n")) >= 0:
            raw_line = bytes(pending[:newline])
            del pending[: newline + 1]
            row += 1
            if len(raw_line) > MAX_LINE:
                raise ValueError("JSONL line exceeds parser limit.")
            line = raw_line.decode(
                "utf-8-sig" if row == 1 else "utf-8", errors="replace"
            )
            if line.strip():
                _record_row(name, _json_loads(line), records, tables, row, "json")
        if len(pending) > MAX_LINE:
            raise ValueError("JSONL line exceeds parser limit.")
    if pending:
        row += 1
        line = bytes(pending).decode(
            "utf-8-sig" if row == 1 else "utf-8", errors="replace"
        )
        if line.strip():
            _record_row(name, _json_loads(line), records, tables, row, "json")


def _is_directory(info, is_rar):
    return info.is_dir() or (is_rar and info.filename.endswith("/"))


def _analyze(blob, filename):
    with tempfile.NamedTemporaryFile(
        prefix="log-explorer-", suffix=".archive"
    ) as stream:
        stream.write(blob)
        stream.flush()
        return _analyze_path(Path(stream.name), filename)


def _analyze_path(path, filename):
    with open(path, "rb") as source:
        magic = source.read(8)
    is_zip = magic.startswith(b"PK\x03\x04") or magic.startswith(b"PK\x05\x06")
    is_rar = magic.startswith(b"Rar!\x1a\x07")
    if not (is_zip or is_rar):
        raise ValueError("Unsupported archive format (expected ZIP or RAR).")
    if is_rar:
        try:
            import rarfile

            archive = rarfile.RarFile(path, errors="strict")
        except Exception as exc:
            raise ValueError(f"Invalid or unsupported RAR archive: {exc}") from exc
    else:
        try:
            archive = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, OSError) as exc:
            raise ValueError("Invalid ZIP archive.") from exc
    with archive as zf:
        infos = zf.infolist()
        seen, total = set(), 0
        for info in infos:
            member_path = PurePosixPath(info.filename)
            mode = (
                (getattr(info, "external_attr", 0) or 0) >> 16
                if is_zip
                else (getattr(info, "mode", 0) or 0)
            )
            normalized = str(member_path).casefold()
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or "\\" in info.filename
                or normalized in seen
                or stat.S_ISLNK(mode)
                or (is_rar and (info.is_symlink() or getattr(info, "file_redir", None)))
            ):
                raise ValueError("Unsafe or duplicate ZIP path.")
            seen.add(normalized)
            if is_zip and info.flag_bits & 1:
                raise ValueError("Encrypted ZIP members are unsupported.")
            if is_rar and info.needs_password():
                raise ValueError("Encrypted RAR members are unsupported.")
            if is_zip and info.compress_type not in (
                zipfile.ZIP_STORED,
                zipfile.ZIP_DEFLATED,
            ):
                raise ValueError("Unsupported ZIP compression method.")
            total += info.file_size
            if info.file_size and (
                not info.compress_size
                or info.file_size / info.compress_size > MAX_RATIO
            ):
                raise ValueError("Archive compression ratio exceeds limit.")
        files, records, warnings, json_runs = [], PreviewRecords(), [], []
        tables = {table: [] for table in TABLES}
        tables["_counts"] = {table: 0 for table in TABLES}

        def consume(name, size, blocks):
            before = records.total
            if _kind(name) == "jsonl":
                _parse_jsonl_blocks(name, blocks, records, tables)
            else:
                raw = b"".join(blocks)
                if len(raw) != size:
                    raise ValueError(f"Short archive member: {name}")
                _parse_member(name, raw, records, tables, warnings, json_runs)
            files.append(
                {
                    "name": name,
                    "size": size,
                    "kind": _kind(name),
                    "records": records.total - before,
                }
            )

        if is_rar:
            import libarchive

            with libarchive.file_reader(os.fspath(path)) as entries:
                info_iter = (i for i in infos if not _is_directory(i, is_rar))
                for entry in entries:
                    if entry.filetype == stat.S_IFDIR or entry.pathname.endswith("/"):
                        continue
                    try:
                        info = next(info_iter)
                    except StopIteration as exc:
                        raise ValueError(
                            "RAR member index does not match stream."
                        ) from exc
                    if entry.pathname != info.filename or entry.size != info.file_size:
                        raise ValueError(f"RAR stream/member mismatch: {info.filename}")
                    consume(info.filename, info.file_size, entry.get_blocks())
                if next(info_iter, None) is not None:
                    raise ValueError("RAR stream omitted archive members.")
        else:
            for info in infos:
                if info.is_dir():
                    continue
                before = records.total
                raw = zf.read(info)
                try:
                    _parse_member(
                        info.filename, raw, records, tables, warnings, json_runs
                    )
                except (
                    ValueError,
                    json.JSONDecodeError,
                    csv.Error,
                    UnicodeError,
                    zipfile.BadZipFile,
                ) as exc:
                    raise ValueError(f"Could not parse {info.filename}: {exc}") from exc
                files.append(
                    {
                        "name": info.filename,
                        "size": info.file_size,
                        "kind": _kind(info.filename),
                        "records": records.total - before,
                    }
                )
    metrics, runs = {}, list(json_runs)
    experiment = status = None
    for rec in records:
        data = rec.get("data")
        if rec["kind"] == "event" and isinstance(data, dict):
            if data.get("event") == "experiment":
                experiment = str(data.get("value", "")).split(" status=", 1)[0]
                status = data.get("status") or status
            if data.get("event") == "status":
                status = data.get("value")
        if rec["kind"] == "metric" and isinstance(data, dict):
            key, value = data.get("metric"), data.get("value")
            if key == "runs" and isinstance(value, list):
                runs.extend(value)
                for run_index, run in enumerate(value, 1):
                    if (
                        isinstance(run, dict)
                        and len(tables["runs"]) < MAX_PREVIEW_TABLE_ROWS
                    ):
                        tables["runs"].append(
                            {"source": rec["source"], "row": run_index, **run}
                        )
            metrics[key] = value
    attempts = runs
    unique_runs = {}
    for run in attempts:
        if isinstance(run, dict):
            identity = run.get("run_id") or run.get("attempt_id")
            if identity is not None:
                unique_runs.setdefault(str(identity), run)
    runs = list(unique_runs.values()) if unique_runs else attempts
    fields = []
    for ri, rec in enumerate(records):
        if "data" in rec:
            for path, value in _flatten(rec["data"]):
                if len(fields) >= MAX_FIELDS:
                    raise ValueError("Archive exceeds field expansion limit.")
                leaf = path.rsplit(".", 1)[-1].split("[")[0]
                if path == "value" and rec.get("kind") == "metric":
                    leaf = rec["data"].get("metric", leaf)
                desc = FIELD_DESCRIPTIONS.get(
                    leaf,
                    (
                        f"{leaf}: исходное значение {rec['source']}; "
                        "единица измерения не задана схемой."
                    ),
                )
                if ".engine_source_hashes." in path:
                    desc = (
                        "SHA-256 исходного файла движка; проверяемая запись "
                        "происхождения, не веса модели."
                    )
                fields.append(
                    {
                        "path": f"records[{ri}].data.{path}",
                        "value": value,
                        "type": type(value).__name__,
                        "description": desc,
                        "source": rec["source"],
                    }
                )
    accepted = sum(
        isinstance(run, dict) and run.get("primary_status") == "accepted"
        for run in runs
    )
    table_counts = tables.pop("_counts")
    tables["_meta"] = {
        "counts": table_counts,
        "preview_limit": MAX_PREVIEW_TABLE_ROWS,
        "preview_rows": {key: len(tables[key]) for key in TABLES},
        "is_preview": any(table_counts[key] > len(tables[key]) for key in TABLES),
    }
    return {
        "archive": {
            "name": filename,
            "format": "rar" if is_rar else "zip",
            "files": files,
            "warnings": warnings,
        },
        "metrics": metrics,
        "runs": runs,
        "records": records,
        "fields": fields,
        "tables": tables,
        "summary": {
            "experiment": experiment,
            "status": status,
            "run_count": len(runs),
            "attempt_count": len(attempts),
            "accepted_count": accepted,
            "record_count": records.total,
            "record_preview_count": len(records),
            "record_preview_limit": MAX_ROWS,
            "records_are_preview": records.total > len(records),
            "warnings": warnings,
        },
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):  # noqa: B008 - FastAPI request declaration
    if not PARSE_GATE.acquire(blocking=False):
        raise HTTPException(503, "Service busy; retry later.")
    try:
        with tempfile.NamedTemporaryFile(
            prefix="upload-", suffix=".archive", delete=False
        ) as target:
            spool_path = Path(target.name)
            await file.seek(0)
            while chunk := await file.read(1024 * 1024):
                target.write(chunk)
        import anyio

        return await anyio.to_thread.run_sync(
            _analyze_path, spool_path, Path(file.filename or "upload.zip").name
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            400,
            str(exc) if isinstance(exc, ValueError) else "Could not process archive.",
        ) from exc
    finally:
        if "spool_path" in locals():
            spool_path.unlink(missing_ok=True)
        PARSE_GATE.release()
        await file.close()


@app.get("/api/example")
def example():
    try:
        return _analyze((BASE / "sample.zip").read_bytes(), "sample.zip")
    except Exception as exc:
        raise HTTPException(500, "Bundled example unavailable.") from exc


if FRONTEND.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")

    @app.api_route("/", methods=["GET", "HEAD"])
    def index():
        return FileResponse(FRONTEND / "index.html")
