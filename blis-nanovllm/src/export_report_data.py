"""Экспорт компактных данных для отчёта BLIS без токенов текста и больших логов.
Запуск: python export_report_data.py data/ds_qwen_3_8b/runs --out report_8b.zip
"""

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

RUN_FIELDS = [
    "run_id",
    "attempt_id",
    "status",
    "split_group_id",
    "seed",
    "prefix_cache_mode",
    "scheduler_mode",
    "wall_seconds",
    "request_count",
    "input_tokens",
    "output_tokens",
    "prefix_reused_tokens",
    "cache_hit_ratio",
    "tensor_parallel_size",
    "gpu_name",
]
REQ_FIELDS = [
    "run_id",
    "request_id",
    "prompt_length",
    "input_token_count",
    "requested_output_tokens",
    "output_token_count",
    "actual_output_tokens",
    "prefix_reused_tokens",
    "cache_hit_ratio",
    "enqueue_monotonic_ns",
    "first_scheduled_monotonic_ns",
    "first_generated_monotonic_ns",
    "first_host_observed_monotonic_ns",
    "last_generated_monotonic_ns",
    "finish_monotonic_ns",
    "status",
]
STEP_FIELDS = [
    "run_id",
    "step_id",
    "step_index",
    "phase",
    "actual_batch_size",
    "scheduled_tokens",
    "fresh_prefill_tokens",
    "recompute_tokens",
    "prefix_reused_tokens",
    "same_request_cached_tokens",
    "produced_tokens",
    "step_host_start_ns",
    "step_host_end_ns",
    "modelcall_start_ns",
    "modelcall_end_ns",
    "schedule_start_ns",
    "schedule_end_ns",
]


def iter_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def find_attempts(root):
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("R"):
            continue
        candidates = [folder] + sorted(p for p in folder.iterdir() if p.is_dir())
        good = [
            p
            for p in candidates
            if all(
                (p / n).is_file()
                for n in ("runs.json", "steps.jsonl", "requests.jsonl")
            )
        ]
        if not good:
            raise ValueError(f"Нет runs.json/steps.jsonl/requests.jsonl в {folder}")
        if len(good) > 1:
            raise ValueError(f"Несколько попыток в {folder}: выбери успешную вручную")
        yield folder.name, good[0]


def main():
    ap = argparse.ArgumentParser(
        description="Собрать компактные входные данные для отчёта BLIS"
    )
    ap.add_argument("runs", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.runs.is_dir():
        ap.error(f"Папка не существует: {args.runs}")
    rows = {name: [] for name in ["runs_summary", "requests_summary", "steps_summary"]}
    checked = set()
    for rid, dirpath in find_attempts(args.runs):
        run = json.loads((dirpath / "runs.json").read_text(encoding="utf-8"))
        if rid in checked:
            raise ValueError(f"Повторный run_id: {rid}")
        checked.add(rid)
        cfg = run.get("effective_config", {})
        inner = cfg.get("effective_config", {})
        hw = {}
        if (dirpath / "hardware.json").is_file():
            hw = json.loads((dirpath / "hardware.json").read_text(encoding="utf-8"))
        devs = hw.get("gpu_devices") or []
        gpu = devs[0].get("name") if devs else None
        meta = {
            "run_id": rid,
            "attempt_id": run.get("attempt_id"),
            "status": run.get("status"),
            "split_group_id": run.get("split_group_id"),
            "seed": run.get("seed"),
            "prefix_cache_mode": run.get("prefix_cache_mode")
            or cfg.get("prefix_cache_mode"),
            "scheduler_mode": run.get("scheduler_mode") or cfg.get("scheduler_mode"),
            "tensor_parallel_size": inner.get("tensor_parallel_size"),
            "gpu_name": gpu,
        }
        meta.update(
            {
                key: run.get("metrics", {}).get(key)
                for key in RUN_FIELDS
                if key
                in [
                    "wall_seconds",
                    "request_count",
                    "input_tokens",
                    "output_tokens",
                    "prefix_reused_tokens",
                    "cache_hit_ratio",
                ]
            }
        )
        rows["runs_summary"].append({k: meta.get(k) for k in RUN_FIELDS})
        for r in iter_jsonl(dirpath / "requests.jsonl"):
            rows["requests_summary"].append(
                {"run_id": rid, **{k: r.get(k) for k in REQ_FIELDS if k != "run_id"}}
            )
        for s in iter_jsonl(dirpath / "steps.jsonl"):
            rows["steps_summary"].append(
                {"run_id": rid, **{k: s.get(k) for k in STEP_FIELDS if k != "run_id"}}
            )
    if not checked:
        raise ValueError("Не найдены прогоны")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7
    ) as z:
        for name, fields in [
            ("runs_summary", RUN_FIELDS),
            ("requests_summary", REQ_FIELDS),
            ("steps_summary", STEP_FIELDS),
        ]:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[name])
            z.writestr(name + ".csv", buf.getvalue())
    print(
        f"Прогонов: {len(checked)}, запросов: {len(rows['requests_summary'])}, шагов: {len(rows['steps_summary'])}"
    )
    print(f"Архив сохранён: {args.out}")


if __name__ == "__main__":
    main()
