"""Физические компоненты BLIS для nano-vLLM. Без обучения коэффициентов."""

import argparse
import csv
import io
import json
from data_source import open_dataset
from collections import defaultdict
from pathlib import Path


def jsonl(archive, filename):
    with archive.open(filename) as stream:
        for line in io.TextIOWrapper(stream, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def build(zip_path, out_dir, peak_flops, hbm_bandwidth, net_bandwidth):
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "step_id",
        "phase",
        "batch_size",
        "measured_seconds",
        "t_prefill_compute",
        "t_prefill_kv",
        "t_decode_compute",
        "t_decode_kv",
        "t_tp",
        "tp_communication_bytes",
    ]
    counts = defaultdict(int)
    with (
        open_dataset(zip_path) as archive,
        (out_dir / "blis_features.csv").open("w", newline="", encoding="utf-8") as file,
    ):
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for filename in sorted(
            n for n in archive.namelist() if n.endswith("/runs.json")
        ):
            base = filename.rsplit("/", 1)[0]
            run = json.loads(archive.read(filename))
            model = json.loads(archive.read(base + "/model.json"))
            tp = int(
                run["effective_config"]["effective_config"]["tensor_parallel_size"]
            )
            layers = int(model["layer_count"])
            hidden = int(model["hidden_size"])
            intermediate = int(model["intermediate_size"])
            q_width = int(model["attention_head_count"]) * int(model["head_dim"])
            kv_width = int(model["key_value_head_count"]) * int(model["head_dim"])
            dtype = str(model["weight_dtype"]).lower()
            if "bfloat16" in dtype or "float16" in dtype or "bf16" in dtype:
                value_bytes = 2
            elif "float32" in dtype or "fp32" in dtype:
                value_bytes = 4
            else:
                raise ValueError(f"Неизвестная точность весов: {dtype}")
            # QKV, O, gate/up/down проекции; FLOPs за один новый токен по всей модели.
            linear_per_token = (
                2
                * layers
                * (
                    hidden * (q_width + 2 * kv_width)
                    + hidden * q_width
                    + 3 * hidden * intermediate
                )
            )
            kv_per_token = 2 * layers * kv_width * value_bytes
            by_step = defaultdict(list)
            for request in jsonl(archive, base + "/step_requests.jsonl"):
                by_step[request["step_id"]].append(request)
            for step in jsonl(archive, base + "/steps.jsonl"):
                requests = by_step[step["step_id"]]
                batch = int(step["actual_batch_size"])
                if len(requests) != batch:
                    raise ValueError(
                        f"Неверное число запросов: {run['run_id']}/{step['step_id']}"
                    )
                phase = step["phase"]
                pf_flops = dc_flops = pf_kv = dc_kv = 0.0
                if phase == "prefill":
                    fresh = sum(int(r["fresh_prefill_tokens"]) for r in requests)
                    pairs = 0
                    past = 0
                    for r in requests:
                        q = int(r["fresh_prefill_tokens"])
                        c = int(r.get("prefix_reused_tokens", 0)) + int(
                            r.get("private_kv_cached_tokens", 0)
                        )
                        pairs += q * c + q * (q + 1) / 2
                        past += c
                    pf_flops = (
                        linear_per_token * fresh + 4 * layers * q_width * pairs
                    ) / tp
                    # Оценка HBM-трафика: запись нового KV и чтение ранее созданного контекста.
                    pf_kv = kv_per_token * (fresh + past) / tp
                    token_count = fresh
                elif phase == "decode":
                    context = sum(int(r["context_tokens_before"]) for r in requests)
                    dc_flops = (
                        linear_per_token * batch + 4 * layers * q_width * context
                    ) / tp
                    dc_kv = kv_per_token * (context + batch) / tp
                    token_count = batch
                else:
                    raise ValueError(f"Неожиданная фаза: {phase}")
                # ОЦЕНКА TP: две all-reduce активаций на слой (O-проекция, MLP),
                # ring all-reduce: 2*(TP-1)/TP переданных байтов на одну GPU.
                # Точная реализация nano-vLLM/топология может отличаться.
                tp_bytes = (
                    2
                    * layers
                    * hidden
                    * value_bytes
                    * token_count
                    * (2 * (tp - 1) / tp)
                )
                begin, end = step["step_host_start_ns"], step["step_host_end_ns"]
                if end <= begin:
                    raise ValueError("Некорректная длительность шага")
                writer.writerow(
                    {
                        "run_id": run["run_id"],
                        "step_id": step["step_id"],
                        "phase": phase,
                        "batch_size": batch,
                        "measured_seconds": (end - begin) / 1e9,
                        "t_prefill_compute": pf_flops / peak_flops,
                        "t_prefill_kv": pf_kv / hbm_bandwidth,
                        "t_decode_compute": dc_flops / peak_flops,
                        "t_decode_kv": dc_kv / hbm_bandwidth,
                        "t_tp": tp_bytes / net_bandwidth,
                        "tp_communication_bytes": tp_bytes,
                    }
                )
                counts[phase] += 1
    if not sum(counts.values()):
        raise ValueError("Прогоны не найдены в архиве")
    print("Обработано шагов:", sum(counts.values()), "; по фазам:", dict(counts))
    print("Признаки сохранены:", out_dir / "blis_features.csv")
    print(
        "Внимание: t_tp — аналитическая оценка; пропускная способность связи задана вручную."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Расчёт физических признаков BLIS")
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs_v2"))
    parser.add_argument(
        "--peak-flops",
        type=float,
        default=76.7e12,
        help="Пиковая BF16 производительность одной GPU, FLOP/с",
    )
    parser.add_argument(
        "--hbm-bandwidth",
        type=float,
        default=448e9,
        help="Полоса памяти одной GPU, байт/с",
    )
    parser.add_argument(
        "--net-bandwidth",
        type=float,
        required=True,
        help="Проверенная или сценарная скорость меж-GPU связи, байт/с",
    )
    args = parser.parse_args()
    if min(args.peak_flops, args.hbm_bandwidth, args.net_bandwidth) <= 0:
        parser.error("Все пропускные способности должны быть положительными")
    build(
        args.zip_path, args.out, args.peak_flops, args.hbm_bandwidth, args.net_bandwidth
    )
