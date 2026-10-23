"""Задает воспроизводимые нагрузки nano-vLLM и матрицу экспериментальных запусков.

Модуль объявляет профили размера запросов S/L/M и 24 сочетания режима кэша,
планировщика и повторного seed. Он создает перемешанную по seed матрицу со
стабильными идентификаторами запуска/условия/пары, генерирует повторяемые
запросы с целочисленными ID токенов и сводные объемы, а также моделирует
ограничение upstream-поиска блоков общего префикса. Модель и движок инференса
не загружаются; сформированные запросы можно неизменно повторно использовать
для сравнения экспериментальных режимов.
"""

from __future__ import annotations

import random
from typing import Any

WORKLOADS = {
    "S": {
        "short_tokens": 128,
        "long_tokens": 128,
        "short_count": 32,
        "prefix_tokens": 0,
        "output_tokens": 128,
    },
    "L": {
        "short_tokens": 2048,
        "long_tokens": 2048,
        "short_count": 0,
        "prefix_tokens": 0,
        "output_tokens": 512,
    },
    "M": {
        "short_tokens": 128,
        "long_tokens": 2048,
        "short_count": 16,
        "prefix_tokens": 1024,
        "output_tokens": 512,
    },
}
CONDITIONS = (
    ("S", "on", "upstream_dynamic"),
    ("S", "on", "fixed_batch"),
    ("S", "off", "upstream_dynamic"),
    ("S", "off", "fixed_batch"),
    ("L", "on", "upstream_dynamic"),
    ("L", "on", "fixed_batch"),
    ("L", "off", "upstream_dynamic"),
    ("L", "off", "fixed_batch"),
    ("M", "on", "upstream_dynamic"),
    ("M", "on", "fixed_batch"),
    ("M", "off", "upstream_dynamic"),
    ("M", "off", "fixed_batch"),
)


def build_run_matrix(order_seed: int = 20261023) -> list[dict[str, Any]]:
    """Return the contract's fixed 24-run grid, with seeded order independent IDs."""
    rows: list[dict[str, Any]] = []
    for condition_index, (workload, prefix, scheduler) in enumerate(CONDITIONS, 1):
        for repeat_index, seed in enumerate((42, 43), 1):
            number = (condition_index - 1) * 2 + repeat_index
            rows.append(
                {
                    "run_id": f"R{number:02}",
                    "condition_id": f"C{condition_index:02}",
                    "pair_id": f"{workload}-seed-{seed}",
                    "split_group_id": f"{workload}-family",
                    "repeat_id": f"{repeat_index:02}",
                    "seed": seed,
                    "workload_id": workload,
                    "prefix_cache_mode": prefix,
                    "scheduler_mode": scheduler,
                }
            )
    rng = random.Random(order_seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    rank = {index: position + 1 for position, index in enumerate(order)}
    for index, row in enumerate(rows):
        row["run_order"] = rank[index]
    return sorted(rows, key=lambda row: row["run_order"])


def build_workload(name: str, seed: int) -> dict[str, Any]:
    """Create token-ID requests once per workload seed; modes reuse this artifact."""
    if name not in WORKLOADS:
        raise ValueError(
            f"unknown workload {name!r}; expected one of {tuple(WORKLOADS)}"
        )
    spec = WORKLOADS[name]
    rng = random.Random(seed)
    shared_prefix = [rng.randrange(1000, 50000) for _ in range(spec["prefix_tokens"])]
    requests = []
    for index in range(32):
        is_short = name == "S" or (name == "M" and index % 2 == 0)
        klass = "short" if is_short else "long"
        length = spec["short_tokens"] if is_short else spec["long_tokens"]
        suffix_length = length - len(shared_prefix) if not is_short else length
        suffix = [rng.randrange(1000, 50000) for _ in range(suffix_length)]
        input_ids = shared_prefix + suffix if klass == "long" else suffix
        requests.append(
            {
                "request_id": f"{name}-S{seed}-Q{index + 1:02}",
                "arrival_index": index,
                "planned_arrival_offset_ns": 0,
                "workload_class": klass,
                "prefix_family_id": f"{name}-shared-long"
                if name == "M" and klass == "long"
                else None,
                "prefix_group_id": "M-long-1024"
                if name == "M" and klass == "long"
                else None,
                "prefix_token_count": len(shared_prefix) if klass == "long" else 0,
                "input_token_ids": input_ids,
                "requested_output_tokens": 512
                if name == "L" or klass == "long"
                else 128,
                "temperature": 0.6,
                "ignore_eos": True,
            }
        )
    return {
        "workload_id": f"{name}-seed-{seed}",
        "workload_name": name,
        "seed": seed,
        "requests": requests,
        "summary": {
            "request_count": len(requests),
            "short_count": sum(r["workload_class"] == "short" for r in requests),
            "long_count": sum(r["workload_class"] == "long" for r in requests),
            "input_token_counts": [len(r["input_token_ids"]) for r in requests],
            **(
                {"requested_output_tokens_each": spec["output_tokens"]}
                if name != "M"
                else {}
            ),
            "requested_output_token_counts": [
                request["requested_output_tokens"] for request in requests
            ],
            "logical_prompt_tokens": sum(len(r["input_token_ids"]) for r in requests),
            "requested_output_token_total": sum(
                r["requested_output_tokens"] for r in requests
            ),
            "token_ids_reused_across_modes": True,
        },
    }


def validate_shared_prefix_boundary(
    block_size: int, prefix_tokens: int, sequence_tokens: int | None = None
) -> dict[str, Any]:
    """Model upstream lookup range(ceil(sequence/block)-1), bounded by shared prefix."""
    sequence_tokens = sequence_tokens if sequence_tokens is not None else prefix_tokens
    if block_size <= 0 or prefix_tokens < 0 or sequence_tokens < prefix_tokens:
        raise ValueError("invalid block, prefix, or sequence token count")
    eligible_sequence_blocks = max(
        0, (sequence_tokens + block_size - 1) // block_size - 1
    )
    eligible = min(prefix_tokens // block_size, eligible_sequence_blocks)
    return {
        "eligible_blocks": eligible,
        "excluded_tail_tokens": max(0, prefix_tokens - eligible * block_size),
        "reason": "upstream_lookup_excludes_last_full_block",
    }
