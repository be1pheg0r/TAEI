"""Paired, offline request-count and prefix-cache scaling pilot."""

from __future__ import annotations

import random
from typing import Any

PROTOCOL = "qwen3_8b_request_scaling_v1"
NAME = "qwen3_8b_request_scaling"
COUNTS = (1, 2, 4, 8, 16)
SEEDS = (42, 43, 44)
LENGTHS = tuple(3000 + (10000 * i + 7) // 15 for i in range(16))
PREFIX = 1024
OUTPUT = 1000


def build_scaling_workload(name: str, seed: int) -> dict[str, Any]:
    """Reconstruct a nested workload from the count encoded in its name."""
    count = int(name.removeprefix(NAME + "_n"))
    if name != f"{NAME}_n{count}" or count not in COUNTS:
        raise ValueError(f"unknown scaling workload: {name}")
    rng = random.Random(seed)
    lengths = list(LENGTHS)
    rng.shuffle(lengths)
    requests = []
    for index, length in enumerate(lengths[:count]):
        ids = [rng.randrange(1000, 90000) for _ in range(length)]
        family = f"scaling-s{seed}-r{index:02}"
        requests.append(
            {
                "request_id": family,
                "arrival_index": index,
                "planned_arrival_offset_ns": 0,
                "workload_class": "hit",
                "cache_intent": "hit",
                "prompt_length": length,
                "prefix_family_id": family,
                "prefix_group_id": family,
                "prefix_token_count": PREFIX,
                "expected_cache_hit_tokens_on": PREFIX,
                "target_cache_hit_ratio_on": PREFIX / length,
                "prime_input_token_ids": ids[:PREFIX] + [90001],
                "input_token_ids": ids,
                "requested_output_tokens": OUTPUT,
                "temperature": 0.6,
                "ignore_eos": True,
            }
        )
    return {
        "workload_id": f"{name}-seed-{seed}",
        "workload_name": NAME,
        "set_id": NAME,
        "seed": seed,
        "requests": requests,
        "summary": {
            "request_count": count,
            "prompt_lengths": lengths[:count],
            "input_token_counts": lengths[:count],
            "requested_output_tokens_each": OUTPUT,
            "requested_output_token_total": count * OUTPUT,
            "logical_prompt_tokens": sum(lengths[:count]),
            "cache_hit_targets": count,
            "cache_miss_targets": 0,
            "cache_hit_tokens_on_targets": [PREFIX] * count,
            "priming_requests_excluded_from_measurement": True,
        },
    }


def build_scaling_plan(
    order_seed: int, context_cap: int, max_num_seqs: int, max_num_batched_tokens: int
) -> dict[str, Any]:
    """Construct independent randomized blocks with paired cache treatments."""
    if context_cap < 14000:
        raise ValueError("configured max_model_len is below workload requirement 14000")
    if max_num_seqs < 1 or max_num_batched_tokens < 1:
        raise ValueError("engine sequence and batched-token limits must be positive")
    rng = random.Random(order_seed)
    matrix = []
    summaries = {}
    prompt_total = 0
    for repeat, seed in enumerate(SEEDS, 1):
        block = []
        for count in COUNTS:
            name = f"{NAME}_n{count}"
            workload = build_scaling_workload(name, seed)
            summaries[workload["workload_id"]] = workload["summary"]
            for cache in ("on", "off"):
                block.append(
                    {
                        "run_id": f"S{repeat}-{count:02}-{cache}",
                        "condition_id": f"n{count}-{cache}",
                        "set_id": NAME,
                        "pair_id": f"scaling-s{seed}-n{count}",
                        "split_group_id": f"scaling-s{seed}",
                        "repeat_id": f"{repeat:02}",
                        "seed": seed,
                        "workload_id": name,
                        "request_count": count,
                        "prefix_cache_mode": cache,
                        "scheduler_mode": "upstream_dynamic",
                    }
                )
                prompt_total += workload["summary"]["logical_prompt_tokens"]
        rng.shuffle(block)
        for row in block:
            row["run_order"] = len(matrix) + 1
            matrix.append(row)
    return {
        "matrix": matrix,
        "run_count": len(matrix),
        "condition_count": len(COUNTS) * 2,
        "request_count": sum(row["request_count"] for row in matrix),
        "workload_protocol": PROTOCOL,
        "workload_summaries": summaries,
        "seeds": list(SEEDS),
        "logical_prompt_tokens": prompt_total,
        "requested_output_tokens": sum(row["request_count"] for row in matrix) * OUTPUT,
        "validation_status": "disabled",
        "schedule": {
            "prompt_lengths": list(LENGTHS),
            "request_counts": list(COUNTS),
            "requested_output_tokens_each": OUTPUT,
            "prefix_tokens": PREFIX,
            "donor_requests_on": sum(COUNTS) * len(SEEDS),
            "donor_requests_off": 0,
            "warmup_requests_per_launch": 1,
            "max_prompt_plus_output_tokens": 14000,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "order_seed": order_seed,
        },
    }
