"""Deterministic independent paired workloads for cache experiments."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

WORKLOAD_NAME = "prefix_sweep"
WORKLOAD_SETS = tuple(f"prefix_sweep_set_{i:02d}" for i in range(1, 7))
BLOCK_SIZE = 256
OUTPUT_TOKEN_CHOICES = (16, 32, 64, 128, 256)
SEEDS = (42, 43)
CONDITIONS = tuple(
    (cache, scheduler)
    for cache in ("on", "off")
    for scheduler in ("upstream_dynamic", "fixed_batch")
)


def build_run_matrix(order_seed: int = 20261023) -> list[dict[str, Any]]:
    """Build 48 stable set/seed/treatment launches in seeded shuffled order."""
    rows: list[dict[str, Any]] = []
    number = 0
    for set_index, workload_name in enumerate(WORKLOAD_SETS, 1):
        for seed_index, seed in enumerate(SEEDS, 1):
            for condition_index, (cache, scheduler) in enumerate(CONDITIONS, 1):
                number += 1
                rows.append({
                    "run_id": f"R{number:02}",
                    "condition_id": f"C{condition_index:02}",
                    "set_id": workload_name,
                    "pair_id": f"{workload_name}-seed-{seed}",
                    "split_group_id": f"{workload_name}-family",
                    "repeat_id": f"{seed_index:02}",
                    "seed": seed,
                    "workload_id": workload_name,
                    "prefix_cache_mode": cache,
                    "scheduler_mode": scheduler,
                })
    order = list(range(len(rows)))
    random.Random(order_seed).shuffle(order)
    for rank, index in enumerate(order, 1):
        rows[index]["run_order"] = rank
    return sorted(rows, key=lambda row: row["run_order"])


def _set_index(name: str) -> int:
    if name == WORKLOAD_NAME:  # compatibility for the legacy overhead workload
        return 1
    try:
        return WORKLOAD_SETS.index(name) + 1
    except ValueError:
        return 1


def build_workload(name: str, seed: int) -> dict[str, Any]:
    """Build 8 paired hit/miss requests with varied lengths and reuse blocks."""
    set_index = _set_index(name)
    workload_name = WORKLOAD_SETS[set_index - 1]
    rng = random.Random((seed << 16) + set_index)
    lengths: list[int] = []
    # Eight disjoint strata tile 512..4096; selected values are always unaligned.
    for i in range(8):
        low = 512 + i * 448
        high = 512 + (i + 1) * 448
        length = rng.randrange(low, high)
        if length % BLOCK_SIZE == 0:
            length += 1
        lengths.append(length)
    requests = []
    prefix_rng = random.Random(f"prefix-blocks:{set_index}:{seed}")
    used_blocks: set[int] = set()
    used_ratios: set[Fraction] = set()
    for i, length in enumerate(lengths):
        # Vary reuse independently of the length stratum, while choosing only
        # eligible full blocks and keeping positive counts/ratios distinct.
        prefix_blocks = prefix_rng.choice([
            count
            for count in range(1, length // BLOCK_SIZE + 1)
            if count not in used_blocks
            and Fraction(count * BLOCK_SIZE, length) not in used_ratios
        ])
        used_blocks.add(prefix_blocks)
        used_ratios.add(Fraction(prefix_blocks * BLOCK_SIZE, length))
        output_tokens = rng.choice(OUTPUT_TOKEN_CHOICES)
        for intent in ("hit", "miss"):
            low, high = (1000, 50000) if intent == "hit" else (50001, 90000)
            input_ids = [rng.randrange(low, high) for _ in range(length)]
            # Stay inside the same ordinary-token namespace as the remaining
            # IDs; set/pair identity must not become an out-of-vocabulary ID.
            input_ids[0] = low + set_index * 16 + i
            family = f"{workload_name}-seed-{seed}-pair-{i + 1:02}" if intent == "hit" else None
            prefix_tokens = prefix_blocks * BLOCK_SIZE if intent == "hit" else 0
            donor = input_ids[:prefix_tokens] + [90001] if prefix_tokens else []
            requests.append({
                "request_id": f"{workload_name}-S{seed}-P{i + 1:02}-{intent}",
                "arrival_index": len(requests),
                "planned_arrival_offset_ns": 0,
                "workload_class": intent,
                "cache_intent": intent,
                "prompt_length": length,
                "prefix_family_id": family,
                "prefix_group_id": family,
                "prefix_token_count": prefix_tokens,
                "expected_cache_hit_tokens_on": prefix_tokens,
                "target_cache_hit_ratio_on": prefix_tokens / length,
                "prime_input_token_ids": donor,
                "input_token_ids": input_ids,
                "requested_output_tokens": output_tokens,
                "temperature": 0.6,
                "ignore_eos": True,
            })
    inputs = [len(r["input_token_ids"]) for r in requests]
    outputs = [r["requested_output_tokens"] for r in requests]
    return {
        "workload_id": f"{workload_name}-seed-{seed}",
        "workload_name": workload_name,
        "set_id": workload_name,
        "seed": seed,
        "requests": requests,
        "summary": {
            "request_count": len(requests),
            "prompt_lengths": lengths,
            "requests_per_length": 2,
            "input_token_counts": inputs,
            "requested_output_token_counts": outputs,
            "logical_prompt_tokens": sum(inputs),
            "requested_output_token_total": sum(outputs),
            "token_ids_reused_across_modes": True,
            "cache_hit_targets": 8,
            "cache_miss_targets": 8,
            "cache_hit_tokens_on_targets": [r["expected_cache_hit_tokens_on"] for r in requests],
            "cache_hit_ratios_on_targets": [r["target_cache_hit_ratio_on"] for r in requests],
            "prefix_block_counts_on_targets": [r["prefix_token_count"] // BLOCK_SIZE for r in requests if r["cache_intent"] == "hit"],
            "priming_requests_excluded_from_measurement": True,
            "cache_hit_ratio_definition": "prefix_reused_tokens / input_token_count",
        },
    }


def validate_shared_prefix_boundary(block_size: int, prefix_tokens: int, sequence_tokens: int | None = None) -> dict[str, Any]:
    """Describe upstream tail exclusion; this is not a runtime validation gate."""
    sequence_tokens = sequence_tokens if sequence_tokens is not None else prefix_tokens
    eligible_sequence_blocks = max(0, (sequence_tokens + block_size - 1) // block_size - 1)
    eligible = min(prefix_tokens // block_size, eligible_sequence_blocks)
    return {"eligible_blocks": eligible, "excluded_tail_tokens": max(0, prefix_tokens - eligible * block_size), "reason": "upstream_lookup_excludes_last_full_block"}
