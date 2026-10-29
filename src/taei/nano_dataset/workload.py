"""Deterministic independent paired workloads for cache experiments."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

WORKLOAD_NAME = "prefix_sweep"
WORKLOAD_SETS = tuple(f"prefix_sweep_set_{i:02d}" for i in range(1, 7))
LONG_CONTEXT_SETS = tuple(f"qwen3_8b_long_context_set_{i:02d}" for i in range(1, 7))
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
    for workload_name in WORKLOAD_SETS:
        for seed_index, seed in enumerate(SEEDS, 1):
            for condition_index, (cache, scheduler) in enumerate(CONDITIONS, 1):
                number += 1
                rows.append(
                    {
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
                    }
                )
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
    if name.startswith("qwen3_8b_request_scaling_n"):
        from .request_scaling import build_scaling_workload

        return build_scaling_workload(name, seed)
    if name == "qwen3_8b_cache_on_diagnostic":
        return build_cache_on_diagnostic_workload(seed)
    if (
        name.endswith("_prefill_only")
        and name.removesuffix("_prefill_only") in LONG_CONTEXT_SETS
    ):
        return build_prefill_only_workload(name, seed)
    if name.startswith("qwen3_8b_long_context_"):
        return build_long_context_workload(name, seed)
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
        prefix_blocks = prefix_rng.choice(
            [
                count
                for count in range(1, length // BLOCK_SIZE + 1)
                if count not in used_blocks
                and Fraction(count * BLOCK_SIZE, length) not in used_ratios
            ]
        )
        used_blocks.add(prefix_blocks)
        used_ratios.add(Fraction(prefix_blocks * BLOCK_SIZE, length))
        output_tokens = rng.choice(OUTPUT_TOKEN_CHOICES)
        for intent in ("hit", "miss"):
            low, high = (1000, 50000) if intent == "hit" else (50001, 90000)
            input_ids = [rng.randrange(low, high) for _ in range(length)]
            # Stay inside the same ordinary-token namespace as the remaining
            # IDs; set/pair identity must not become an out-of-vocabulary ID.
            input_ids[0] = low + set_index * 16 + i
            family = (
                f"{workload_name}-seed-{seed}-pair-{i + 1:02}"
                if intent == "hit"
                else None
            )
            prefix_tokens = prefix_blocks * BLOCK_SIZE if intent == "hit" else 0
            donor = input_ids[:prefix_tokens] + [90001] if prefix_tokens else []
            requests.append(
                {
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
                }
            )
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
            "cache_hit_tokens_on_targets": [
                r["expected_cache_hit_tokens_on"] for r in requests
            ],
            "cache_hit_ratios_on_targets": [
                r["target_cache_hit_ratio_on"] for r in requests
            ],
            "prefix_block_counts_on_targets": [
                r["prefix_token_count"] // BLOCK_SIZE
                for r in requests
                if r["cache_intent"] == "hit"
            ],
            "priming_requests_excluded_from_measurement": True,
            "cache_hit_ratio_definition": "prefix_reused_tokens / input_token_count",
        },
    }


LONG_CONTEXT_PROMPT_LENGTHS = (5121, 8705, 12289, 15873, 19457, 23041, 26625, 30209)
LONG_CONTEXT_OUTPUT_BUDGETS = (512, 512, 1024, 1024, 1024, 2048, 2048, 2048)
# Nearest block-aligned prefix counts to targets 5%, 18%, ..., 95%.
# These are independently distinct and fit upstream's eligible-block limit.
LONG_CONTEXT_REUSE_BLOCKS = (1, 6, 15, 27, 43, 62, 85, 112)
DIAGNOSTIC_WORKLOAD_PROTOCOL = "qwen3_8b_cache_on_diagnostic_v1"
DIAGNOSTIC_PROMPT_LENGTH = 13_000
DIAGNOSTIC_OUTPUT_TOKENS = 1_000
DIAGNOSTIC_PREFIX_TOKENS = (2_048, 4_096, 6_144, 8_192)


def build_cache_on_diagnostic_workload(seed: int) -> dict[str, Any]:
    """Build four measured hit/miss pairs at 13K input and 1K output tokens."""
    rng = random.Random((seed << 16) + 908_001)
    requests = []
    for index, prefix_tokens in enumerate(DIAGNOSTIC_PREFIX_TOKENS, 1):
        shared = [rng.randrange(1000, 50000) for _ in range(prefix_tokens)]
        for intent in ("hit", "miss"):
            tail = [
                rng.randrange(1000, 50000)
                for _ in range(DIAGNOSTIC_PROMPT_LENGTH - prefix_tokens)
            ]
            input_ids = shared + tail
            if intent == "miss":
                input_ids[:BLOCK_SIZE] = [
                    rng.randrange(50001, 90000) for _ in range(BLOCK_SIZE)
                ]
            family = f"cache-on-diagnostic-seed-{seed}-pair-{index:02d}"
            requests.append(
                {
                    "request_id": f"cache-on-diagnostic-S{seed}-P{index:02d}-{intent}",
                    "arrival_index": len(requests),
                    "planned_arrival_offset_ns": 0,
                    "workload_class": intent,
                    "cache_intent": intent,
                    "prompt_length": DIAGNOSTIC_PROMPT_LENGTH,
                    "prefix_family_id": family if intent == "hit" else None,
                    "prefix_group_id": family if intent == "hit" else None,
                    "prefix_token_count": prefix_tokens if intent == "hit" else 0,
                    "expected_cache_hit_tokens_on": prefix_tokens
                    if intent == "hit"
                    else 0,
                    "target_cache_hit_ratio_on": (
                        prefix_tokens / DIAGNOSTIC_PROMPT_LENGTH
                        if intent == "hit"
                        else 0
                    ),
                    "prime_input_token_ids": shared + [90001]
                    if intent == "hit"
                    else [],
                    "input_token_ids": input_ids,
                    "requested_output_tokens": DIAGNOSTIC_OUTPUT_TOKENS,
                    "temperature": 0.6,
                    "ignore_eos": True,
                }
            )
    return {
        "workload_id": f"qwen3_8b_cache_on_diagnostic_seed_{seed}",
        "workload_name": "qwen3_8b_cache_on_diagnostic",
        "set_id": "qwen3_8b_cache_on_diagnostic",
        "seed": seed,
        "requests": requests,
        "summary": {
            "request_count": len(requests),
            "prompt_lengths": [DIAGNOSTIC_PROMPT_LENGTH] * 4,
            "requested_output_token_counts": [DIAGNOSTIC_OUTPUT_TOKENS] * 8,
            "input_token_counts": [DIAGNOSTIC_PROMPT_LENGTH] * 8,
            "logical_prompt_tokens": DIAGNOSTIC_PROMPT_LENGTH * 8,
            "requested_output_token_total": DIAGNOSTIC_OUTPUT_TOKENS * 8,
            "max_prompt_plus_output_tokens": DIAGNOSTIC_PROMPT_LENGTH
            + DIAGNOSTIC_OUTPUT_TOKENS,
            "cache_hit_targets": 4,
            "cache_miss_targets": 4,
            "cache_hit_tokens_on_targets": [
                row["prefix_token_count"] for row in requests
            ],
            "priming_requests_excluded_from_measurement": True,
        },
    }


def build_long_context_workload(name: str, seed: int) -> dict[str, Any]:
    """Build matched cache-hit/miss prompts up to the 32K configured context."""
    if name not in LONG_CONTEXT_SETS:
        raise ValueError(f"unknown long-context workload set: {name}")
    set_index = LONG_CONTEXT_SETS.index(name) + 1
    rng = random.Random((seed << 16) + set_index + 900_000)
    requests = []
    for index, (length, output_tokens, reuse_blocks) in enumerate(
        zip(
            LONG_CONTEXT_PROMPT_LENGTHS,
            LONG_CONTEXT_OUTPUT_BUDGETS,
            LONG_CONTEXT_REUSE_BLOCKS,
            strict=True,
        ),
        1,
    ):
        prefix_tokens = reuse_blocks * BLOCK_SIZE
        shared = [rng.randrange(1000, 50000) for _ in range(prefix_tokens)]
        for intent in ("hit", "miss"):
            tail = [rng.randrange(1000, 90000) for _ in range(length - prefix_tokens)]
            input_ids = shared + tail
            if intent == "miss":
                input_ids[:BLOCK_SIZE] = [
                    rng.randrange(50001, 90000) for _ in range(BLOCK_SIZE)
                ]
            family = f"{name}-seed-{seed}-pair-{index:02d}" if intent == "hit" else None
            requests.append(
                {
                    "request_id": f"{name}-S{seed}-P{index:02d}-{intent}",
                    "arrival_index": len(requests),
                    "planned_arrival_offset_ns": 0,
                    "workload_class": intent,
                    "cache_intent": intent,
                    "prompt_length": length,
                    "prefix_family_id": family,
                    "prefix_group_id": family,
                    "prefix_token_count": prefix_tokens if intent == "hit" else 0,
                    "expected_cache_hit_tokens_on": prefix_tokens
                    if intent == "hit"
                    else 0,
                    "target_cache_hit_ratio_on": (
                        prefix_tokens / length if intent == "hit" else 0
                    ),
                    "prime_input_token_ids": (
                        shared + [90001] if intent == "hit" else []
                    ),
                    "input_token_ids": input_ids,
                    "requested_output_tokens": output_tokens,
                    "temperature": 0.6,
                    "ignore_eos": True,
                }
            )
    return {
        "workload_id": f"{name}-seed-{seed}",
        "workload_name": "qwen3_8b_long_context",
        "set_id": name,
        "seed": seed,
        "requests": requests,
        "summary": {
            "request_count": len(requests),
            "prompt_lengths": list(LONG_CONTEXT_PROMPT_LENGTHS),
            "requested_output_token_counts": list(LONG_CONTEXT_OUTPUT_BUDGETS),
            "input_token_counts": [len(row["input_token_ids"]) for row in requests],
            "logical_prompt_tokens": sum(
                len(row["input_token_ids"]) for row in requests
            ),
            "requested_output_token_total": sum(
                row["requested_output_tokens"] for row in requests
            ),
            "max_prompt_plus_output_tokens": max(
                row["prompt_length"] + row["requested_output_tokens"]
                for row in requests
            ),
            "cache_hit_tokens_on_targets": [
                row["prefix_token_count"] for row in requests
            ],
            "cache_hit_ratios_on_targets": [
                row["target_cache_hit_ratio_on"] for row in requests
            ],
            "cache_hit_targets": 8,
            "cache_miss_targets": 8,
            "prefix_block_counts_on_targets": [
                row["prefix_token_count"] // BLOCK_SIZE
                for row in requests
                if row["cache_intent"] == "hit"
            ],
            "donor_prompt_token_counts": [
                len(row["prime_input_token_ids"])
                for row in requests
                if row["prime_input_token_ids"]
            ],
            "donor_prefill_tokens": sum(
                len(row["prime_input_token_ids"])
                for row in requests
                if row["prime_input_token_ids"]
            ),
            "priming_requests_excluded_from_measurement": True,
        },
    }


def build_prefill_only_workload(name: str, seed: int) -> dict[str, Any]:
    """Reuse the exact long-context inputs with one requested output token."""
    base_name = name.removesuffix("_prefill_only")
    if name != f"{base_name}_prefill_only" or base_name not in LONG_CONTEXT_SETS:
        raise ValueError(f"unknown prefill-only workload set: {name}")
    workload = build_long_context_workload(base_name, seed)
    workload["workload_id"] = f"{name}-seed-{seed}"
    for request in workload["requests"]:
        request["request_id"] += "-prefill-only"
        request["requested_output_tokens"] = 1
    summary = workload["summary"]
    summary["requested_output_token_counts"] = [1] * len(LONG_CONTEXT_OUTPUT_BUDGETS)
    summary["requested_output_token_total"] = len(workload["requests"])
    summary["max_prompt_plus_output_tokens"] = max(LONG_CONTEXT_PROMPT_LENGTHS) + 1
    return workload


def build_long_context_plan(
    context_cap: int = 32768,
    order_seed: int = 20261023,
    *,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 16384,
) -> dict[str, Any]:
    """Return exact deterministic launch/request schedule for long-context runs."""
    required = max(
        p + o
        for p, o in zip(
            LONG_CONTEXT_PROMPT_LENGTHS, LONG_CONTEXT_OUTPUT_BUDGETS, strict=True
        )
    )
    if max_num_seqs < 1 or max_num_batched_tokens < 1:
        raise ValueError("engine sequence and batched-token limits must be positive")
    if context_cap < required:
        raise ValueError(
            f"configured max_model_len {context_cap} is below workload "
            f"requirement {required}"
        )
    rows = []
    for set_name in LONG_CONTEXT_SETS:
        for seed_index, seed in enumerate(SEEDS, 1):
            for condition_index, (cache, scheduler) in enumerate(CONDITIONS, 1):
                rows.append(
                    {
                        "run_id": f"L{len(rows) + 1:02}",
                        "condition_id": f"C{condition_index:02}",
                        "set_id": set_name,
                        "pair_id": f"{set_name}-seed-{seed}",
                        "split_group_id": f"{set_name}-family",
                        "repeat_id": f"{seed_index:02}",
                        "seed": seed,
                        "workload_id": set_name,
                        "prefix_cache_mode": cache,
                        "scheduler_mode": scheduler,
                    }
                )
    order = list(range(len(rows)))
    random.Random(order_seed).shuffle(order)
    for rank, index in enumerate(order, 1):
        rows[index]["run_order"] = rank
    matrix = sorted(rows, key=lambda row: row["run_order"])
    for index, original in enumerate(list(matrix), 1):
        matrix.append(
            {
                **original,
                "run_id": f"P{index:02}",
                "pair_id": original["pair_id"] + "-prefill-only",
                "workload_id": original["workload_id"] + "_prefill_only",
                "run_order": len(matrix) + 1,
            }
        )
    return {
        "matrix": matrix,
        "run_count": len(matrix),
        "request_count": len(matrix) * 16,
        "workload_protocol": "qwen3_8b_long_context_32k_v1",
        "schedule": {
            "prompt_lengths": list(LONG_CONTEXT_PROMPT_LENGTHS),
            "output_budgets": list(LONG_CONTEXT_OUTPUT_BUDGETS),
            "prefill_only_output_budget": 1,
            "prefill_only_run_count": 48,
            "prefix_blocks_per_hit": list(LONG_CONTEXT_REUSE_BLOCKS),
            "target_reuse_fractions_per_hit": [
                blocks * BLOCK_SIZE / length
                for blocks, length in zip(
                    LONG_CONTEXT_REUSE_BLOCKS, LONG_CONTEXT_PROMPT_LENGTHS, strict=True
                )
            ],
            "max_prompt_plus_output_tokens": required,
            "context_cap": context_cap,
            "sequence_budget_enforcement": (
                "prompt_length + requested_output_tokens <= context_cap"
            ),
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "dynamic_max_num_seqs_semantics": (
                "per_schedule_selection_cap; upstream dynamic running set may exceed it"
            ),
            "fixed_batch_policy": "admit waves no larger than max_num_seqs",
            "prefill_policy": (
                "pinned_scheduler_chunks_only_the_first_scheduled_sequence_when_"
                "remaining_step_budget_is_insufficient"
            ),
            "measured_requests_per_launch": 16,
            "donor_requests_per_launch": 8,
            "warmup_requests_per_launch": 1,
            "seeds": list(SEEDS),
            "treatments_per_set_seed": 4,
            "set_count": len(LONG_CONTEXT_SETS),
        },
    }


def validate_shared_prefix_boundary(
    block_size: int, prefix_tokens: int, sequence_tokens: int | None = None
) -> dict[str, Any]:
    """Describe upstream tail exclusion; this is not a runtime validation gate."""
    sequence_tokens = sequence_tokens if sequence_tokens is not None else prefix_tokens
    eligible_sequence_blocks = max(
        0, (sequence_tokens + block_size - 1) // block_size - 1
    )
    eligible = min(prefix_tokens // block_size, eligible_sequence_blocks)
    return {
        "eligible_blocks": eligible,
        "excluded_tail_tokens": max(0, prefix_tokens - eligible * block_size),
        "reason": "upstream_lookup_excludes_last_full_block",
    }
