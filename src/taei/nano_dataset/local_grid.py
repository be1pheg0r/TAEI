"""Local single-GPU Qwen3-0.6B grid: controlled prefill/decode series and checks.

Each run carries an explicit ``workload_spec``: a list of requests with prompt
length L, output length O, an optional shared prefix (``family`` and
``prefix_blocks``) and an arrival offset. Token IDs are drawn from the run seed;
requests of one family share their first ``prefix_blocks`` blocks, and the
first of them by arrival carries the untimed donor ``prefix + [90001]`` that
puts the prefix into the cache (the existing priming path), so R = prefix / L.

Series (``phase`` A then B):
  REF  prefix_sweep_set_01, seed 42, cache off, upstream: noise and drift check;
  S0   the 48-run server matrix of Qwen3-0.6B, engine with max_num_seqs 8;
  S1   prefill only (O = 1): one request over L x R, k requests in one step,
       and hit + miss in one step;
  S2   decode: B identical requests of length L with O = 256 (190 at L = 65),
       so B is fixed
       and the summed context grows linearly over the steps;
  S3   closed batches of N requests whose O sets the prefill share of time,
       computed from the phase A coefficients;
  S4   Poisson arrivals of 128 requests at a load rho of the model capacity.

Phase C:
  S5   closed loop: exactly N requests in the system, a new one submitted as
       soon as one finishes; 20 warm-up and 200 measured completions (60 at
       N = 1), then the tail drains. One request pool per seed (the S4 mix)
       for every N, so points differ only in N.

Every point is checked against the KV capacity before it is planned: blocks
held by donors plus all measured requests at once must fit in
``kv_fill`` of the cache, otherwise the point is dropped and listed.
"""

from __future__ import annotations

import math
import random
from typing import Any

from .workload import (
    BLOCK_SIZE,
    CONDITIONS,
    OUTPUT_TOKEN_CHOICES,
    build_run_matrix,
)

PROTOCOL = "local_grid_v1"
DONOR_MARK = 90001
TOKEN_RANGE = (1000, 90000)
MAX_BATCHED_TOKENS = 16384

S1_LENGTHS = (
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097,
    6145, 8193, 12289, 16385, 20481, 24577, 32769, 36865,
)  # fmt: skip
S1_REUSE = (0.1, 0.25, 0.5, 0.75, 0.9, 0.97)
S1_STEP_SIZES = (2, 4, 8, 16, 32, 64)
S1_STEP_LENGTHS = (129, 257, 513, 1025, 2049, 4097, 8193)
S1_MIXED_LENGTHS = (513, 1025, 2049, 3073, 4097, 6145)
S1_REPEATS = 5
S2_BATCHES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160)
# (L, O): L + O <= 256 keeps a request in one KV block, so B reaches 160.
S2_POINTS = (
    (65, 190), (129, 256), (257, 256), (513, 256), (1025, 256),
    (2049, 256), (4097, 256), (8193, 256), (16385, 256), (32769, 256),
)  # fmt: skip
S2_REPEATS = 3
S3_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
S3_SHARES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
S3_LENGTHS = (257, 513, 1025, 4097)
S4_REQUESTS = 256
S4_LOADS = (0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25)
S4_FAMILIES = 4
S4_FIXED_BATCH = 8
S5_SIZES = {
    ("off", "upstream_dynamic"): (1, 2, 4, 6, 8, 12, 16, 24, 32),
    ("on", "upstream_dynamic"): (1, 2, 4, 6, 8, 12, 16, 24, 32),
    ("off", "fixed_batch"): (2, 4, 6, 8, 12, 16),
}
S5_WARMUP = 20
S5_MEASURE = 200
S5_MEASURE_N1 = 60
S5_POOL = 260
S5_BUDGET_S = 2.5 * 3600
# Per variant: latency of a lone request (s) and the saturated rate (req/s),
# both from the S4 traces (runs/20261028_190653/concurrency.json, rho = 0.25
# and the plateau above the knee); fixed_batch alone waits for its wave.
S5_RATES = {
    ("off", "upstream_dynamic"): (1.58, 4.0),
    ("on", "upstream_dynamic"): (1.40, 5.3),
    ("off", "fixed_batch"): (2.2, 2.05),
}
S5_OVERHEAD_S = 10.0
SEEDS = (42, 43, 44)
REF_EVERY = 25
REF_EVERY_C = 6  # about ten minutes of S5
# Phase D. S6/S7: one batch submitted at once, O = 4 warm-up + 36 measured
# decode steps. S8: closed loops on workloads W1-W4 with a 3-minute window.
D_DECODE_WARMUP = 4
D_DECODE_STEPS = 40
S6_BATCHES = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96)
S6_SUMS = (8000, 16000, 24000, 32000, 36000, 40000, 44000)
S6_MIXED_BATCHES = (8, 16, 32)
S6_MIXED_SUMS = (16000, 32000, 44000)
S7_BATCHES = (4, 8, 16, 24, 32)
S7_LENGTHS = (1025, 2049, 4097)
S7_REUSE = (0.25, 0.5, 0.75, 0.9)
S7_FAMILIES = (1, 4)
S8_GRID = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
S8_EXTRA = (3, 6, 96, 128)  # >= 4 sizes below the KV knee and 2 above it
S8_FIXED = (2, 4, 6, 8, 12, 16)
S8_FAMILIES = 4
S8_WARMUP = 20
S8_MEASURE = 200
S8_MIN_MEASURE = 60
S8_MAX_WINDOW_S = 180.0
S8_REPEATS = {"W1": 3, "W2": 3, "W3": 2, "W4": 2}
D_POOL = 320  # >= the largest N + 20 + 200 - 1
P4_SIZES = {
    ("off", "upstream_dynamic"): (8, 24),
    ("on", "upstream_dynamic"): (48, 64),
}
D_BLOCKS = ("S6", "S7", "S8-W1", "S8-W2", "S8-W3", "S8-W4", "S5")
D_REF_EVERY_S = 1800.0
D_MARGIN = 1.2  # the phase A model underestimates long-context decode
D_OVERHEAD_S = 3.0  # per attempt, phases A and C
REF_SECONDS = 8.0


def _condition(cache: str, scheduler: str) -> str:
    return f"C{CONDITIONS.index((cache, scheduler)) + 1:02}"


def _blocks(tokens: int) -> int:
    return -(-tokens // BLOCK_SIZE)


def _shared_blocks(spec: dict[str, Any]) -> int:
    """Prefix blocks of all families, each stored once (held from priming)."""
    longest: dict[str, int] = {}
    for r in spec["requests"]:
        if r["family"] is not None:
            longest[r["family"]] = max(longest.get(r["family"], 0), r["prefix_blocks"])
    return sum(longest.values())


def _own_blocks(request: dict[str, Any]) -> int:
    return _blocks(request["L"] + request["O"]) - request["prefix_blocks"]


def kv_blocks_needed(spec: dict[str, Any]) -> int:
    """Shared prefixes once plus the own blocks of every request at once."""
    return _shared_blocks(spec) + sum(_own_blocks(r) for r in spec["requests"])


def largest_request_blocks(spec: dict[str, Any]) -> int:
    return _shared_blocks(spec) + max(_own_blocks(r) for r in spec["requests"])


def _request(L, out, family=None, prefix_blocks=0, offset_ns=0, donor=False) -> dict:
    return {
        "L": int(L),
        "O": int(out),
        "family": family,
        "prefix_blocks": int(prefix_blocks),
        "arrival_offset_ns": int(offset_ns),
        "donor": bool(donor and prefix_blocks),
    }


def _shared(n, L, out, prefix_blocks, family="f0") -> list[dict]:
    """n requests sharing one prefix; the first carries the donor."""
    if not prefix_blocks:
        return [_request(L, out) for _ in range(n)]
    return [_request(L, out, family, prefix_blocks, donor=i == 0) for i in range(n)]


def prefix_blocks_for(L: int, R: float) -> int:
    """Nearest whole-block prefix for reuse R; upstream caches only full blocks
    before the last one, so at most (L - 1) // 256 blocks."""
    return min(max(round(R * L / BLOCK_SIZE), 1), (L - 1) // BLOCK_SIZE)


def build_spec_workload(name: str, seed: int, spec: dict[str, Any]) -> dict[str, Any]:
    """Token IDs and donors for an explicit request list (deterministic in seed)."""
    rng = random.Random(f"{name}:{seed}")
    longest: dict[str, int] = {}
    for item in spec["requests"]:
        if item["family"] is not None:
            longest[item["family"]] = max(
                longest.get(item["family"], 0), item["prefix_blocks"] * BLOCK_SIZE
            )
    prefixes = {
        family: [rng.randrange(*TOKEN_RANGE) for _ in range(size)]
        for family, size in sorted(longest.items())
    }
    requests = []
    order = sorted(
        range(len(spec["requests"])),
        key=lambda i: spec["requests"][i]["arrival_offset_ns"],
    )
    for rank, index in enumerate(order):
        item = spec["requests"][index]
        L, blocks = item["L"], item["prefix_blocks"]
        family = item["family"]
        shared = prefixes[family][: blocks * BLOCK_SIZE] if family else []
        tail = [rng.randrange(*TOKEN_RANGE) for _ in range(L - len(shared))]
        input_ids = shared + tail
        prefix_tokens = len(shared)
        requests.append(
            {
                "request_id": f"{name}-S{seed}-Q{index + 1:03}",
                "arrival_index": rank,
                "planned_arrival_offset_ns": item["arrival_offset_ns"],
                "workload_class": "hit" if prefix_tokens else "miss",
                "cache_intent": "hit" if prefix_tokens else "miss",
                "prompt_length": L,
                "prefix_family_id": family,
                "prefix_group_id": family,
                "prefix_token_count": prefix_tokens,
                "expected_cache_hit_tokens_on": prefix_tokens,
                "target_cache_hit_ratio_on": prefix_tokens / L,
                "prime_input_token_ids": (
                    shared + [DONOR_MARK] if item.get("donor") else []
                ),
                "input_token_ids": input_ids,
                "requested_output_tokens": item["O"],
                "temperature": 0.6,
                "ignore_eos": True,
                **(
                    {"closed_loop": spec["closed_loop"]}
                    if "closed_loop" in spec
                    else {}
                ),
            }
        )
    requests.sort(key=lambda r: r["request_id"])
    inputs = [len(r["input_token_ids"]) for r in requests]
    outputs = [r["requested_output_tokens"] for r in requests]
    return {
        "workload_id": f"{name}-seed-{seed}",
        "workload_name": name,
        "set_id": spec["series"],
        "seed": seed,
        "requests": requests,
        "summary": {
            "request_count": len(requests),
            "input_token_counts": inputs,
            "requested_output_token_counts": outputs,
            "logical_prompt_tokens": sum(inputs),
            "requested_output_token_total": sum(outputs),
            "point": spec.get("point", {}),
            "kv_blocks_needed": kv_blocks_needed(spec),
            "priming_requests_excluded_from_measurement": True,
            "cache_hit_ratio_definition": "prefix_reused_tokens / input_token_count",
        },
    }


def _row(series, name, seed, cache, scheduler, spec, repeat=1, **extra) -> dict:
    return {
        "series": series,
        "condition_id": _condition(cache, scheduler),
        "set_id": series,
        "pair_id": f"{name}-seed-{seed}",
        "split_group_id": name,
        "repeat_id": f"{repeat:02}",
        "seed": seed,
        "workload_id": name,
        "prefix_cache_mode": cache,
        "scheduler_mode": scheduler,
        "workload_spec": {"series": series, **spec},
        **extra,
    }


def _s1() -> list[dict]:
    points = []
    for L in S1_LENGTHS:
        points.append(("off", f"S1-L{L}-R0-k1", _shared(1, L, 1, 0), {"k": 1}))
        points.append(("on", f"S1-L{L}-R0-k1-on", _shared(1, L, 1, 0), {"k": 1}))
        seen = set()
        for R in S1_REUSE:
            p = prefix_blocks_for(L, R)
            if p in seen:
                continue
            seen.add(p)
            points.append(("on", f"S1-L{L}-p{p}-k1", _shared(1, L, 1, p), {"k": 1}))
    for k in S1_STEP_SIZES:
        for L in S1_STEP_LENGTHS:
            if k * L > MAX_BATCHED_TOKENS:
                continue
            points.append(("off", f"S1-L{L}-R0-k{k}", _shared(k, L, 1, 0), {"k": k}))
            p = prefix_blocks_for(L, 0.5)
            points.append(("on", f"S1-L{L}-p{p}-k{k}", _shared(k, L, 1, p), {"k": k}))
    for L in S1_MIXED_LENGTHS:
        p = prefix_blocks_for(L, 0.9)
        for hits in (1, 2):  # hits share one prefix, misses have none
            if 2 * hits * L > MAX_BATCHED_TOKENS:
                continue
            reqs = _shared(hits, L, 1, p) + [_request(L, 1) for _ in range(hits)]
            k = 2 * hits
            points.append(("on", f"S1-L{L}-mixed-k{k}", reqs, {"k": k, "mixed": True}))
    rows = []
    for cache, name, reqs, point in points:
        for repeat in range(1, S1_REPEATS + 1):
            spec = {"requests": reqs, "point": point}
            rows.append(
                _row("S1", name, 42, cache, "upstream_dynamic", spec, repeat=repeat)
            )
    return rows


def _s2() -> list[dict]:
    rows = []
    for L, out in S2_POINTS:
        for B in S2_BATCHES:
            name = f"S2-L{L}-B{B}"
            spec = {"requests": _shared(B, L, out, 0), "point": {"B": B}}
            for repeat in range(1, S2_REPEATS + 1):
                rows.append(
                    _row("S2", name, 42, "off", "upstream_dynamic", spec, repeat)
                )
    return rows


def prefill_seconds(coef: dict, n: int, L: int, R: float) -> float:
    """Prefill time of n requests of length L with reuse R, by formula 1."""
    new = n * L * (1 - R)
    steps = max(math.ceil(new / MAX_BATCHED_TOKENS), 1)
    return coef["a"] * new + coef["b"] * n * L**2 * (1 - R**2) + coef["c"] * steps


def output_for_share(coef: dict, n: int, L: int, R: float, share: float) -> int:
    """O such that prefill takes ``share`` of the time of a closed batch.

    Decode of B = n requests for O steps with mean context L + O / 2:
    O (d0 + d1 n L + d2 n) + d1 n O^2 / 2 = T_pf (1 - s) / s.
    """
    target = prefill_seconds(coef, n, L, R) * (1 - share) / share
    lin = coef["d0"] + coef["d1"] * n * L + coef["d2"] * n
    quad = coef["d1"] * n / 2
    if quad <= 0:
        return round(target / lin)
    return round((-lin + math.sqrt(lin**2 + 4 * quad * target)) / (2 * quad))


def _s3(coef: dict) -> list[dict]:
    rows = []
    for n in S3_SIZES:
        for L in S3_LENGTHS:
            for share in S3_SHARES:
                for cache, R in (("off", 0.0), ("on", 0.5)):
                    p = prefix_blocks_for(L, R) if R else 0
                    R_real = p * BLOCK_SIZE / L
                    out = output_for_share(coef, n, L, R_real, share)
                    if out < 2:
                        continue
                    name = f"S3-N{n}-L{L}-s{round(share * 100)}-{cache}"
                    point = {"N": n, "target_prefill_share": share, "R": R_real}
                    spec = {"requests": _shared(n, L, out, p), "point": point}
                    for seed in SEEDS:
                        rows.append(
                            _row("S3", name, seed, cache, "upstream_dynamic", spec)
                        )
    return rows


def _s4(coef: dict) -> list[dict]:
    rows = []
    mean_L = (513 + 4097) / 2
    mean_O = sum(OUTPUT_TOKEN_CHOICES) / len(OUTPUT_TOKEN_CHOICES)
    # Token prices (formula 5) for a mean request and a decode step of B = 16.
    B, ell = 16, mean_L + mean_O / 2
    a_t = coef["a"] + coef["b"] * mean_L + coef["c"] / mean_L
    b_t = coef["d0"] / B + coef["d1"] * ell + coef["d2"]
    capacity = 1 / (a_t * mean_L + b_t * mean_O)  # requests per second
    for rho in S4_LOADS:
        for cache in ("off", "on"):
            for scheduler in ("upstream_dynamic", "fixed_batch"):
                for seed in SEEDS:
                    rng = random.Random(f"S4:{rho}:{cache}:{seed}")
                    family_blocks = [rng.randint(2, 8) for _ in range(S4_FAMILIES)]
                    t = 0.0
                    reqs, donors = [], set()
                    for _ in range(S4_REQUESTS):
                        t += rng.expovariate(rho * capacity)
                        L = rng.randrange(513, 4097)
                        L += L % BLOCK_SIZE == 0
                        out = rng.choice(OUTPUT_TOKEN_CHOICES)
                        f = rng.randrange(S4_FAMILIES)
                        p = min(family_blocks[f], (L - 1) // BLOCK_SIZE)
                        reqs.append(
                            _request(
                                L, out, f"f{f}", p, int(t * 1e9), donor=f not in donors
                            )
                        )
                        donors.add(f)
                    name = f"S4-rho{round(rho * 100)}-{cache}-{scheduler}"
                    point = {"rho": rho, "capacity_rps": capacity}
                    spec = {"requests": reqs, "point": point, "allow_queueing": True}
                    rows.append(
                        _row(
                            "S4",
                            name,
                            seed,
                            cache,
                            scheduler,
                            spec,
                            fixed_batch_size=S4_FIXED_BATCH
                            if scheduler == "fixed_batch"
                            else None,
                        )
                    )
    return rows


def _s5_pool(seed: int, size: int = S5_POOL) -> list[dict]:
    """The S4 request mix in submission order; the first of a family is donor.

    A larger ``size`` extends the same pool: its first requests do not change.
    """
    rng = random.Random(f"S5:pool:{seed}")
    family_blocks = [rng.randint(2, 8) for _ in range(S4_FAMILIES)]
    pool, donors = [], set()
    for _ in range(size):
        L = rng.randrange(513, 4097)
        L += L % BLOCK_SIZE == 0
        out = rng.choice(OUTPUT_TOKEN_CHOICES)
        f = rng.randrange(S4_FAMILIES)
        p = min(family_blocks[f], (L - 1) // BLOCK_SIZE)
        pool.append(_request(L, out, f"f{f}", p, 0, donor=f not in donors))
        donors.add(f)
    return pool


def s5_measure(n: int) -> int:
    return S5_MEASURE_N1 if n == 1 else S5_MEASURE


def _s5(
    sizes: dict = S5_SIZES,
    warmup: int = S5_WARMUP,
    measure=s5_measure,
    pool_size: int = S5_POOL,
) -> list:
    rows = []
    for seed in SEEDS:
        pool = _s5_pool(seed, pool_size)
        for (cache, scheduler), ns in sizes.items():
            for n in ns:
                m = measure(n)
                loop = {"N": n, "warmup": warmup, "measure": m}
                name = f"S5-N{n}-{cache}-{scheduler}"
                spec = {
                    "requests": pool[: n + warmup + m - 1],
                    "point": loop,
                    "closed_loop": loop,
                    "allow_queueing": True,
                }
                rows.append(
                    _row(
                        "S5",
                        name,
                        seed,
                        cache,
                        scheduler,
                        spec,
                        fixed_batch_size=S4_FIXED_BATCH
                        if scheduler == "fixed_batch"
                        else None,
                    )
                )
    return rows


def s5_estimate_seconds(rows: list[dict]) -> float:
    """Wall time of S5 attempts: completions at min(N / t1, plateau) plus tail."""
    total = 0.0
    for row in rows:
        if row["series"] != "S5":
            continue
        t1, plateau = S5_RATES[(row["prefix_cache_mode"], row["scheduler_mode"])]
        loop = row["workload_spec"]["closed_loop"]
        n = loop["N"]
        rate = min(n / t1, plateau)
        total += (loop["warmup"] + loop["measure"]) / rate
        total += t1 + (n - 1) / plateau + S5_OVERHEAD_S
    return total


def _s5_within_budget() -> list[dict]:
    """Full S5 grid, or without N = 32 when the estimate exceeds the budget."""
    rows = _s5()
    if s5_estimate_seconds(rows) > S5_BUDGET_S:
        rows = [r for r in rows if r["workload_spec"]["closed_loop"]["N"] != 32]
    return rows


def kv_concurrency(requests: list[dict], cache: str, kv_blocks: int) -> float:
    """Requests that fit in the KV cache at once, at mid-decode on average.

    Blocks are whole (256 tokens); with the cache on, the prefixes of all
    families are held once and a request owns only the blocks past its prefix.
    """
    on = cache == "on"
    shared = _shared_blocks({"requests": requests}) if on else 0
    own = sum(
        _blocks(r["L"] + r["O"] // 2) - (r["prefix_blocks"] if on else 0)
        for r in requests
    ) / len(requests)
    return (kv_blocks - shared) / own


def _decode_rows(name, lengths, cache, point, requests=None) -> list[dict]:
    reqs = requests or [_request(L, D_DECODE_STEPS) for L in lengths]
    spec = {"requests": reqs, "point": {**point, "warmup_steps": D_DECODE_WARMUP}}
    return [
        _row(name.split("-")[0], name, seed, cache, "upstream_dynamic", spec)
        for seed in SEEDS
    ]


def _s6(kv_blocks: int, kv_fill: float, max_model_len: int) -> list[dict]:
    """B requests of distinct content, submitted at once, cache off."""
    rows = []
    limit = int(kv_fill * kv_blocks)
    for B in S6_BATCHES:
        # The largest equal length whose blocks, with all O tokens, still fit.
        ell_max = min(
            (limit // B) * BLOCK_SIZE - D_DECODE_STEPS, max_model_len - D_DECODE_STEPS
        )
        for total, label in [(t, f"{t // 1000}k") for t in S6_SUMS] + [
            (ell_max * B, "max")
        ]:
            if total < B:
                continue
            base, extra = divmod(total, B)
            lengths = [base + (i < extra) for i in range(B)]
            point = {"B": B, "sum_ell": total, "mixed": False}
            rows += _decode_rows(f"S6-B{B}-S{label}", lengths, "off", point)

    def mixed(B: int, total: int) -> list[int]:
        mean = total / B
        lengths = [round(mean * (0.25 + 1.5 * i / (B - 1))) for i in range(B)]
        lengths[-1] += total - sum(lengths)
        return lengths

    def blocks(lengths: list[int]) -> int:
        return sum(_blocks(L + D_DECODE_STEPS) for L in lengths)

    for B in S6_MIXED_BATCHES:
        for target in S6_MIXED_SUMS:
            # Mixed lengths waste more partial blocks: a sum that does not fit
            # becomes the largest one of the same shape that does ("max").
            total, label = target, f"{target // 1000}k"
            while blocks(mixed(B, total)) > limit:
                total, label = total - 100, "max"
            lengths = mixed(B, total)
            random.Random(f"S6m:{B}:{target}").shuffle(lengths)
            point = {"B": B, "sum_ell": total, "target": target, "mixed": True}
            rows += _decode_rows(f"S6-B{B}-S{label}-mixed", lengths, "off", point)
    return rows


def _s7() -> list[dict]:
    """B requests of length L sharing prefixes of reuse R in 1 or 4 families;
    at R = 0 the same batch with the cache on and off."""
    rows = []
    for B in S7_BATCHES:
        for L in S7_LENGTHS:
            for cache in ("on", "off"):
                point = {"B": B, "L": L, "R": 0.0, "families": 0}
                rows += _decode_rows(f"S7-B{B}-L{L}-R0-{cache}", [L] * B, cache, point)
            for R in S7_REUSE:
                p = prefix_blocks_for(L, R)
                for F in S7_FAMILIES:
                    reqs = [
                        _request(L, D_DECODE_STEPS, f"f{i % F}", p, donor=i < F)
                        for i in range(B)
                    ]
                    point = {
                        "B": B,
                        "L": L,
                        "R": p * BLOCK_SIZE / L,
                        "R_target": R,
                        "families": F,
                    }
                    name = f"S7-B{B}-L{L}-R{round(R * 100)}-F{F}"
                    rows += _decode_rows(name, None, "on", point, reqs)
    return rows


def _s8_pool(workload: str, seed: int, size: int = D_POOL) -> list[dict]:
    """Requests of one S8 workload in submission order, 4 prefix families; the
    first request of a family with a prefix carries the donor."""
    rng = random.Random(f"S8:{workload}:{seed}")
    pool, donors = [], set()
    for _ in range(size):
        f = rng.randrange(S8_FAMILIES)
        if workload == "W1":  # fixed: L 2048 with a 1024-token family prefix
            L, out, p = 2048, 128, 1024 // BLOCK_SIZE
        elif workload == "W2":  # long answers
            L, out = rng.randint(256, 1024), rng.randint(256, 1024)
            p = prefix_blocks_for(L, 0.5) if L > BLOCK_SIZE else 0
        elif workload == "W3":  # long prompts
            L, out = rng.randint(4096, 12288), rng.randint(16, 64)
            p = prefix_blocks_for(L, 0.5)
        elif workload == "W4":  # the S5 mix with R about 0.8
            L = rng.randrange(513, 4097)
            L += L % BLOCK_SIZE == 0
            out = rng.choice(OUTPUT_TOKEN_CHOICES)
            p = prefix_blocks_for(L, 0.8)
        else:
            raise ValueError(f"unknown S8 workload: {workload}")
        family = f"f{f}" if p else None
        pool.append(_request(L, out, family, p, donor=bool(p) and f not in donors))
        if p:
            donors.add(f)
    return pool


def s8_sizes(pool: list[dict], cache: str, kv_blocks: int) -> tuple[float, list[int]]:
    """The KV knee and N: the grid up to it (at least 4 points, filled from the
    extra sizes) and the first 2 sizes above it."""
    knee = kv_concurrency(pool, cache, kv_blocks)
    below = [n for n in S8_GRID if n <= knee]
    for n in S8_EXTRA:
        if len(below) < 4 and n <= knee and n not in below:
            below.append(n)
    above = [n for n in sorted(set(S8_GRID) | set(S8_EXTRA)) if n > knee][:2]
    return knee, sorted(below) + above


def _s8(kv_blocks: int) -> list[dict]:
    rows = []
    for w, repeats in S8_REPEATS.items():
        variants = [("off", "upstream_dynamic"), ("on", "upstream_dynamic")]
        if w == "W1":
            variants += [("off", "fixed_batch"), ("on", "fixed_batch")]
        reference = _s8_pool(w, SEEDS[0])  # one N grid for every seed
        for cache, scheduler in variants:
            if scheduler == "fixed_batch":
                knee, sizes = float(S4_FIXED_BATCH), list(S8_FIXED)
            else:
                knee, sizes = s8_sizes(reference, cache, kv_blocks)
            if w == "W2":
                sizes = [n for n in sizes if n != 1]
            for seed in SEEDS[:repeats]:
                pool = _s8_pool(w, seed)
                for n in sizes:
                    short = w == "W2" and n <= 4
                    loop = {
                        "N": n,
                        "warmup": 5 if short else S8_WARMUP,
                        "measure": S8_MEASURE,
                        "min_measure": 40 if w == "W2" and n == 2 else S8_MIN_MEASURE,
                        "max_window_s": S8_MAX_WINDOW_S,
                    }
                    spec = {
                        "requests": pool[: n + loop["warmup"] + S8_MEASURE - 1],
                        "point": {"workload": w, "kv_concurrency": knee, **loop},
                        "closed_loop": loop,
                        "allow_queueing": True,
                    }
                    name = f"S8-{w}-N{n}-{cache}-{scheduler}"
                    rows.append(
                        _row(
                            "S8",
                            name,
                            seed,
                            cache,
                            scheduler,
                            spec,
                            block=f"S8-{w}",
                            fixed_batch_size=S4_FIXED_BATCH
                            if scheduler == "fixed_batch"
                            else None,
                        )
                    )
    return rows


def _d_smoke(kv_blocks: int, kv_fill: float, max_model_len: int) -> list[dict]:
    """One S6, S7 and short S8 attempt: new step fields and the time limit."""
    s6 = [r for r in _s6(kv_blocks, kv_fill, max_model_len) if r["seed"] == 42]
    s7 = [r for r in _s7() if r["seed"] == 42]
    pool = _s8_pool("W1", 42)
    loop = {"N": 4, "warmup": 5, "measure": 200, "min_measure": 10, "max_window_s": 5}
    spec = {
        "requests": pool[: 4 + 5 + 200 - 1],
        "point": {"workload": "W1", **loop},
        "closed_loop": loop,
        "allow_queueing": True,
    }
    s8 = _row("S8", "S8-W1-N4-on-upstream_dynamic", 42, "on", "upstream_dynamic", spec)
    return [
        next(r for r in s6 if r["workload_id"] == "S6-B16-S44k"),
        next(r for r in s7 if r["workload_id"] == "S7-B8-L2049-R90-F4"),
        s8,
    ]


def closed_loop_rate(
    requests: list[dict], cache: str, scheduler: str, n: int, coef: dict, kv_blocks
) -> float:
    """Completions per second with N in the system, by formulas 1 and 5:
    1 / (prefill per request + O decode steps of B shared by B requests),
    B = min(N, KV concurrency) or the wave of fixed_batch."""
    on = cache == "on"
    L = sum(r["L"] for r in requests) / len(requests)
    out = sum(r["O"] for r in requests) / len(requests)
    prefill = sum(
        prefill_seconds(
            coef, 1, r["L"], r["prefix_blocks"] * BLOCK_SIZE / r["L"] if on else 0.0
        )
        for r in requests
    ) / len(requests)
    if scheduler == "fixed_batch":
        B = min(n, S4_FIXED_BATCH)
    else:
        B = min(n, kv_concurrency(requests, cache, kv_blocks))
    step = coef["d0"] + coef["d1"] * B * (L + out / 2) + coef["d2"] * B
    return 1 / (prefill + out * step / B)


def d_row_seconds(row: dict, coef: dict, kv_blocks: int) -> float:
    """Expected wall time of one phase D attempt, with a margin."""
    if row["series"] == "REF":
        return REF_SECONDS
    spec = row["workload_spec"]
    reqs = spec["requests"]
    loop = spec.get("closed_loop")
    if loop is None:  # one batch: prefill of every request, then O decode steps
        on = row["prefix_cache_mode"] == "on"
        prefill = sum(
            prefill_seconds(
                coef, 1, r["L"], r["prefix_blocks"] * BLOCK_SIZE / r["L"] if on else 0.0
            )
            for r in reqs
        )
        B, out = len(reqs), max(r["O"] for r in reqs)
        ell = sum(r["L"] for r in reqs) / B + out / 2
        decode = out * (coef["d0"] + coef["d1"] * B * ell + coef["d2"] * B)
        return D_MARGIN * (prefill + decode) + D_OVERHEAD_S
    rate = closed_loop_rate(
        reqs,
        row["prefix_cache_mode"],
        row["scheduler_mode"],
        loop["N"],
        coef,
        kv_blocks,
    )
    rate /= D_MARGIN
    measure = loop["measure"] / rate
    if loop.get("max_window_s"):
        measure = min(measure, max(loop["max_window_s"], loop["min_measure"] / rate))
    return (loop["warmup"] + loop["N"]) / rate + measure + D_OVERHEAD_S


def _order_d(rows: list[dict], order_seed: int, coef: dict, kv_blocks: int) -> list:
    """Blocks in priority order, shuffled inside; REF before every block (five
    before the first), about every 30 minutes and at the end of the block."""
    ordered: list[dict] = []

    def ref(block: str) -> None:
        index = sum(r["series"] == "REF" for r in ordered) + 1
        ordered.append({**_ref(index), "block": block})

    for block in D_BLOCKS:
        body = [r for r in rows if r["block"] == block]
        if not body:
            continue
        random.Random(f"{order_seed}:{block}").shuffle(body)
        for _ in range(5 if not ordered else 1):
            ref(block)
        since = 0.0
        for row in body:
            ordered.append(row)
            since += d_row_seconds(row, coef, kv_blocks)
            if since >= D_REF_EVERY_S:
                ref(block)
                since = 0.0
        if ordered[-1]["series"] != "REF":
            ref(block)
    return ordered


def _ref(index: int) -> dict:
    return {
        "series": "REF",
        "condition_id": "C03",
        "set_id": "prefix_sweep_set_01",
        "pair_id": "prefix_sweep_set_01-seed-42",
        "split_group_id": "REF",
        "repeat_id": f"{index:02}",
        "seed": 42,
        "workload_id": "prefix_sweep_set_01",
        "prefix_cache_mode": "off",
        "scheduler_mode": "upstream_dynamic",
    }


def _s0() -> list[dict]:
    rows = []
    for row in build_run_matrix():
        rows.append(
            {
                **row,
                "series": "S0",
                "server_run_id": row["run_id"],
                "engine_overrides": {"max_num_seqs": 8},
            }
        )
    return rows


def _fits(row: dict, kv_blocks: int, kv_fill: float) -> str | None:
    spec = row.get("workload_spec")
    if spec is None:
        return None
    if spec.get("allow_queueing"):
        # Queueing and preemption are part of the series; only a request that
        # can never be scheduled is a planning error.
        if largest_request_blocks(spec) > kv_blocks:
            return "request larger than the KV cache"
        return None
    if kv_blocks_needed(spec) > kv_fill * kv_blocks:
        return f"needs {kv_blocks_needed(spec)} of {kv_blocks} KV blocks"
    return None


def build_local_plan(
    phase: str,
    *,
    kv_blocks: int | None,
    kv_fill: float = 0.9,
    coef: dict | None = None,
    order_seed: int = 20261028,
    max_model_len: int = 40960,
) -> dict[str, Any]:
    """Shuffled attempts of one phase with reference runs interleaved.

    ``smoke`` is one reference run (it reports the KV block count the other
    phases need); ``stress`` repeats the heaviest points for the clock check.
    """
    if phase == "smoke":
        ref = {**_ref(1), "run_id": "smoke-REF-0001", "run_order": 1}
        return {
            "workload_protocol": PROTOCOL,
            "phase": phase,
            "matrix": [ref],
            "run_count": 1,
            "series_counts": {"REF": 1},
            "dropped": [],
        }
    if kv_blocks is None:
        raise ValueError("set dataset.kv_blocks from the smoke run first")
    if phase == "A":
        body, tail = _s1() + _s2(), _s0()
    elif phase == "B":
        if coef is None:
            raise ValueError("phase B needs the phase A coefficients")
        body, tail = _s3(coef) + _s4(coef), []
    elif phase == "C":
        body, tail = _s5_within_budget(), []
    elif phase == "C_smoke":
        sizes = {("on", "upstream_dynamic"): (4,)}
        body = _s5(sizes, warmup=5, measure=lambda n: 20)[:1]
        tail = []
    elif phase in ("D", "D_smoke"):
        if coef is None:
            raise ValueError("phase D needs the phase A coefficients for its estimate")
        if phase == "D":
            body = _s6(kv_blocks, kv_fill, max_model_len) + _s7() + _s8(kv_blocks)
            body += _s5(P4_SIZES, pool_size=D_POOL)
        else:
            body = _d_smoke(kv_blocks, kv_fill, max_model_len)
        for row in body:
            row.setdefault("block", "smoke" if phase == "D_smoke" else row["series"])
        tail = []
    elif phase == "stress":
        heavy = [r for r in _s1() + _s2() if r["workload_id"] in STRESS_POINTS]
        body, tail = [dict(r) for _ in range(4) for r in heavy], []
    else:
        raise ValueError(f"unknown phase: {phase}")
    dropped, kept = [], []
    for row in body + tail:
        reason = _fits(row, kv_blocks, kv_fill)
        too_long = any(
            r["L"] + r["O"] > max_model_len
            for r in (row.get("workload_spec") or {}).get("requests", [])
        )
        if reason or too_long:
            dropped.append(
                {"workload_id": row["workload_id"], "reason": reason or "too long"}
            )
        else:
            kept.append(row)
    body = [r for r in kept if r["series"] != "S0"]
    tail = [r for r in kept if r["series"] == "S0"]
    if phase == "D":
        ordered = _order_d(body, order_seed, coef, kv_blocks)
    elif phase == "D_smoke":
        ordered = [{**_ref(1), "block": "smoke"}, *body]
    else:
        random.Random(order_seed).shuffle(body)
        random.Random(order_seed + 1).shuffle(tail)
        ordered = (
            [] if phase in ("stress", "C_smoke") else [_ref(i) for i in range(1, 6)]
        )
        ref_every = REF_EVERY_C if phase == "C" else REF_EVERY
        for i, row in enumerate(body, 1):
            ordered.append(row)
            if phase not in ("stress", "C_smoke") and i % ref_every == 0:
                index = len([r for r in ordered if r["series"] == "REF"]) + 1
                ordered.append(_ref(index))
        if phase not in ("stress", "C_smoke"):
            ordered.append(_ref(len([r for r in ordered if r["series"] == "REF"]) + 1))
        ordered += tail
    counters: dict[str, int] = {}
    for rank, row in enumerate(ordered, 1):
        counters[row["series"]] = counters.get(row["series"], 0) + 1
        row["run_id"] = f"{phase}-{row['series']}-{counters[row['series']]:04d}"
        row["run_order"] = rank
    series = {}
    for row in ordered:
        series[row["series"]] = series.get(row["series"], 0) + 1
    return {
        "workload_protocol": PROTOCOL,
        "phase": phase,
        "matrix": ordered,
        "run_count": len(ordered),
        "series_counts": series,
        "dropped": dropped,
        "kv_blocks": kv_blocks,
        "kv_fill": kv_fill,
        "coefficients": coef,
        "kv_tokens": kv_blocks * BLOCK_SIZE,
        "estimate_seconds": (
            sum(d_row_seconds(r, coef, kv_blocks) for r in ordered)
            if phase.startswith("D")
            else s5_estimate_seconds(ordered) or None
        ),
        **({"blocks": _block_table(ordered, coef, kv_blocks)} if phase == "D" else {}),
        "request_count": sum(
            len((r.get("workload_spec") or {}).get("requests", [None] * 16))
            for r in ordered
        ),
        "logical_prompt_tokens": sum(
            sum(q["L"] for q in (r.get("workload_spec") or {}).get("requests", []))
            for r in ordered
        ),
        "requested_output_tokens": sum(
            sum(q["O"] for q in (r.get("workload_spec") or {}).get("requests", []))
            for r in ordered
        ),
    }


def _block_table(rows: list[dict], coef: dict, kv_blocks: int) -> dict:
    """Attempts, points, seeds and estimated hours of every phase D block."""
    table = {}
    for block in D_BLOCKS:
        part = [r for r in rows if r["block"] == block]
        work = [r for r in part if r["series"] != "REF"]
        table[block] = {
            "attempts": len(part),
            "ref": len(part) - len(work),
            "points": len({r["workload_id"] for r in work}),
            "seeds": sorted({r["seed"] for r in work}),
            "hours": sum(d_row_seconds(r, coef, kv_blocks) for r in part) / 3600,
        }
    return table


# Heaviest prefill and decode points: a sustained load for the clock check.
STRESS_POINTS = (
    "S1-L16385-R0-k1",
    "S1-L36865-R0-k1",
    "S1-L257-R0-k64",
    "S2-L65-B160",
    "S2-L129-B64",
    "S2-L1025-B24",
)
