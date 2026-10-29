"""Проверяет план локальной сетки Qwen3-0.6B на CPU.

Каждая точка помещается в KV-кэш, R задаётся целым числом блоков общего
префикса, донор префикса один на семью и идёт первым по приходу, а доля prefill
в смесях S3 по формуле совпадает с целевой.
"""

from __future__ import annotations

import pytest

from taei.nano_dataset.local_grid import (
    BLOCK_SIZE,
    build_local_plan,
    build_spec_workload,
    kv_blocks_needed,
    output_for_share,
    prefill_seconds,
)

COEF = {"a": 5e-5, "b": 1e-9, "c": 0.01, "d0": 0.015, "d1": 1e-8, "d2": 2e-4}


@pytest.mark.parametrize("phase", ["A", "B", "stress"])
def test_planned_points_fit_the_kv_cache(phase):
    plan = build_local_plan(phase, kv_blocks=170, coef=COEF)
    assert plan["run_count"] == len({r["run_id"] for r in plan["matrix"]})
    for row in plan["matrix"]:
        spec = row.get("workload_spec")
        if spec and not spec.get("allow_queueing"):
            assert kv_blocks_needed(spec) <= 0.9 * 170


def test_phase_a_interleaves_reference_runs_and_keeps_server_matrix_last():
    plan = build_local_plan("A", kv_blocks=170)
    series = [r["series"] for r in plan["matrix"]]
    assert series[:5] == ["REF"] * 5
    assert series[-48:] == ["S0"] * 48
    assert all(
        r["engine_overrides"] == {"max_num_seqs": 8} for r in plan["matrix"][-48:]
    )
    assert series.count("REF") >= 5 + (len(series) - 48 - 5) // 26


def test_shared_prefix_gives_reuse_in_whole_blocks_with_one_donor():
    plan = build_local_plan("A", kv_blocks=170)
    row = next(r for r in plan["matrix"] if r["workload_id"] == "S1-L1025-p2-k4")
    workload = build_spec_workload(
        row["workload_id"], row["seed"], row["workload_spec"]
    )
    requests = workload["requests"]
    prefix = 2 * BLOCK_SIZE
    assert all(r["prefix_token_count"] == prefix for r in requests)
    assert len({tuple(r["input_token_ids"][:prefix]) for r in requests}) == 1
    assert len({tuple(r["input_token_ids"][prefix:]) for r in requests}) == 4
    donors = [r for r in requests if r["prime_input_token_ids"]]
    assert len(donors) == 1
    assert (
        donors[0]["prime_input_token_ids"][:-1]
        == requests[0]["input_token_ids"][:prefix]
    )
    again = build_spec_workload(row["workload_id"], row["seed"], row["workload_spec"])
    assert again["requests"] == requests


def test_s3_output_length_sets_the_prefill_share():
    for n, L, share in ((1, 1025, 0.3), (8, 513, 0.5), (4, 4097, 0.7)):
        out = output_for_share(COEF, n, L, 0.0, share)
        decode = sum(
            COEF["d0"] + COEF["d1"] * n * (L + i) + COEF["d2"] * n for i in range(out)
        )
        prefill = prefill_seconds(COEF, n, L, 0.0)
        assert prefill / (prefill + decode) == pytest.approx(share, abs=0.02)


def test_s4_arrivals_are_increasing_and_donor_comes_first_in_its_family():
    plan = build_local_plan("B", kv_blocks=170, coef=COEF)
    row = next(r for r in plan["matrix"] if r["series"] == "S4")
    workload = build_spec_workload(
        row["workload_id"], row["seed"], row["workload_spec"]
    )
    by_arrival = sorted(workload["requests"], key=lambda r: r["arrival_index"])
    offsets = [r["planned_arrival_offset_ns"] for r in by_arrival]
    assert offsets == sorted(offsets) and offsets[0] > 0
    seen = set()
    for request in by_arrival:
        family = request["prefix_family_id"]
        assert bool(request["prime_input_token_ids"]) == (family not in seen)
        seen.add(family)


def test_phase_c_closed_loop_pools_share_one_order_and_fit_the_budget():
    from taei.nano_dataset.local_grid import (
        S5_BUDGET_S,
        S5_SIZES,
        S5_WARMUP,
        REF_EVERY_C,
        s5_estimate_seconds,
        s5_measure,
    )

    plan = build_local_plan("C", kv_blocks=186)
    s5 = [r for r in plan["matrix"] if r["series"] == "S5"]
    assert len(s5) == 3 * sum(len(ns) for ns in S5_SIZES.values())
    assert s5_estimate_seconds(plan["matrix"]) <= S5_BUDGET_S
    # REF every REF_EVERY_C attempts after the five leading ones.
    series = [r["series"] for r in plan["matrix"]]
    assert series[:5] == ["REF"] * 5
    assert series[5 + REF_EVERY_C] == "REF" and series[-1] == "REF"
    by_seed = {}
    for row in s5:
        spec = row["workload_spec"]
        loop = spec["closed_loop"]
        n, m = loop["N"], loop["measure"]
        assert m == s5_measure(n) and loop["warmup"] == S5_WARMUP
        assert len(spec["requests"]) == n + S5_WARMUP + m - 1
        # Every point is a prefix of one pool per seed: points differ only in N.
        pool = by_seed.setdefault(row["seed"], spec["requests"])
        shorter, longer = sorted((pool, spec["requests"]), key=len)
        assert longer[: len(shorter)] == shorter
        if len(spec["requests"]) > len(pool):
            by_seed[row["seed"]] = spec["requests"]
        donors = [q["family"] for q in spec["requests"] if q["donor"]]
        assert (
            len(donors)
            == len(set(donors))
            == len({q["family"] for q in spec["requests"]})
        )
        assert (row["fixed_batch_size"] == 8) == (
            row["scheduler_mode"] == "fixed_batch"
        )
    workload = build_spec_workload(
        "S5-N4-on-upstream_dynamic", 42, s5[0]["workload_spec"]
    )
    assert all(
        q["closed_loop"] == s5[0]["workload_spec"]["closed_loop"]
        for q in workload["requests"]
    )


def test_phase_d_blocks_grids_and_budget():
    from taei.nano_dataset.local_grid import (
        D_BLOCKS,
        D_DECODE_STEPS,
        S8_EXTRA,
        S8_GRID,
        S8_REPEATS,
        _s5_pool,
        kv_concurrency,
    )

    kv = 186
    plan = build_local_plan("D", kv_blocks=kv, kv_fill=0.95, coef=COEF)
    rows = plan["matrix"]
    assert plan["run_count"] == len({r["run_id"] for r in rows})
    # Blocks in priority order, each opened and closed by a REF.
    order = [r["block"] for r in rows]
    assert [b for i, b in enumerate(order) if i == 0 or order[i - 1] != b] == list(
        D_BLOCKS
    )
    assert [r["series"] for r in rows[:5]] == ["REF"] * 5
    for block in D_BLOCKS:
        part = [r for r in rows if r["block"] == block]
        assert part[0]["series"] == part[-1]["series"] == "REF"
    work = [r for r in rows if r["series"] != "REF"]
    for row in work:
        spec = row["workload_spec"]
        if row["series"] in ("S6", "S7"):
            # One batch without preemption: every block of every request fits.
            assert kv_blocks_needed(spec) <= 0.95 * kv
            assert {q["O"] for q in spec["requests"]} == {D_DECODE_STEPS}
    s6 = {r["workload_id"] for r in work if r["series"] == "S6"}
    assert {"S6-B16-S44k", "S6-B96-Smax", "S6-B8-Smax-mixed"} <= s6
    s7 = {r["workload_id"] for r in work if r["series"] == "S7"}
    assert {"S7-B8-L2049-R0-on", "S7-B8-L2049-R0-off"} <= s7
    assert sum(r["series"] == "S7" and r["prefix_cache_mode"] == "off" for r in work)
    # S8: at least 4 sizes up to the KV knee and 2 above it.
    for row in (r for r in work if r["series"] == "S8"):
        point = row["workload_spec"]["point"]
        assert row["seed"] in (42, 43, 44)[: S8_REPEATS[point["workload"]]]
        if row["scheduler_mode"] == "fixed_batch":
            continue
        sizes = {
            r["workload_spec"]["point"]["N"]
            for r in work
            if r["series"] == "S8"
            and r["workload_spec"]["point"]["workload"] == point["workload"]
            and r["prefix_cache_mode"] == row["prefix_cache_mode"]
            and r["scheduler_mode"] == row["scheduler_mode"]
        }
        knee = point["kv_concurrency"]
        assert len([n for n in sizes if n > knee]) == 2
        assert len([n for n in sizes if n <= knee]) >= 4 - (point["workload"] == "W2")
        assert sizes <= set(S8_GRID) | set(S8_EXTRA)
        if point["workload"] == "W2":
            assert 1 not in sizes
            assert point["warmup"] == (5 if point["N"] <= 4 else 20)
        loop = row["workload_spec"]["closed_loop"]
        assert len(row["workload_spec"]["requests"]) == (
            loop["N"] + loop["warmup"] + loop["measure"] - 1
        )
    w3 = next(r for r in work if r["workload_id"].startswith("S8-W3-N1-on"))
    pool = w3["workload_spec"]["requests"]
    assert kv_concurrency(pool, "on", kv) < 8
    # P4 extends the S5 pools: the first S5 requests are the same.
    p4 = [r for r in work if r["series"] == "S5"]
    assert {r["workload_spec"]["closed_loop"]["N"] for r in p4} == {8, 24, 48, 64}
    for row in p4:
        reqs = row["workload_spec"]["requests"]
        base = _s5_pool(row["seed"])
        assert reqs[: len(base)] == base[: len(reqs)]
    assert plan["estimate_seconds"] < 10 * 3600
    assert set(plan["blocks"]) == set(D_BLOCKS)


def test_phase_d_smoke_has_a_time_limited_closed_loop():
    plan = build_local_plan("D_smoke", kv_blocks=186, kv_fill=0.95, coef=COEF)
    assert [r["series"] for r in plan["matrix"]] == ["REF", "S6", "S7", "S8"]
    loop = plan["matrix"][-1]["workload_spec"]["closed_loop"]
    assert loop["max_window_s"] and loop["min_measure"] < loop["measure"]
