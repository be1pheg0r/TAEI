import json

import numpy as np
import pandas as pd
import pytest

from taei.nano_time import closed_loop, economics
from taei.nano_time import decode_variants as dv
from taei.nano_time import system as sm
from taei.nano_time.data import unique_tokens
from taei.nano_time.fit import fit_nonneg

PREFILL = {"a": 2e-5, "b": 4e-9, "c": 0.01}


def _step(d1=2e-7, F=0.0, kappa=1.0):
    coef = {"d0": 0.01, "d1": d1, "d2": 3e-5, "F": F, "kappa": kappa}
    return sm.StepModel(PREFILL, dv.DecodeVariant("kappa", coef))


def _pool(L, out, family=None, prefix=0, cache="off"):
    spec = [
        {"L": int(l_), "O": int(o_), "family": family, "prefix_blocks": prefix}
        for l_, o_ in zip(L, out, strict=True)
    ]
    return sm.Pool.from_spec(spec, cache)


def test_unique_tokens_hold_the_longest_prefix_of_a_family_once():
    ctx = {"a": 1000, "b": 1000, "c": 500}
    assert unique_tokens(ctx, {}) == 2500
    shared = {"a": ("f0", 512), "b": ("f0", 256)}
    assert unique_tokens(ctx, shared) == 2500 - 256


def test_kv_knee_of_a_uniform_pool_with_and_without_the_cache():
    eng = sm.Engine()
    # O = 1: no growth; 4 own blocks each, a new request needs 4 (2 held free).
    off = _pool([1000] * 50, [1] * 50)
    assert sm.kv_knee(sm.running(off, eng, draws=50), eng) == pytest.approx(46.0)
    # Cache: 2 prefix blocks held once, 2 own; a new one needs 4 − 2 = 2.
    on = _pool([1000] * 50, [1] * 50, family="f0", prefix=2, cache="on")
    assert sm.kv_knee(sm.running(on, eng, draws=50), eng) == pytest.approx(91.5)


def test_dynamic_rps_little_identity_and_queue_above_the_knee():
    eng = sm.Engine()
    pool = _pool([500, 1500] * 40, [10, 30] * 40)
    run = sm.running(pool, eng, draws=400)
    step = _step()
    p = sm.predict_dynamic(pool, eng, step, 8, run)
    o = pool.out.mean()
    T_dc = step.decode.predict(8, run.at(8)["sum_l"], run.at(8)["sum_u"])
    assert p["rps"] == pytest.approx(1 / (p["T_pf"] + (o - 1) * T_dc / 8))
    assert p["tpot"] * (o - 1) * p["rps"] == pytest.approx(8)
    assert p["queue"] == 0 and not p["above"]
    knee = sm.kv_knee(run, eng)
    hi1, hi2 = (sm.predict_dynamic(pool, eng, step, knee + d, run) for d in (5, 25))
    assert hi1["rps"] == pytest.approx(hi2["rps"])
    assert hi2["queue"] == pytest.approx((knee + 25 - hi2["B"]) / hi2["rps"])
    assert hi2["ttft_p50"] - hi1["ttft_p50"] == pytest.approx(20 / hi2["rps"])


def test_cohort_ttft_p90_is_the_prefill_of_the_whole_cohort():
    eng = sm.Engine()
    pool = _pool([2048] * 64, [128] * 64)
    step = _step()
    p = sm.predict_dynamic(pool, eng, step, 16, sm.running(pool, eng, draws=200))
    one = step.prefill_step(8 * 2048, 8 * 2048**2)  # 16 384 tokens: 8 per step
    assert p["branch"] == "cohort"
    assert p["ttft_p50"] == pytest.approx(one)
    assert p["ttft_p90"] == pytest.approx(2 * one)


def test_fixed_batch_wave_of_equal_outputs():
    eng = sm.Engine()
    pool = _pool([1000] * 300, [50] * 300)
    step = _step(d1=0.0)  # a constant decode step
    p = sm.predict_fixed(pool, eng, step, 8)
    wave = step.prefill_step(8 * 1000, 8 * 1000**2) + 49 * step.decode.predict(8, 0)
    assert p["rps"] == pytest.approx(8 / wave)
    assert p["B"] == 8


def test_budget_point_lies_on_the_line_and_on_the_ray():
    eng = sm.Engine()
    step = _step()
    x, y = sm.budget_point(2000, 0.5, 100, 4, step, eng)
    new = 2000 - 3 * 256  # 3 whole prefix blocks
    assert y / x == pytest.approx(100 / new)
    pool = _pool([2000], [100], family="f0", prefix=3, cache="on")
    p = sm.predict_dynamic(pool, eng, step, 4, sm.running(pool, eng, draws=200))
    assert p["a_tilde"] * x + p["b_tilde"] * y == pytest.approx(1)


def test_economics_revenue_payback_and_slo():
    prices = {"input": 0.1, "cached_input": 0.05, "output": 0.2}
    gpu = {"price": 600.0, "cost_per_hour": 0.1}
    r = economics.point(prices, gpu, 1000.0, 500.0, 100.0)
    assert r["revenue_per_hour"] == pytest.approx(3600 * (100 + 25 + 20) / 1e6)
    assert r["payback_hours"] == pytest.approx(600 / (0.522 - 0.1))
    assert economics.point(prices, gpu, 0, 0, 0)["payback_hours"] is None
    rows = [
        {"N": 1, "ttft_p90": 0.1, "tpot": 0.01},
        {"N": 4, "ttft_p90": 0.4, "tpot": 0.02},
        {"N": 8, "ttft_p90": 2.0, "tpot": 0.02},
    ]
    assert economics.max_n_under_slo(rows, {"ttft_p90": 0.5, "tpot": 0.03})["N"] == 4
    assert economics.max_n_under_slo(rows, {"ttft_p90": 0.05, "tpot": 0.03}) is None


def _variant_steps(F=0.012, kappa=0.5):
    rng = np.random.default_rng(1)
    d0, d1, d2 = 0.01, 3e-7, 2.5e-5
    rows, points = [], []
    for series, n_runs, shared in (
        ("S2", 40, False),
        ("S6", 20, False),
        ("S7", 30, True),
    ):
        for i in range(n_runs):
            run = f"D-{series}-{i:04d}"
            B = int(rng.choice([1, 2, 8, 16, 32]))
            ell = float(rng.uniform(500, 4000))
            share = rng.uniform(0.25, 0.8) if shared else 0.0
            points.append({"run_id": run, "ell": ell, "R_target": share, "families": 1})
            for k in range(8):
                sum_l = B * (ell + k)
                sum_u = sum_l * (1 - share) if B > 1 else sum_l
                lin = d0 + d1 * (sum_u + kappa * (sum_l - sum_u)) + d2 * B
                T = max(F, lin) * (1 + rng.normal(0, 0.005))
                rows.append(
                    {"run_id": run, "step_index": k, "T": T, "B": B}
                    | {"sum_l": sum_l, "sum_u": sum_u}
                )
    return pd.DataFrame(rows), pd.DataFrame(points)


def test_decode_variants_recover_the_floor_and_kappa():
    steps, points = _variant_steps()
    fit = dv.fit_all(steps, points, {}, train_series=("S2", "S6"), test_series=("S7",))
    floor = fit.variants["floor"].coef
    assert floor["F"] == pytest.approx(0.012, rel=0.1)
    assert fit.variants["kappa"].coef["kappa"] == pytest.approx(0.5, abs=0.05)
    assert fit.choice == "kappa"
    base = steps[steps["run_id"].str.contains("S2|S6")]
    X = np.column_stack([np.ones(len(base)), base["sum_l"], base["B"]])
    d = fit_nonneg(X, base["T"], "regression")
    assert [fit.variants["base"].coef[k] for k in ("d0", "d1", "d2")] == pytest.approx(
        d
    )


def test_closed_loop_counts_completions_inside_the_window(tmp_path):
    lo, hi = 1_000_000_000, 3_000_000_000
    run = {
        "run_id": "D-S8-0001",
        "series": "S8",
        "block": "S8-W1",
        "status": "success",
        "scheduler_mode": "upstream_dynamic",
        "prefix_cache_mode": "off",
        "seed": 42,
        "closed_loop": {"N": 2, "window_start_ns": lo, "window_end_ns": hi},
        "workload_spec": {"requests": [{"L": 100, "O": 3}]},
    }
    (tmp_path / "runs.json").write_text(json.dumps(run))
    reqs = [
        # finishes at the window start: a warm-up request, outside
        (0, 400_000_000, lo),
        (500_000_000, 1_200_000_000, 2_000_000_000),
        (1_500_000_000, 2_100_000_000, hi),
    ]
    with (tmp_path / "requests.jsonl").open("w") as f:
        for enq, first, fin in reqs:
            f.write(
                json.dumps(
                    {
                        "input_token_count": 100,
                        "prefix_reused_tokens": 0,
                        "output_token_count": 3,
                        "enqueue_monotonic_ns": enq,
                        "first_scheduled_monotonic_ns": enq,
                        "first_host_observed_monotonic_ns": first,
                        "finish_monotonic_ns": fin,
                    }
                )
                + "\n"
            )
    (tmp_path / "steps.jsonl").write_text("")
    row = closed_loop.attempt_row(run, tmp_path)
    assert row["completed"] == 2
    assert row["rps"] == pytest.approx(1.0)
    assert row["otps"] == pytest.approx(3.0)
    assert row["ttft_p50"] == pytest.approx(0.65)
