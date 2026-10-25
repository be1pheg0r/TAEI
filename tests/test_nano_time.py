import json

import numpy as np
import pandas as pd
import pytest

from taei.nano_time import data as nano_data
from taei.nano_time import split
from taei.nano_time.decode import registry, select
from taei.nano_time.decode.additive import Additive
from taei.nano_time.decode.multiplicative import interpolate, k_points
from taei.nano_time.fit import fit_nonneg
from taei.nano_time.prefill import PrefillModel


@pytest.mark.parametrize("method", ["regression", "optuna"])
def test_fit_coefficients_nonnegative_and_recover_positive_form(method):
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [rng.uniform(1, 100, 200), rng.uniform(1, 1e4, 200), np.ones(200)]
    )
    y = X @ np.array([0.02, 0.0, 0.5])
    coef = fit_nonneg(X, y, method, seed=0, trials=300)
    assert np.all(coef >= 0)
    rel = np.abs(X @ coef - y) / y
    assert rel.mean() < (0.01 if method == "regression" else 0.05)


def test_prefill_prediction_nonnegative_and_monotone():
    steps = pd.DataFrame(
        {
            "new": [10.0, 100, 1000],
            "quad": [1e4, 1e5, 1e6],
            "T": [0.1, 0.2, 0.9],
            "run_id": ["a", "a", "b"],
        }
    )
    model = PrefillModel(coef={"a": 1e-3, "b": 1e-7, "c": 0.05})
    pred = model.predict_steps(steps)
    assert np.all(pred >= 0)
    assert np.all(np.diff(pred) > 0)
    assert model.predict_runs(steps).to_dict() == pytest.approx(
        {"a": pred[0] + pred[1], "b": pred[2]}
    )


def test_additive_decode_step_monotone_in_sum_l_and_batch():
    method = Additive()
    method.coef = {"d0": 0.05, "d1": 1e-7, "d2": 1e-3}
    base = pd.DataFrame({"sum_l": [1000.0, 2000.0, 2000.0], "B": [2, 2, 4]})
    pred = method.predict_steps(base)
    assert np.all(pred > 0)
    assert pred[1] >= pred[0] and pred[2] >= pred[1]


def test_k_interpolation_merges_equal_p_and_clamps():
    points = k_points(pd.Series([0.1, 0.1, 0.2, 0.3]), pd.Series([2.0, 4.0, 5.0, 7.0]))
    assert points["p"].tolist() == [0.1, 0.2, 0.3]
    assert points["k"].tolist() == [3.0, 5.0, 7.0]  # median of 2 and 4
    k = interpolate(points, [0.05, 0.15, 0.3, 0.9])
    assert k.tolist() == pytest.approx([3.0, 4.0, 7.0, 7.0])


def _runs():
    rows = []
    for s in range(3):
        for cache in ("on", "off"):
            for i in range(4):
                rows.append(
                    {"run_id": f"R{s}{cache}{i}", "set_id": f"set{s}", "cache": cache}
                )
    return pd.DataFrame(rows)


def test_stratified_split_deterministic_disjoint_balanced():
    runs = _runs()
    a, b = split.stratified(runs, 42), split.stratified(runs, 42)
    assert a == b
    assert not set(a["train"]) & set(a["test"])
    assert set(a["train"]) | set(a["test"]) == set(runs["run_id"])
    tagged = runs.assign(train=runs["run_id"].isin(a["train"]))
    assert (tagged.groupby(["set_id", "cache"])["train"].sum() == 2).all()


def test_cache_split():
    sp = split.cache_off_on(_runs())
    assert all("off" in r for r in sp["train"]) and all("on" in r for r in sp["test"])


def test_registry_finds_both_methods():
    assert {"additive", "multiplicative"} <= set(registry())
    assert set(select("both")) == set(registry())
    with pytest.raises(ValueError):
        select("unknown")


def _write_run(tmp_path, fresh=314, cached=512, before=826):
    attempt = tmp_path / "R01" / "uuid"
    attempt.mkdir(parents=True)
    (attempt / "runs.json").write_text(
        json.dumps(
            {
                "run_id": "R01",
                "status": "success",
                "set_id": "s",
                "seed": 1,
                "condition_id": "C01",
                "prefix_cache_mode": "on",
                "scheduler_mode": "x",
                "metrics": {
                    "wall_seconds": 1.0,
                    "input_tokens": before,
                    "cache_hit_tokens": cached,
                    "output_tokens": 1,
                },
            }
        )
    )
    (attempt / "environment.json").write_text(
        json.dumps({"cuda_visible_devices": "G0"})
    )
    step = {
        "step_id": "step-1",
        "step_index": 1,
        "phase": "prefill",
        "step_host_start_ns": 0,
        "step_host_end_ns": 500_000_000,
        "actual_batch_size": 1,
        "contexts_before": {"q": before},
    }
    (attempt / "steps.jsonl").write_text(json.dumps(step) + "\n")
    req = {
        "step_id": "step-1",
        "request_id": "q",
        "relation": "prefill",
        "prefix_reused_tokens": cached,
        "fresh_prefill_tokens": fresh,
        "context_tokens_before": before,
    }
    (attempt / "step_requests.jsonl").write_text(json.dumps(req) + "\n")
    tel = {
        "monotonic_ns": 0,
        "gpu_devices": [
            {"uuid": "G0", "sm_clock_mhz": 1500, "temperature_c": 80, "power_w": 100},
            {"uuid": "other", "sm_clock_mhz": 900, "temperature_c": 50, "power_w": 50},
        ],
    }
    (attempt / "telemetry.jsonl").write_text(json.dumps(tel) + "\n")
    return tmp_path


def test_loader_features_and_l_equals_context(tmp_path):
    ds = nano_data.load(_write_run(tmp_path), "toy")
    req = ds.prefill_requests.iloc[0]
    assert req["L"] == 826 and req["C"] == 512
    assert req["new"] == 314 and req["quad"] == 826**2 - 512**2
    step = ds.prefill_steps.iloc[0]
    assert step["T"] == pytest.approx(0.5)
    assert step["sm_clock_mhz"] == 1500  # only the run's visible GPU
    assert ds.runs.iloc[0]["p_decode"] == pytest.approx(1 / (314 + 1))


def test_loader_rejects_inconsistent_l(tmp_path):
    with pytest.raises(ValueError):
        nano_data.load(_write_run(tmp_path, before=900), "toy")
