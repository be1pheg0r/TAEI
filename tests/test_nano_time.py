import json

import numpy as np
import pandas as pd
import pytest

from taei.nano_time import data as nano_data
from taei.nano_time import prices, split
from taei.nano_time.decode import DecodeModel
from taei.nano_time.evaluate import TRAIN_SOURCE, evaluate
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


def test_decode_step_monotone_in_sum_l_and_batch():
    method = DecodeModel(coef={"d0": 0.05, "d1": 1e-7, "d2": 1e-3})
    base = pd.DataFrame({"sum_l": [1000.0, 2000.0, 2000.0], "B": [2, 2, 4]})
    pred = method.predict_steps(base)
    assert np.all(pred > 0)
    assert pred[1] >= pred[0] and pred[2] >= pred[1]


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


def test_cache_split_and_none_without_cached_runs():
    sp = split.cache_off_on(_runs())
    assert all("off" in r for r in sp["train"]) and all("on" in r for r in sp["test"])
    assert split.cache_off_on(_runs().assign(cache="off")) is None


def test_stratified_split_by_custom_keys():
    runs = _runs()
    runs["workload"] = np.where(runs["run_id"].str[-1].isin(["0", "1"]), "P", "L")
    keys = ["set_id", "cache", "workload"]
    sp = split.stratified(runs, 0, keys)
    tagged = runs.assign(train=runs["run_id"].isin(sp["train"]))
    assert (tagged.groupby(keys)["train"].sum() == 1).all()
    # The default keys keep the old split.
    assert split.stratified(runs, 0) == split.stratified(runs, 0, ["set_id", "cache"])


def _write_run(tmp_path, rows=None, run_id="R01"):
    """One run; prefill rows: fresh, recompute, private, reused, before, produced."""
    rows = rows or [(314, 0, 512, 512, 826, 1)]
    attempt = tmp_path / run_id / "uuid"
    attempt.mkdir(parents=True)
    input_tokens = sum(r[4] for r in rows if r[5])
    (attempt / "runs.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "success",
                "set_id": "s",
                "seed": 1,
                "condition_id": "C01",
                "workload_id": "s-seed-1",
                "prefix_cache_mode": "on",
                "scheduler_mode": "x",
                "metrics": {
                    "wall_seconds": 1.0,
                    "input_tokens": input_tokens,
                    "cache_hit_tokens": 0,
                    "output_tokens": 1,
                    "requested_output_tokens": 1,
                },
            }
        )
    )
    steps, reqs = [], []
    for i, (fresh, recompute, private, reused, before, produced) in enumerate(rows):
        steps.append(
            {
                "step_id": f"step-{i}",
                "step_index": i + 1,
                "phase": "prefill",
                "step_host_start_ns": i * 10**9,
                "step_host_end_ns": i * 10**9 + 500_000_000,
                "actual_batch_size": 1,
                "contexts_before": {"q": before},
            }
        )
        reqs.append(
            {
                "step_id": f"step-{i}",
                "request_id": "q",
                "relation": "prefill",
                "fresh_prefill_tokens": fresh,
                "recompute_tokens": recompute,
                "private_kv_cached_tokens": private,
                "prefix_reused_tokens": reused,
                "context_tokens_before": before,
                "produced_tokens": produced,
            }
        )
    write = lambda name, items: (attempt / name).write_text(  # noqa: E731
        "".join(json.dumps(x) + "\n" for x in items)
    )
    (attempt / "environment.json").write_text(
        json.dumps({"cuda_visible_devices": "G0"})
    )
    gpus = [
        {
            "uuid": "G0",
            "sm_clock_mhz": 1500,
            "utilization_gpu_percent": 80,
            "power_w": 100,
        },
        {
            "uuid": "other",
            "sm_clock_mhz": 900,
            "utilization_gpu_percent": 0,
            "power_w": 50,
        },
    ]
    write("telemetry.jsonl", [{"monotonic_ns": 0, "gpu_devices": gpus}])
    write("steps.jsonl", steps)
    write("step_requests.jsonl", reqs)
    write(
        "requests.jsonl",
        [
            {
                "request_id": "q",
                "prompt_length": input_tokens,
                "actual_output_tokens": 1,
                "requested_output_tokens": 1,
                "finish_reason": "length",
                "enqueue_monotonic_ns": 0,
                "first_host_observed_monotonic_ns": len(rows) * 10**9,
                "finish_monotonic_ns": len(rows) * 10**9,
            }
        ],
    )
    write(
        "token_events.jsonl",
        [{"request_id": "q", "step_id": f"step-{len(rows) - 1}", "token_index": 0}],
    )
    return tmp_path


def test_loader_features_and_l_equals_context(tmp_path):
    ds = nano_data.load(_write_run(tmp_path), "toy")
    req = ds.prefill_requests.iloc[0]
    assert req["L"] == 826 and req["C"] == 512
    assert req["new"] == 314 and req["quad"] == 826**2 - 512**2
    assert ds.prefill_steps.iloc[0]["T"] == pytest.approx(0.5)
    run = ds.runs.iloc[0]
    assert run["I"] == 826 and run["O"] == 1 and run["lost_tokens"] == 0
    assert ds.requests.iloc[0]["first_step"] == 1
    assert run["gpu_sm_mhz"] == 1500  # only the run's visible GPU


def test_loader_reads_chunked_prefill_and_recompute(tmp_path):
    # Two chunks of a 19457-token prompt, then a recompute of 16129 tokens.
    rows = [
        (16384, 0, 0, 0, 19457, 0),
        (3073, 0, 16384, 0, 19457, 1),
        (0, 16129, 0, 0, 16129, 1),
    ]
    reqs = nano_data.load(_write_run(tmp_path, rows), "toy").prefill_requests
    assert reqs["L"].tolist() == [16384, 19457, 16129]
    assert reqs["C"].tolist() == [0, 16384, 0]
    assert reqs["new"].tolist() == [16384, 3073, 16129]


def test_loader_rejects_inconsistent_l(tmp_path):
    with pytest.raises(ValueError):
        nano_data.load(_write_run(tmp_path, [(314, 0, 512, 512, 900, 1)]), "toy")


def _synthetic_dataset(model="toy", n_sets=2, scale=1.0) -> nano_data.Dataset:
    rng = np.random.default_rng(1)
    runs, pf, reqs, dc, rq = [], [], [], [], []
    for s in range(n_sets):
        for cache in ("on", "off"):
            for rep in range(2):
                run_id = f"R{s}{cache}{rep}"
                prompt = 1000.0 * (s + 1)
                cached = 400.0 if cache == "on" else 0.0
                new = prompt - cached
                t_pf = scale * (1e-4 * new + 0.02)
                steps_dc = 50
                sum_l = prompt + np.arange(steps_dc)
                t_dc = scale * (0.05 + 1e-6 * sum_l) * rng.uniform(0.95, 1.05, steps_dc)
                pf.append(
                    {
                        "run_id": run_id,
                        "step_id": "s1",
                        "step_index": 1,
                        "new": new,
                        "quad": prompt**2 - cached**2,
                        "T": t_pf,
                        "B": 1,
                        "model": model,
                    }
                )
                reqs.append(
                    {
                        "run_id": run_id,
                        "step_id": "s1",
                        "request_id": "q",
                        "L": prompt,
                        "C": cached,
                        "R": cached / prompt,
                        "new": new,
                        "quad": prompt**2 - cached**2,
                        "model": model,
                    }
                )
                dc.append(
                    pd.DataFrame(
                        {
                            "run_id": run_id,
                            "step_id": [f"d{i}" for i in range(steps_dc)],
                            "step_index": np.arange(steps_dc) + 2,
                            "B": 1,
                            "sum_l": sum_l,
                            "T": t_dc,
                            "model": model,
                        }
                    )
                )
                first_ns = int(t_pf * 1e9)
                rq.append(
                    {
                        "run_id": run_id,
                        "request_id": "q",
                        "prompt": prompt,
                        "O": steps_dc + 1,
                        "O_requested": steps_dc + 1,
                        "finish_reason": "length",
                        "enqueue_ns": 0,
                        "first_ns": first_ns,
                        "finish_ns": first_ns + int(t_dc.sum() * 1e9),
                        "first_step": 1,
                        "last_step": steps_dc + 1,
                        "model": model,
                    }
                )
                runs.append(
                    {
                        "run_id": run_id,
                        "model": model,
                        "set_id": f"set{s}",
                        "seed": rep,
                        "condition_id": "C",
                        "workload": f"set{s}",
                        "cache": cache,
                        "scheduler": "x",
                        "wall_seconds": t_pf + t_dc.sum(),
                        "I": prompt,
                        "O": steps_dc + 1,
                        "O_requested": steps_dc + 1,
                        "lost_tokens": 0,
                        "steps_seconds": t_pf + t_dc.sum(),
                        "prefill_seconds": t_pf,
                        "decode_seconds": t_dc.sum(),
                        "n_prefill_steps": 1,
                        "n_decode_steps": steps_dc,
                        "L_mean": prompt,
                        "R_run": cached / prompt,
                    }
                )
    return nano_data.Dataset(
        pd.DataFrame(runs),
        pd.DataFrame(pf),
        pd.DataFrame(reqs),
        pd.concat(dc, ignore_index=True),
        pd.DataFrame(rq),
    )


def _evaluation():
    train = _synthetic_dataset()
    other = _synthetic_dataset("big", n_sets=1, scale=3.0)
    return evaluate(train, {"big model": other}, "regression", 0, {})


def _document():
    from taei.nano_time import report_data

    ev = _evaluation()
    meta = {"fit": "regression", "seed": 0}
    model = report_data.build(ev, "toy", meta, ["note"])
    return report_data.document(meta, [model, model])


def test_token_prices_add_up_to_predicted_total():
    ev = _evaluation()
    p = prices.run_prices(ev).merge(ev.runs[["source", "run_id", "pred"]])
    total = (p["a_model"] * p["I"] + p["b_model"] * p["O"]).to_numpy()
    assert total == pytest.approx(p["pred"].to_numpy())
    budget = (p["pred"] / p["wall_seconds"]).to_numpy()
    assert p["budget_model"].to_numpy() == pytest.approx(budget)
    # Synthetic wall is the sum of steps, so the measured budget is exactly 1.
    assert p["budget_meas"].to_numpy() == pytest.approx(np.ones(len(p)))
    lines = prices.lines(ev)
    by_r = [line["itps_max"] for line in lines["by_R"]]
    assert by_r == sorted(by_r)  # more reuse -> cheaper input tokens


def test_request_ttft_and_tpot_follow_the_trace():
    ev = _evaluation()
    reqs = prices.requests(ev)
    pf = ev.prefill_steps.set_index(["source", "run_id"])["pred"]
    dc = ev.decode_steps.groupby(["source", "run_id"])["pred"].sum()
    key = list(zip(reqs["source"], reqs["run_id"], strict=True))
    assert reqs["ttft_hat"].to_numpy() == pytest.approx(pf.reindex(key).to_numpy())
    tpot = dc.reindex(key).to_numpy() / (reqs["O"] - 1).to_numpy()
    assert reqs["tpot_hat"].to_numpy() == pytest.approx(tpot)
    assert reqs["ttft"].to_numpy() == pytest.approx(
        ev.prefill_steps.set_index(["source", "run_id"])["T"].reindex(key).to_numpy(),
        rel=1e-6,
    )


def test_report_document_is_json_and_generic():
    from taei.nano_time import report_data

    doc = _document()
    json.dumps(doc, allow_nan=False)  # no NaN or numpy values left
    assert doc["schema_version"] == report_data.SCHEMA_VERSION == 2
    model = doc["models"][0]
    assert [s["role"] for s in model["sources"]] == ["train", "transfer"]
    assert len(model["math"]["formulas"]) == 6 and len(model["pipeline"]) == 6
    assert {"by_R", "by_B", "ref"} <= set(model["prices"]["lines"])
    assert model["decode_binned"]["B"] and "sm_clock_mhz" not in model["decode_binned"]
    forms = model["prefill_forms"]
    assert forms["table"] and all(r["set_cv_mean"] >= 0 for r in forms["table"])
    total = [r for r in model["metrics"] if r["what"] == "T_total per run"]
    assert total and all("mean_signed" in r for r in total)
    for row in model["metrics"]:
        assert row["mean_abs"] >= 0


def test_render_report_embeds_data_safely_and_index_reads_v1_and_v2(tmp_path):
    from taei.nano_time import report

    doc = _document()
    doc["models"][0]["notes"] = ["</script><b>x</b>"]
    page = report.render_report(doc)
    assert "__REPORT_DATA__" not in page and "</script><b>" not in page
    assert page.count("<title>") == 1
    with pytest.raises(ValueError):
        report.render_report({**doc, "schema_version": 1})
    run = tmp_path / "20260101_000000"
    run.mkdir()
    (run / "time_model.json").write_text(json.dumps(doc), encoding="utf-8")
    report.write_report(run / "time_model.json")
    old = tmp_path / "20250101_000000"
    old.mkdir()
    v1 = {
        "schema_version": 1,
        "meta": {"fit": "regression", "decode": "both"},
        "sources": [{"id": TRAIN_SOURCE, "label": "toy", "role": "train"}],
        "metrics": [
            {"what": "multiplicative: T_total per run", "part": "test", "mean_abs": 0.2}
        ],
    }
    (old / "time_model.json").write_text(json.dumps(v1), encoding="utf-8")
    index = report.write_index(tmp_path).read_text(encoding="utf-8")
    assert "20260101_000000/report.html" in index
    assert "20250101_000000/report.html" in index
    entry = report.index_entry(old / "time_model.json", tmp_path)
    assert entry["models"][0]["test"] == {"multiplicative": 0.2}


def test_run_factor_fit_recovers_shape_and_factors():
    from taei.nano_time.prefill_forms import fit_with_run_factor

    rng = np.random.default_rng(0)
    rows = []
    factors = {f"R{i}": f for i, f in enumerate([0.8, 1.0, 1.25, 0.9, 1.1])}
    for run, factor in factors.items():
        for _ in range(4):
            new, cached, b = (
                rng.uniform(2e3, 1.5e4),
                rng.uniform(0, 1e4),
                rng.integers(1, 9),
            )
            t = factor * (1e-4 * new + 1e-5 * cached + 0.02 * b)
            rows.append(
                {"run_id": run, "N": new, "sumC": cached, "B": float(b), "T": t}
            )
    steps = pd.DataFrame(rows)
    coef, fitted = fit_with_run_factor(steps, ["N", "sumC", "B"])
    scale = coef[0] / 1e-4  # shape is identified up to the geometric-mean factor
    assert coef == pytest.approx(scale * np.array([1e-4, 1e-5, 0.02]), rel=1e-3)
    ratio = fitted / pd.Series(factors)
    assert ratio.std() / ratio.mean() < 1e-3
