"""Request-count scaling experiment contracts."""

from collections import Counter

import pytest

from taei.nano_dataset.orchestrator import build_plan
from taei.nano_dataset.workload import build_workload

PROTOCOL = "qwen3_8b_request_scaling_v1"


def test_scaling_plan_and_paired_workloads():
    from jsonschema import Draft202012Validator

    from taei.nano_dataset.schema import DATASET_SCHEMA

    plan = build_plan(workload_protocol=PROTOCOL, context_cap=14000)
    assert plan["run_count"] == 30
    assert plan["request_count"] == 186
    assert plan["requested_output_tokens"] == 186000
    assert Counter(
        (row["request_count"], row["prefix_cache_mode"]) for row in plan["matrix"]
    ) == {(n, cache): 3 for n in (1, 2, 4, 8, 16) for cache in ("on", "off")}
    assert all(row["scheduler_mode"] == "upstream_dynamic" for row in plan["matrix"])
    workloads = {}
    for row in plan["matrix"]:
        workload = build_workload(row["workload_id"], row["seed"])
        Draft202012Validator(DATASET_SCHEMA["$defs"]["workload"]).validate(workload)
        requests = workload["requests"]
        assert len(requests) == row["request_count"]
        assert all(len(r["input_token_ids"]) == r["prompt_length"] for r in requests)
        assert all(
            r["requested_output_tokens"] == 1000 and r["ignore_eos"] for r in requests
        )
        assert all(
            r["prime_input_token_ids"][:-1] == r["input_token_ids"][:1024]
            for r in requests
        )
        key = (row["seed"], row["request_count"])
        if key in workloads:
            assert workloads[key] == workload
        workloads[key] = workload
    for seed in (42, 43, 44):
        full = workloads[(seed, 16)]["requests"]
        assert sorted(r["prompt_length"] for r in full) == [
            3000,
            3667,
            4333,
            5000,
            5667,
            6333,
            7000,
            7667,
            8333,
            9000,
            9667,
            10333,
            11000,
            11667,
            12333,
            13000,
        ]
        for count in (1, 2, 4, 8):
            assert workloads[(seed, count)]["requests"] == full[:count]


def test_scaling_rejects_insufficient_context():
    with pytest.raises(ValueError, match="14000"):
        build_plan(workload_protocol=PROTOCOL, context_cap=13999)


def test_scaling_hydra_plan_runs_without_gpu_or_model(tmp_path):
    import json
    import subprocess
    import sys

    output = tmp_path / "plan"
    result = subprocess.run(
        [
            sys.executable,
            "experiments/nano_vllm_qwen3_8b_long_context/run.py",
            "--config-name",
            "request_scaling",
            "mode=plan",
            f"hydra.run.dir={output}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads((output / "plan.json").read_text())
    assert plan["run_count"] == 30 and plan["request_count"] == 186
    assert plan["schedule"]["donor_requests_on"] == 93
    assert plan["schedule"]["donor_requests_off"] == 0
    assert not (output / "model-download.json").exists()


@pytest.mark.parametrize("cache,expected", [("on", 1), ("off", 0)])
def test_attempt_primes_only_cache_on(tmp_path, monkeypatch, cache, expected):
    from taei.nano_dataset import engine
    from taei.nano_dataset.orchestrator import run_attempt

    received = []

    class Backend:
        def __init__(self, *args):
            pass

        def prepare(self):
            return {}

        def prime_requests(self, requests):
            received.append(len(requests))
            raise RuntimeError("stop after priming")

        def close(self):
            pass

    monkeypatch.setattr(engine, "NanoVLLMBackend", Backend)
    row = next(
        row
        for row in build_plan(workload_protocol=PROTOCOL)["matrix"]
        if row["request_count"] == 1 and row["prefix_cache_mode"] == cache
    )
    result = run_attempt(row, str(tmp_path / "model"), {}, tmp_path / "runs")
    assert result["status"] == "failed"
    assert received == [expected]
