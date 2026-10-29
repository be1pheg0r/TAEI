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
            r["prime_input_token_ids"][:-1]
            == r["input_token_ids"][: r["prefix_token_count"]]
            and r["prefix_token_count"] % 256 == 0
            and 0.10 <= r["target_cache_hit_ratio_on"] <= 0.95
            and r["target_cache_hit_ratio_on"]
            == r["prefix_token_count"] / r["prompt_length"]
            for r in requests
        )
        key = (row["seed"], row["request_count"])
        if key in workloads:
            assert workloads[key] == workload
        workloads[key] = workload
    for seed in (42, 43, 44):
        full = workloads[(seed, 16)]["requests"]
        assert full[0]["target_cache_hit_ratio_on"] < 0.13
        assert full[1]["target_cache_hit_ratio_on"] > 0.90
        assert sum(r["target_cache_hit_ratio_on"] >= 0.5 for r in full) >= 10
        longest = max(full, key=lambda r: r["prefix_token_count"])
        assert all(
            r["input_token_ids"][: r["prefix_token_count"]]
            == longest["input_token_ids"][: r["prefix_token_count"]]
            for r in full
        )
        assert max(r["prefix_token_count"] for r in full) // 256 < 218
        assert len({r["prefix_token_count"] for r in full}) > 8
        assert workloads[(seed, 16)]["summary"]["cache_hit_tokens_on_targets"] == [
            r["prefix_token_count"] for r in full
        ]
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


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_shared_prefixes_prime_and_hit_with_real_block_manager(seed):
    from types import SimpleNamespace

    blocks = pytest.importorskip("nanovllm.engine.block_manager")
    sequence = pytest.importorskip("nanovllm.engine.sequence")
    pytest.importorskip("torch")
    from taei.nano_dataset.engine import NanoVLLMBackend

    class Engine:
        def __init__(self):
            self.scheduler = SimpleNamespace(
                block_manager=blocks.BlockManager(218, 256), waiting=[]
            )

        def add_request(self, prompt, sampling):
            self.scheduler.waiting.append(sequence.Sequence(prompt))

        def generate(self, prompts, sampling, **kwargs):
            manager = self.scheduler.block_manager
            for prompt, params in zip(prompts, sampling, strict=True):
                seq = sequence.Sequence(prompt, params)
                cached = manager.can_allocate(seq)
                assert cached >= 0
                manager.allocate(seq, cached)
                seq.num_scheduled_tokens = len(seq) - seq.num_cached_tokens
                manager.hash_blocks(seq)
                manager.deallocate(seq)

    requests = build_workload("qwen3_8b_request_scaling_n16", seed)["requests"]
    backend = NanoVLLMBackend.__new__(NanoVLLMBackend)
    backend.engine = Engine()
    backend._originals = []
    backend._primed_blocks_by_request = {}
    backend._prompt_cache_hits = {}
    backend.prefix_cache_mode = "on"
    backend.measurement_profile = "primary"
    backend.seed = seed
    priming = backend.prime_requests(requests)
    assert priming["retained_prefix_blocks"] == max(
        r["prefix_token_count"] // 256 for r in requests
    )
    backend._metric_request_ids = iter(r["request_id"] for r in requests)
    manager = backend.engine.scheduler.block_manager
    for request in requests:
        backend.engine.add_request(request["input_token_ids"], None)
        seq = backend.engine.scheduler.waiting.pop()
        cached = manager.can_allocate(seq)
        assert cached * 256 == request["expected_cache_hit_tokens_on"]
        manager.allocate(seq, cached)
        manager.deallocate(seq)
    assert not backend._primed_blocks_by_request
    assert len(manager.free_block_ids) == 218


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
