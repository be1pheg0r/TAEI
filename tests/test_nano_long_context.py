"""Contracts for the Qwen3-8B long-context paired workload."""

from taei.nano_dataset.workload import (
    build_long_context_plan,
    build_long_context_workload,
    build_workload,
)


def test_prefill_only_runs_extend_original_matrix_with_matching_inputs():
    plan = build_long_context_plan()
    original = [row for row in plan["matrix"] if row["run_id"].startswith("L")]
    prefill = [row for row in plan["matrix"] if row["run_id"].startswith("P")]
    assert len(original) == len(prefill) == 48
    assert plan["request_count"] == 1536
    for added in prefill:
        old = next(
            row
            for row in original
            if row["set_id"] == added["set_id"]
            and row["seed"] == added["seed"]
            and row["condition_id"] == added["condition_id"]
        )
        before = build_workload(old["workload_id"], old["seed"])
        after = build_workload(added["workload_id"], added["seed"])
        assert len(after["requests"]) == len(before["requests"]) == 16
        for previous, new in zip(before["requests"], after["requests"], strict=True):
            assert new["input_token_ids"] == previous["input_token_ids"]
            assert new["prime_input_token_ids"] == previous["prime_input_token_ids"]
            assert new["prefix_token_count"] == previous["prefix_token_count"]
            assert new["requested_output_tokens"] == 1
        assert after["summary"]["requested_output_token_total"] == 16


def test_long_context_schedule_stays_within_configured_sequence_cap():
    context_cap = 32768
    plan = build_long_context_plan(context_cap=context_cap)
    from taei.nano_dataset.orchestrator import build_plan

    aggregate = build_plan(
        workload_protocol="qwen3_8b_long_context_32k_v1", context_cap=context_cap
    )

    assert plan["run_count"] == 96
    assert plan["request_count"] == 1536
    assert len(plan["matrix"]) == 96
    assert plan["schedule"]["prompt_lengths"] == [
        5121,
        8705,
        12289,
        15873,
        19457,
        23041,
        26625,
        30209,
    ]
    assert plan["schedule"]["output_budgets"] == [
        512,
        512,
        1024,
        1024,
        1024,
        2048,
        2048,
        2048,
    ]
    assert plan["schedule"]["prefix_blocks_per_hit"] == [1, 6, 15, 27, 43, 62, 85, 112]
    assert plan["schedule"]["max_prompt_plus_output_tokens"] <= context_cap
    assert plan["schedule"]["max_prompt_plus_output_tokens"] == 32257
    assert aggregate["logical_prompt_tokens"] == 27_133_440
    assert aggregate["requested_output_tokens"] == 983_808
    assert plan["schedule"]["max_num_batched_tokens"] == 16384
    assert plan["schedule"]["max_num_seqs"] == 8
    assert (
        "prompt_length + requested_output_tokens"
        in plan["schedule"]["sequence_budget_enforcement"]
    )
    assert (
        "running set may exceed" in plan["schedule"]["dynamic_max_num_seqs_semantics"]
    )
    assert (
        plan["schedule"]["fixed_batch_policy"]
        == "admit waves no larger than max_num_seqs"
    )
    for run in plan["matrix"]:
        workload = build_workload(run["workload_id"], run["seed"])
        for request in workload["requests"]:
            assert (
                request["prompt_length"] + request["requested_output_tokens"]
                <= context_cap
            )
            assert request["requested_output_tokens"] in (1, 512, 1024, 2048)
            assert request["prompt_length"] >= 5121


def test_long_context_workload_is_paired_and_records_exact_schedule():
    workload = build_long_context_workload("qwen3_8b_long_context_set_01", 42)
    requests = workload["requests"]
    assert len(requests) == 16
    hits, misses = requests[::2], requests[1::2]
    assert [row["prefix_token_count"] // 256 for row in hits] == [
        1,
        6,
        15,
        27,
        43,
        62,
        85,
        112,
    ]
    assert len({row["prefix_token_count"] for row in hits}) == 8
    for hit, miss in zip(hits, misses, strict=True):
        assert hit["prompt_length"] == miss["prompt_length"]
        assert hit["requested_output_tokens"] == miss["requested_output_tokens"]
        assert hit["input_token_ids"] != miss["input_token_ids"]
        prefix_tokens = hit["prefix_token_count"]
        assert prefix_tokens % 256 == 0
        assert (
            hit["input_token_ids"][:prefix_tokens]
            == hit["prime_input_token_ids"][:prefix_tokens]
        )
        assert len(hit["prime_input_token_ids"]) == prefix_tokens + 1
        assert (
            hit["prime_input_token_ids"][-1]
            not in hit["input_token_ids"][prefix_tokens:]
        )
        assert miss["prefix_token_count"] == miss["expected_cache_hit_tokens_on"] == 0
        assert miss["prime_input_token_ids"] == []
        assert miss["target_cache_hit_ratio_on"] == 0
        assert hit["target_cache_hit_ratio_on"] == prefix_tokens / hit["prompt_length"]
        assert 0.045 <= hit["target_cache_hit_ratio_on"] <= 0.955
        # The miss has no matching full block with this pair's prime donor.
        for end in range(256, len(hit["prime_input_token_ids"]) - 1, 256):
            assert miss["input_token_ids"][:end] != hit["prime_input_token_ids"][:end]
    assert workload["summary"]["prompt_lengths"] == [
        5121,
        8705,
        12289,
        15873,
        19457,
        23041,
        26625,
        30209,
    ]
    assert workload["summary"]["requested_output_token_counts"] == [
        512,
        512,
        1024,
        1024,
        1024,
        2048,
        2048,
        2048,
    ]
    from jsonschema import Draft202012Validator

    from taei.nano_dataset.schema import DATASET_SCHEMA

    Draft202012Validator(DATASET_SCHEMA["$defs"]["workload"]).validate(workload)


def test_existing_workload_plan_remains_unchanged():
    from taei.nano_dataset.orchestrator import build_plan

    plan = build_plan()
    assert plan["run_count"] == 48
    assert plan["request_count"] == 768
    assert plan["workload_protocol"] == "six_set_paired_prefix_sweep_v1"


def test_long_context_hydra_plan_is_exact_and_cpu_only(tmp_path):
    import json
    import subprocess
    import sys

    from omegaconf import OmegaConf

    output_dir = tmp_path / "long-plan"
    result = subprocess.run(
        [
            sys.executable,
            "experiments/nano_vllm_qwen3_8b_long_context/run.py",
            "mode=plan",
            f"hydra.run.dir={output_dir}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    config = OmegaConf.load(output_dir / "config.yaml")
    plan = json.loads((output_dir / "plan.json").read_text())
    assert config.engine.max_model_len == 32768
    assert config.model.revision == "b968826d9c46dd6066d109eabc6255188de91218"
    assert config.model.upstream_config.max_position_embeddings == 40960
    assert plan["run_count"] == 96
    assert plan["request_count"] == 1536
    assert plan["schedule"]["max_num_seqs"] == config.engine.max_num_seqs == 8
    assert (
        plan["schedule"]["max_num_batched_tokens"]
        == config.engine.max_num_batched_tokens
        == 16384
    )
    assert plan["schedule"]["max_prompt_plus_output_tokens"] == 32257
    assert not (output_dir / "model-download.json").exists()


def test_long_context_plan_rejects_too_small_sequence_capacity():
    import pytest

    with pytest.raises(ValueError, match="below workload requirement 32257"):
        build_long_context_plan(context_cap=32256)


def test_cache_on_diagnostic_has_one_13k_input_1k_output_workload():
    from taei.nano_dataset.orchestrator import build_plan

    protocol = "qwen3_8b_cache_on_diagnostic_v1"
    plan = build_plan(workload_protocol=protocol, context_cap=32768)
    assert plan["run_count"] == 1
    assert plan["request_count"] == 8
    run = plan["matrix"][0]
    assert run["prefix_cache_mode"] == "on"
    assert run["scheduler_mode"] == "upstream_dynamic"
    from jsonschema import Draft202012Validator

    from taei.nano_dataset.schema import DATASET_SCHEMA

    workload = build_workload(run["workload_id"], run["seed"])
    Draft202012Validator(DATASET_SCHEMA["$defs"]["workload"]).validate(workload)
    assert {len(row["input_token_ids"]) for row in workload["requests"]} == {13000}
    assert {row["requested_output_tokens"] for row in workload["requests"]} == {1000}
    hits = [row for row in workload["requests"] if row["cache_intent"] == "hit"]
    misses = [row for row in workload["requests"] if row["cache_intent"] == "miss"]
    assert len(hits) == len(misses) == 4
    assert [row["prefix_token_count"] for row in hits] == [2048, 4096, 6144, 8192]
    assert all(row["prefix_token_count"] == 0 for row in misses)
    for hit, miss in zip(hits, misses, strict=True):
        assert (
            hit["input_token_ids"][: hit["prefix_token_count"]]
            == hit["prime_input_token_ids"][:-1]
        )
        assert miss["input_token_ids"][:256] != hit["prime_input_token_ids"][:256]


def test_cache_on_diagnostic_hydra_composition_and_plan_are_exact(tmp_path):
    import json
    import subprocess
    import sys

    from omegaconf import OmegaConf

    output_dir = tmp_path / "diagnostic-plan"
    result = subprocess.run(
        [
            sys.executable,
            "experiments/nano_vllm_qwen3_8b_long_context/run.py",
            "--config-name",
            "diagnostic",
            "mode=plan",
            f"hydra.run.dir={output_dir}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    config = OmegaConf.load(output_dir / "config.yaml")
    plan = json.loads((output_dir / "plan.json").read_text())
    assert config.dataset.workload_protocol == "qwen3_8b_cache_on_diagnostic_v1"
    assert config.dataset.run_limit == 1
    assert config.engine.max_model_len == 32768
    assert plan["run_count"] == 1
    assert plan["request_count"] == 8
    assert plan["matrix"][0]["prefix_cache_mode"] == "on"
    assert plan["logical_prompt_tokens"] == 104000
    assert plan["requested_output_tokens"] == 8000
    assert not (output_dir / "model-download.json").exists()
