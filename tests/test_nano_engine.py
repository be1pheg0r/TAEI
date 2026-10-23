"""Проверяет адаптер nano-vLLM на закрепленном upstream-коде и CPU-трассах.

Тесты контролируют ленивый импорт GPU-зависимостей и проверку закрепленной
версии исходников, интерфейс адаптера и работу реальных upstream-примитивов
планировщика/кэша на CPU. Проверяются повторное использование общего префикса,
границы блоков, отключенный кэш, фиксированные волны и динамическое планирование,
chunked prefill, ограничения профилей и размер снимка аллокатора. CPU-проверки
не трактуются как доказательство GPU-инференса.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PIN = "bb823b3e06983d71485a8e1f23715ebd87d98ef8"
DEFAULT_UPSTREAM = Path("/root/.hermes/cache/scratch/nano-vllm-pin")
UPSTREAM = Path(os.environ.get("TAEI_NANO_VLLM_SOURCE", DEFAULT_UPSTREAM))


def require_upstream() -> Path:
    if not UPSTREAM.is_dir():
        pytest.skip(
            "pinned nano-vLLM source checkout unavailable; set TAEI_NANO_VLLM_SOURCE"
        )
    return UPSTREAM


def test_engine_module_import_does_not_import_gpu_stack():
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "from taei.nano_dataset.engine import NanoVLLMBackend; "
        "assert 'torch' not in sys.modules; assert 'nanovllm' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_pin_source_is_checked_before_adapter_patching():
    from taei.nano_dataset.engine import pinned_source_manifest, verify_pinned_source

    root = require_upstream()
    manifest = pinned_source_manifest()
    assert (
        manifest["nanovllm/engine/scheduler.py"]
        == "e7302942c967d50bb525384fa4a6e0de988d199490479d4f5965d4211bcdbd08"
    )
    assert verify_pinned_source(root, expected_revision=PIN) == manifest
    altered = dict(manifest)
    altered["nanovllm/engine/scheduler.py"] = "0" * 64
    with pytest.raises(RuntimeError, match="source hash"):
        verify_pinned_source(root, expected_revision=PIN, expected_hashes=altered)


def test_backend_configuration_and_generation_interface_is_stable():
    from taei.nano_dataset.engine import NanoVLLMBackend

    backend = NanoVLLMBackend(
        model_path="/model",
        engine_config={
            "max_num_seqs": 8,
            "max_num_batched_tokens": 16384,
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
            "enforce_eager": False,
            "gpu_memory_utilization": 0.9,
            "kvcache_block_size": 256,
        },
        prefix_cache_mode="off",
        scheduler_mode="fixed_batch",
        seed=42,
        emit=lambda table, row: None,
    )
    assert backend.prefix_cache_mode == "off"
    assert backend.scheduler_mode == "fixed_batch"
    assert callable(backend.prepare)
    assert callable(backend.synchronize)
    assert callable(backend.generate)
    assert callable(backend.close)


def test_pinned_upstream_scheduler_and_block_manager_run_on_cpu():
    """Exercise pinned real sequence/scheduler/cache code, not test doubles."""
    from taei.nano_dataset.engine import load_pinned_cpu_primitives

    primitives = load_pinned_cpu_primitives(require_upstream(), expected_revision=PIN)
    Config = primitives["Config"]
    Sequence = primitives["Sequence"]
    SamplingParams = primitives["SamplingParams"]
    Scheduler = primitives["Scheduler"]
    config = Config.__new__(Config)
    config.max_num_seqs = 2
    config.max_num_batched_tokens = 1024
    config.eos = -1
    config.kvcache_block_size = 256
    config.num_kvcache_blocks = 32
    Sequence.block_size = 256
    scheduler = Scheduler(config)
    first_params = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)
    first = Sequence(list(range(256)) + [900], first_params)
    scheduler.add(first)
    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill is True
    assert scheduled == [first]
    cached_block_id = first.block_table[0]
    scheduler.postprocess(scheduled, [1001], is_prefill)
    cached_block = scheduler.block_manager.blocks[cached_block_id]
    assert cached_block.ref_count == 0
    assert cached_block.hash != -1

    later_params = SamplingParams(temperature=0.6, max_tokens=3, ignore_eos=True)
    second = Sequence(list(range(256)) + [901], later_params)
    scheduler.add(second)
    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill is True
    assert scheduled == [second]
    assert second.num_cached_tokens == 256
    assert second.block_table[0] == cached_block_id
    scheduler.postprocess(scheduled, [1002], is_prefill)
    assert scheduler.block_manager.blocks[cached_block_id].ref_count == 1

    third = Sequence(list(range(256)) + [902], later_params)
    scheduler.add(third)
    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill is True
    assert scheduled == [third]
    assert third.block_table[0] == cached_block_id
    assert scheduler.block_manager.blocks[cached_block_id].ref_count == 2
    assert (
        scheduler.block_manager.hash_to_block_id[cached_block.hash] == cached_block_id
    )


def test_pinned_scheduler_does_not_reuse_the_final_exact_boundary_block():
    from taei.nano_dataset.engine import load_pinned_cpu_primitives

    primitives = load_pinned_cpu_primitives(require_upstream(), expected_revision=PIN)
    Config, Sequence = primitives["Config"], primitives["Sequence"]
    SamplingParams, Scheduler = primitives["SamplingParams"], primitives["Scheduler"]
    config = Config.__new__(Config)
    config.max_num_seqs, config.max_num_batched_tokens = 1, 1024
    config.eos, config.kvcache_block_size, config.num_kvcache_blocks = -1, 256, 16
    Sequence.block_size = 256
    scheduler = Scheduler(config)
    params = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)
    first = Sequence(list(range(256)), params)
    scheduler.add(first)
    seqs, prefill = scheduler.schedule()
    saved_block = first.block_table[0]
    scheduler.postprocess(seqs, [3000], prefill)
    assert scheduler.block_manager.blocks[saved_block].hash != -1
    exact_boundary = Sequence(list(range(256)), params)
    scheduler.add(exact_boundary)
    seqs, prefill = scheduler.schedule()
    assert prefill is True
    assert exact_boundary.num_cached_tokens == 0


def test_cpu_primitive_loader_restores_import_namespace():
    from taei.nano_dataset.engine import load_pinned_cpu_primitives

    before = {
        key: value
        for key, value in sys.modules.items()
        if key == "nanovllm" or key.startswith("nanovllm.")
    }
    transformer_before = sys.modules.get("transformers")
    load_pinned_cpu_primitives(require_upstream(), expected_revision=PIN)
    after = {
        key: value
        for key, value in sys.modules.items()
        if key == "nanovllm" or key.startswith("nanovllm.")
    }
    assert after == before
    assert sys.modules.get("transformers") is transformer_before


def _traced_cpu_engine(primitives, mode, events, prefix_mode="on"):
    from types import SimpleNamespace

    from taei.nano_dataset.engine import NanoVLLMBackend

    class Runner:
        enforce_eager = True

        @staticmethod
        def call(method, seqs, is_prefill):
            return [500 + seq.seq_id for seq in seqs]

    class Engine:
        def __init__(self, scheduler):
            self.scheduler = scheduler
            self.model_runner = Runner()

        def add_request(self, prompt, sampling_params):
            self.scheduler.add(primitives["Sequence"](prompt, sampling_params))

        def step(self):
            seqs, is_prefill = self.scheduler.schedule()
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
            completed = [
                (seq.seq_id, seq.completion_token_ids)
                for seq in seqs
                if seq.is_finished
            ]
            return completed, 0

    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=1024,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=32,
    )
    primitives["Sequence"].block_size = 256
    scheduler = primitives["Scheduler"](config)
    backend = NanoVLLMBackend(
        "/model",
        {"tensor_parallel_size": 1},
        prefix_mode,
        mode,
        42,
        lambda table, row: events.append((table, row)),
    )
    backend.engine = Engine(scheduler)
    backend._patch_live_engine()
    return backend


def test_prefix_on_trace_counts_cross_request_reuse_not_private_decode_context():
    from taei.nano_dataset.engine import load_pinned_cpu_primitives

    primitives = load_pinned_cpu_primitives(require_upstream(), expected_revision=PIN)
    events = []
    backend = _traced_cpu_engine(primitives, "upstream_dynamic", events)
    params = primitives["SamplingParams"](
        temperature=0.6, max_tokens=2, ignore_eos=True
    )
    backend._pending_request_ids = iter(("prefix-a", "prefix-b"))
    prompt_a = list(range(256)) + [901]
    prompt_b = list(range(256)) + [902]
    backend.engine.add_request(prompt_a, params)
    backend.engine.step()
    backend.engine.add_request(prompt_b, params)
    backend.engine.step()
    request_rows = [row for table, row in events if table == "step_requests"]
    second = next(row for row in request_rows if row["request_id"] == "prefix-b")
    assert second["prefix_reused_tokens"] == 256
    backend.engine.step()
    decode_row = [row for table, row in events if table == "step_requests"][-2:]
    assert all(row["prefix_reused_tokens"] == 0 for row in decode_row)
    backend._restore()


def test_fixed_batch_keeps_all_requests_and_advances_only_after_wave_finishes():
    from taei.nano_dataset.engine import load_pinned_cpu_primitives

    primitives = load_pinned_cpu_primitives(require_upstream(), expected_revision=PIN)
    events = []
    backend = _traced_cpu_engine(primitives, "fixed_batch", events)
    SamplingParams = primitives["SamplingParams"]
    backend._pending_request_ids = iter(f"r{i}" for i in range(5))
    for _ in range(5):
        backend.engine.add_request(
            [10, 11], SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)
        )
    assert len(backend.engine.scheduler.waiting) == 5
    backend.engine.step()
    assert len(backend.engine.scheduler.waiting) == 3
    first_wave = [
        row
        for table, row in events
        if table == "scheduler_events" and row.get("event_type") == "schedule"
    ][-1]
    assert first_wave["admitted_request_ids"] == ["r0", "r1"]
    backend.engine.step()
    second_wave = [
        row
        for table, row in events
        if table == "scheduler_events" and row.get("event_type") == "schedule"
    ][-1]
    assert second_wave["admitted_request_ids"] == ["r2", "r3"]
    backend.engine.step()
    third_wave = [
        row
        for table, row in events
        if table == "scheduler_events" and row.get("event_type") == "schedule"
    ][-1]
    assert third_wave["admitted_request_ids"] == ["r4"]
    assert backend.engine.scheduler.is_finished()
    step_rows = [row for table, row in events if table == "steps"]
    token_rows = [row for table, row in events if table == "token_events"]
    joins = [row for table, row in events if table == "step_requests"]
    assert len(step_rows) == 3
    assert len(joins) == 5
    assert len(token_rows) == 5
    assert all(row["produced_tokens"] == row["actual_batch_size"] for row in step_rows)
    assert all(row["host_observed_timestamp_ns"] is not None for row in token_rows)
    assert {row["request_id"] for row in joins} == {f"r{i}" for i in range(5)}
    backend._restore()


def test_prefix_cache_off_disables_hashing_and_lookup_on_real_upstream_scheduler():
    from taei.nano_dataset.engine import load_pinned_cpu_primitives

    primitives = load_pinned_cpu_primitives(require_upstream(), expected_revision=PIN)
    events = []
    backend = _traced_cpu_engine(
        primitives, "upstream_dynamic", events, prefix_mode="off"
    )
    params = primitives["SamplingParams"](
        temperature=0.6, max_tokens=1, ignore_eos=True
    )
    prompt = list(range(256)) + [900]
    backend._pending_request_ids = iter(("off-a", "off-b"))
    backend.engine.add_request(prompt, params)
    backend.engine.scheduler.waiting[-1]
    backend.engine.step()
    assert not backend.engine.scheduler.block_manager.hash_to_block_id
    backend.engine.add_request(prompt[:-1] + [901], params)
    second = backend.engine.scheduler.waiting[-1]
    backend.engine.step()
    assert second.num_cached_tokens == 0
    assert not backend.engine.scheduler.block_manager.hash_to_block_id
    lookups = [
        row
        for table, row in events
        if table == "cache_events" and row.get("operation") == "can_allocate"
    ]
    assert len(lookups) == 2
    assert all(row["prefix_hit_blocks"] == 0 for row in lookups)
    assert all(row["prefix_cache_mode"] == "off" for row in lookups)
    backend._restore()


def test_minimal_profile_is_explicit_and_restricted_to_matched_baseline_modes():
    from taei.nano_dataset.engine import NanoVLLMBackend

    backend = NanoVLLMBackend(
        "/model",
        {"tensor_parallel_size": 1},
        "on",
        "upstream_dynamic",
        42,
        lambda table, row: None,
        measurement_profile="minimal",
    )
    assert backend.measurement_profile == "minimal"
    with pytest.raises(ValueError, match="minimal profile"):
        NanoVLLMBackend(
            "/model",
            {"tensor_parallel_size": 1},
            "off",
            "upstream_dynamic",
            42,
            lambda table, row: None,
            measurement_profile="minimal",
        )


def test_allocator_trace_snapshot_size_is_independent_of_kv_capacity():
    import json
    from types import SimpleNamespace

    from taei.nano_dataset.engine import NanoVLLMBackend

    backend = NanoVLLMBackend(
        "/model",
        {"tensor_parallel_size": 1},
        "on",
        "upstream_dynamic",
        42,
        lambda table, row: None,
    )
    small = SimpleNamespace(
        blocks=[None] * 8,
        free_block_ids=range(8),
        used_block_ids=set(),
        hash_to_block_id={},
    )
    large = SimpleNamespace(
        blocks=[None] * 100_000,
        free_block_ids=range(100_000),
        used_block_ids=set(),
        hash_to_block_id={},
    )
    small_row = backend._block_snapshot(small)
    large_row = backend._block_snapshot(large)
    assert set(small_row) == {
        "capacity_blocks",
        "free_block_count",
        "used_block_count",
        "hashed_block_count",
        "free_queue_head",
    }
    assert len(json.dumps(large_row)) - len(json.dumps(small_row)) < 20


def test_chunked_prefill_only_counts_token_accepted_by_upstream_sequence():
    from taei.nano_dataset.engine import load_pinned_cpu_primitives

    primitives = load_pinned_cpu_primitives(require_upstream(), expected_revision=PIN)
    events = []
    backend = _traced_cpu_engine(primitives, "upstream_dynamic", events)
    backend.engine.scheduler.max_num_batched_tokens = 256
    backend._pending_request_ids = iter(("chunked",))
    backend._requests["chunked"] = {"requested_output_tokens": 1, "ignore_eos": True}
    params = primitives["SamplingParams"](
        temperature=0.6, max_tokens=1, ignore_eos=True
    )
    backend.engine.add_request(list(range(600)), params)
    sequence = backend.engine.scheduler.waiting[0]
    backend.engine.step()
    assert sequence.num_completion_tokens == 0
    backend.engine.step()
    assert sequence.num_completion_tokens == 0
    backend.engine.step()
    assert sequence.num_completion_tokens == 1
    steps = [row for table, row in events if table == "steps"]
    token_events = [row for table, row in events if table == "token_events"]
    assert [row["produced_tokens"] for row in steps] == [0, 0, 1]
    assert len(token_events) == 1
    assert token_events[0]["request_id"] == "chunked"
    assert token_events[0]["token_index"] == 0
    backend._restore()


def test_upstream_dynamic_retains_actual_running_growth_behavior():
    from taei.nano_dataset.engine import load_pinned_cpu_primitives

    primitives = load_pinned_cpu_primitives(require_upstream(), expected_revision=PIN)
    events = []
    backend = _traced_cpu_engine(primitives, "upstream_dynamic", events)
    SamplingParams = primitives["SamplingParams"]
    backend._pending_request_ids = iter(f"d{i}" for i in range(4))
    for _ in range(4):
        backend.engine.add_request(
            [10, 11], SamplingParams(temperature=0.6, max_tokens=3, ignore_eos=True)
        )
    backend.engine.step()
    backend.engine.step()
    schedules = [
        row
        for table, row in events
        if table == "scheduler_events" and row.get("event_type") == "schedule"
    ]
    assert schedules[0]["active_count_after"] == 2
    assert schedules[1]["active_count_after"] == 4
    assert schedules[1]["active_count_may_exceed_max_num_seqs"] is True
    backend._restore()
