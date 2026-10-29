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
import types
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
            "tensor_parallel_size": 4,
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


def test_missing_model_fails_before_loading_inference_stack(tmp_path, monkeypatch):
    from taei.nano_dataset import engine

    def unexpected_upstream_load(*_):
        raise AssertionError("inference stack loaded for missing model")

    monkeypatch.setattr(engine, "_upstream_root", lambda: tmp_path)
    monkeypatch.setattr(engine, "verify_pinned_source", lambda _: {})
    monkeypatch.setattr(engine.importlib, "import_module", unexpected_upstream_load)
    backend = engine.NanoVLLMBackend(
        model_path=str(tmp_path / "missing"),
        engine_config={},
        prefix_cache_mode="on",
        scheduler_mode="upstream_dynamic",
        seed=42,
        emit=lambda *_: None,
    )
    with pytest.raises(FileNotFoundError, match="model directory"):
        backend.prepare()


def _install_fake_model_runner(monkeypatch, runner_class):
    from types import SimpleNamespace

    calls = []

    class FakeCuda:
        @staticmethod
        def set_device(device):
            calls.append(("set_device", device))

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def device(kind, index):
            return (kind, index)

    def original_init(*args, **kwargs):
        calls.append(("dist_init", args, kwargs))

    FakeTorch.distributed = SimpleNamespace(init_process_group=original_init)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    package = types.ModuleType("nanovllm")
    package.__path__ = []
    engine_package = types.ModuleType("nanovllm.engine")
    engine_package.__path__ = []
    model_runner = types.ModuleType("nanovllm.engine.model_runner")
    model_runner.__dict__["ModelRunner"] = runner_class
    monkeypatch.setitem(sys.modules, "nanovllm", package)
    monkeypatch.setitem(sys.modules, "nanovllm.engine", engine_package)
    monkeypatch.setitem(sys.modules, "nanovllm.engine.model_runner", model_runner)
    return FakeTorch, original_init, calls


def test_spawned_model_runner_passes_rank_device_to_distributed_init(monkeypatch):
    from taei.nano_dataset.engine import _create_device_bound_model_runner

    class FakeRunner:
        def __init__(self, config, rank, event):
            import torch

            torch.distributed.init_process_group(
                "nccl", "tcp://localhost:2333", world_size=4, rank=rank
            )

    fake_torch, original_init, calls = _install_fake_model_runner(
        monkeypatch, FakeRunner
    )
    result = _create_device_bound_model_runner(object(), 3, object())

    assert isinstance(result, FakeRunner)
    assert calls == [
        ("set_device", 3),
        (
            "dist_init",
            ("nccl", "tcp://localhost:2333"),
            {"world_size": 4, "rank": 3, "device_id": ("cuda", 3)},
        ),
    ]
    assert fake_torch.distributed.init_process_group is original_init


def test_spawned_model_runner_restores_distributed_init_after_constructor_error(
    monkeypatch,
):
    from taei.nano_dataset.engine import _create_device_bound_model_runner

    class BrokenRunner:
        def __init__(self, config, rank, event):
            import torch

            torch.distributed.init_process_group("nccl", rank=rank)
            raise RuntimeError("runner failed")

    fake_torch, original_init, calls = _install_fake_model_runner(
        monkeypatch, BrokenRunner
    )
    with pytest.raises(RuntimeError, match="runner failed"):
        _create_device_bound_model_runner(object(), 2, object())

    assert calls[-1] == (
        "dist_init",
        ("nccl",),
        {"rank": 2, "device_id": ("cuda", 2)},
    )
    assert fake_torch.distributed.init_process_group is original_init


def test_spawned_model_runner_preserves_explicit_distributed_device_id(monkeypatch):
    from taei.nano_dataset.engine import _create_device_bound_model_runner

    explicit_device = ("cuda", 7)

    class ExplicitRunner:
        def __init__(self, config, rank, event):
            import torch

            torch.distributed.init_process_group(
                "nccl", rank=rank, device_id=explicit_device
            )

    fake_torch, original_init, calls = _install_fake_model_runner(
        monkeypatch, ExplicitRunner
    )
    _create_device_bound_model_runner(object(), 3, object())

    assert calls[-1] == (
        "dist_init",
        ("nccl",),
        {"rank": 3, "device_id": explicit_device},
    )
    assert fake_torch.distributed.init_process_group is original_init


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


def _traced_cpu_engine(
    primitives, mode, events, prefix_mode="on", blocks=32, fixed_batch_size=None
):
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

        def is_finished(self):
            return self.scheduler.is_finished()

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
        num_kvcache_blocks=blocks,
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
        fixed_batch_size=fixed_batch_size,
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
    assert all(row["token_type"] == "sampled_output" for row in token_rows)
    assert all(row["produced_tokens"] == row["actual_batch_size"] for row in step_rows)
    assert all(row["host_observed_timestamp_ns"] is not None for row in token_rows)
    assert {row["request_id"] for row in joins} == {f"r{i}" for i in range(5)}
    backend._restore()


def test_fixed_batch_keeps_a_preempted_request_until_it_finishes():
    from taei.nano_dataset.engine import load_pinned_cpu_primitives

    primitives = load_pinned_cpu_primitives(require_upstream(), expected_revision=PIN)
    events = []
    # Two requests of one block each grow into a second block; with three
    # blocks the second must be preempted, recomputed and still complete.
    backend = _traced_cpu_engine(primitives, "fixed_batch", events, blocks=3)
    params = primitives["SamplingParams"](
        temperature=0.6, max_tokens=40, ignore_eos=True
    )
    backend._pending_request_ids = iter(("a", "b"))
    for _ in range(2):
        backend.engine.add_request(list(range(230)), params)
    for _ in range(500):
        if backend.engine.is_finished():
            break
        backend.engine.step()
    assert backend.engine.is_finished()
    done = [
        row["request_id"]
        for table, row in events
        if table == "scheduler_events" and row.get("event_type") == "complete"
    ]
    assert sorted(done) == ["a", "b"]
    preempted = [
        rid
        for table, row in events
        if table == "scheduler_events" and row.get("event_type") == "schedule"
        for rid in row.get("preempted_request_ids", [])
    ]
    assert preempted
    recompute = [
        row for table, row in events if table == "steps" and row["recompute_tokens"]
    ]
    assert recompute
    backend._restore()


def test_fixed_batch_wave_size_is_separate_from_max_num_seqs():
    from taei.nano_dataset.engine import load_pinned_cpu_primitives

    primitives = load_pinned_cpu_primitives(require_upstream(), expected_revision=PIN)
    events = []
    backend = _traced_cpu_engine(primitives, "fixed_batch", events, fixed_batch_size=1)
    params = primitives["SamplingParams"](
        temperature=0.6, max_tokens=1, ignore_eos=True
    )
    backend._pending_request_ids = iter(("a", "b"))
    for _ in range(2):
        backend.engine.add_request([10, 11], params)
    backend.engine.step()
    first = [
        row
        for table, row in events
        if table == "scheduler_events" and row.get("event_type") == "schedule"
    ][-1]
    assert first["admitted_request_ids"] == ["a"]
    backend._restore()


def test_timed_arrivals_add_requests_in_offset_order_and_wait_when_idle():
    from taei.nano_dataset.engine import load_pinned_cpu_primitives

    primitives = load_pinned_cpu_primitives(require_upstream(), expected_revision=PIN)
    events = []
    backend = _traced_cpu_engine(primitives, "upstream_dynamic", events)
    sampling = primitives["SamplingParams"]
    sys.modules.setdefault(
        "nanovllm.sampling_params", types.SimpleNamespace(SamplingParams=sampling)
    )
    requests = [
        {
            "request_id": rid,
            "input_token_ids": [10 + i, 11],
            "requested_output_tokens": 2,
            "temperature": 0.6,
            "ignore_eos": True,
            "planned_arrival_offset_ns": offset,
        }
        for i, (rid, offset) in enumerate((("late", 30_000_000), ("early", 1)))
    ]
    records = {r["request_id"]: r for r in backend.generate(requests)}
    assert all(r["output_token_count"] == 2 for r in records.values())
    early, late = records["early"], records["late"]
    assert early["enqueue_monotonic_ns"] < late["enqueue_monotonic_ns"]
    # The late request is enqueued only after its offset, so after early ends.
    assert late["enqueue_monotonic_ns"] >= early["finish_monotonic_ns"]
    assert late["enqueue_monotonic_ns"] - early["enqueue_monotonic_ns"] >= 25_000_000
    backend._restore()


def test_closed_loop_keeps_n_in_system_and_marks_roles():
    from taei.nano_dataset.engine import load_pinned_cpu_primitives

    primitives = load_pinned_cpu_primitives(require_upstream(), expected_revision=PIN)
    events = []
    backend = _traced_cpu_engine(primitives, "upstream_dynamic", events)
    sampling = primitives["SamplingParams"]
    sys.modules.setdefault(
        "nanovllm.sampling_params", types.SimpleNamespace(SamplingParams=sampling)
    )
    n, warmup, measure = 2, 2, 3
    loop = {"N": n, "warmup": warmup, "measure": measure}
    requests = [
        {
            "request_id": f"q{i}",
            "arrival_index": i,
            "input_token_ids": [10 + i, 11],
            "requested_output_tokens": 2 + i % 2,
            "temperature": 0.6,
            "ignore_eos": True,
            "planned_arrival_offset_ns": 0,
            "closed_loop": loop,
        }
        for i in range(n + warmup + measure - 1)
    ]
    records = backend.generate(requests)
    assert all(
        r["output_token_count"] == r["requested_output_tokens"]
        for r in [{**q, **rec} for q, rec in zip(requests, records, strict=True)]
    )
    enq = sorted(r["enqueue_monotonic_ns"] for r in records)
    fin = sorted(r["finish_monotonic_ns"] for r in records)
    # Before the stop_after-th completion the system holds at most N, and every
    # completion before it submits the next request.
    for t in enq:
        assert sum(e <= t for e in enq) - sum(f < t for f in fin) <= n
    assert enq[n] >= fin[0]
    roles = [
        r["closed_loop_role"]
        for r in sorted(records, key=lambda r: r["finish_monotonic_ns"])
    ]
    assert roles == ["warmup"] * warmup + ["measure"] * measure + ["tail"] * (n - 1)
    summary = backend.closed_loop_summary
    assert summary["window_start_ns"] == fin[warmup - 1]
    assert summary["window_end_ns"] == fin[warmup + measure - 1]
    backend._restore()


def test_closed_loop_time_limit_stops_submitting_after_min_measure():
    from taei.nano_dataset.engine import load_pinned_cpu_primitives

    primitives = load_pinned_cpu_primitives(require_upstream(), expected_revision=PIN)
    backend = _traced_cpu_engine(primitives, "upstream_dynamic", [])
    sampling = primitives["SamplingParams"]
    sys.modules.setdefault(
        "nanovllm.sampling_params", types.SimpleNamespace(SamplingParams=sampling)
    )
    # Any window is over at once: the loop stops at the first completion
    # past warmup + min_measure; the rest of the pool is never submitted.
    n, warmup, measure, min_measure = 2, 2, 5, 1
    loop = {
        "N": n,
        "warmup": warmup,
        "measure": measure,
        "min_measure": min_measure,
        "max_window_s": 1e-9,
    }
    requests = [
        {
            "request_id": f"q{i}",
            "arrival_index": i,
            "input_token_ids": [10 + i, 11],
            "requested_output_tokens": 2,
            "temperature": 0.6,
            "ignore_eos": True,
            "planned_arrival_offset_ns": 0,
            "closed_loop": loop,
        }
        for i in range(n + warmup + measure - 1)
    ]
    records = backend.generate(requests)
    summary = backend.closed_loop_summary
    assert summary["measure"] == min_measure and summary["planned_measure"] == measure
    assert summary["submitted"] == len(records) == n + warmup + min_measure - 1
    assert all(r["output_token_count"] == 2 for r in records)
    roles = sorted(r["closed_loop_role"] for r in records)
    assert roles.count("warmup") == warmup and roles.count("measure") == min_measure
    backend._restore()


def test_steps_record_sum_of_contexts_unique_kv_blocks_and_cache_fill():
    from taei.nano_dataset.engine import load_pinned_cpu_primitives

    primitives = load_pinned_cpu_primitives(require_upstream(), expected_revision=PIN)
    events = []
    backend = _traced_cpu_engine(primitives, "upstream_dynamic", events)
    params = primitives["SamplingParams"](
        temperature=0.6, max_tokens=3, ignore_eos=True
    )
    shared = list(range(256))  # one full block both requests share
    backend._pending_request_ids = iter(("a", "b"))
    # Blocks are hashed after their prefill, so b comes once a is computed.
    backend.engine.add_request(shared + [900, 901], params)
    backend.engine.step()
    backend.engine.add_request(shared + [902, 903], params)
    while not backend.engine.is_finished():
        backend.engine.step()
    steps = [row for table, row in events if table == "steps"]
    for row in steps:
        assert row["context_tokens_sum"] == sum(row["contexts_before"].values())
        assert row["kv_capacity_blocks"] == 32
        assert row["kv_fill"] == row["kv_used_blocks"] / 32
        assert row["preempted_count"] == 0
    both = [r for r in steps if r["phase"] == "decode" and r["batch_size"] == 2]
    # Two own tail blocks and the shared prefix block counted once.
    assert both and all(row["kv_unique_blocks"] == 3 for row in both)
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
    assert all(row["event_type"] == "capacity_check" for row in lookups)
    assert all("prefix_hit_blocks" not in row for row in lookups)
    assert all(row["prefix_cache_mode"] == "off" for row in lookups)
    schedules = [
        row
        for table, row in events
        if table == "scheduler_events" and row.get("event_type") == "schedule"
    ]
    steps = [row for table, row in events if table == "steps"]
    assert [event["step_id"] for event in schedules] == [
        step["step_id"] for step in steps
    ]
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
