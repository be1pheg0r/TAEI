"""Адаптер закрепленной версии nano-vLLM с проверяемыми трассами хоста.

Модуль подтверждает ревизию и контрольные суммы используемых исходников,
загружает необходимые upstream-примитивы для диагностики и предоставляет
``NanoVLLMBackend`` для настройки и выполнения настоящего ``generate`` с
наблюдаемыми событиями планировщика, кэша, шагов и токенов. GPU-зависимости
подключаются отложенно; CPU-примитивы используются только для тестов и
диагностики, а не как запасной путь инференса.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import importlib
import importlib.util
import sys
import time
import types
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PINNED_REVISION = "bb823b3e06983d71485a8e1f23715ebd87d98ef8"
_PINNED_SOURCE_HASHES = {
    "nanovllm/llm.py": (
        "ffe74fe7f64ba5169471a17596757b8d8f309e75f4db7040ac41b84db2e38179"
    ),
    "nanovllm/config.py": (
        "bca5fddf17092e26d14075348965e46a3b707f5ad780b04b31f48e91bd38e98c"
    ),
    "nanovllm/sampling_params.py": (
        "1e5146d1a06f5230ab049df1ea24814a1d73ba52c3ad22af04bd0f1c6293e0c0"
    ),
    "nanovllm/engine/llm_engine.py": (
        "c8132cd9f5e66faeea51d0ed6a64d7f4fd8ec5a5823dc2505c2aee51ba049aa1"
    ),
    "nanovllm/engine/scheduler.py": (
        "e7302942c967d50bb525384fa4a6e0de988d199490479d4f5965d4211bcdbd08"
    ),
    "nanovllm/engine/block_manager.py": (
        "d0240374f589eaa85c4c7923b8970f2635ab7bc73adc5044fa2d273955b54242"
    ),
    "nanovllm/engine/sequence.py": (
        "d5bb5c522d22554b5c36de745150ec8b601827ee0388058404dcceef1ae8290d"
    ),
    "nanovllm/engine/model_runner.py": (
        "2373c9daad7de3956905c9a0ec9d6b2e0a8d355a209162de8570507c5f419fd7"
    ),
    "nanovllm/layers/sampler.py": (
        "63a342a4ec02e846d5872d2906dc25ef5af1b7d66b85a63a6a88668f23543050"
    ),
}


def pinned_source_manifest() -> dict[str, str]:
    """Return the immutable allowlist for source files the adapter relies on."""
    return dict(_PINNED_SOURCE_HASHES)


def verify_pinned_source(
    source_root: str | Path,
    *,
    expected_revision: str = PINNED_REVISION,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, str]:
    """Fail closed unless exact pinned revision and relied-on source match."""
    root = Path(source_root).resolve()
    if expected_revision != PINNED_REVISION:
        raise RuntimeError(f"nano-vLLM revision mismatch: {expected_revision}")
    git_dir = root / ".git"
    if git_dir.exists():
        import subprocess

        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != expected_revision:
            raise RuntimeError(
                f"nano-vLLM checkout is {revision}, expected {expected_revision}"
            )
    expected = _PINNED_SOURCE_HASHES if expected_hashes is None else expected_hashes
    found = {}
    for relative, expected_hash in expected.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"pinned nano-vLLM source missing: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise RuntimeError(f"nano-vLLM source hash mismatch: {relative}")
        found[relative] = digest
    return found


def _upstream_root() -> Path:
    try:
        spec = importlib.util.find_spec("nanovllm")
    except (ImportError, ValueError):
        spec = None
    if spec and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).parent
    raise RuntimeError("nano-vLLM is not installed; install the pinned inference extra")


def load_pinned_cpu_primitives(
    source_root: str | Path, *, expected_revision: str = PINNED_REVISION
) -> dict[str, Any]:
    """Load the actual upstream CPU scheduler/cache modules without GPU imports.

    This helper is intended for tests and diagnostics, never as an inference fallback.
    """
    root = Path(source_root).resolve()
    verify_pinned_source(root, expected_revision=expected_revision)
    if (
        importlib.util.find_spec("numpy") is None
        or importlib.util.find_spec("xxhash") is None
    ):
        raise RuntimeError("CPU primitive tests require numpy and xxhash")
    import numpy  # noqa: F401

    saved_modules = {
        key: value
        for key, value in sys.modules.items()
        if key == "nanovllm" or key.startswith("nanovllm.")
    }
    saved_transformers = sys.modules.get("transformers")
    package = types.ModuleType("nanovllm")
    package.__path__ = [str(root / "nanovllm")]
    package.__package__ = "nanovllm"
    sys.modules["nanovllm"] = package
    for name in ("nanovllm.engine",):
        module = types.ModuleType(name)
        module.__path__ = [str(root / name.replace(".", "/"))]
        module.__package__ = name
        sys.modules[name] = module

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        transformers.AutoConfig = type("AutoConfig", (), {})
        sys.modules["transformers"] = transformers

    def load(name: str, relative: str) -> Any:
        spec = importlib.util.spec_from_file_location(name, root / relative)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load pinned module: {relative}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    config = load("nanovllm.config", "nanovllm/config.py")
    sampling = load("nanovllm.sampling_params", "nanovllm/sampling_params.py")
    sequence = load("nanovllm.engine.sequence", "nanovllm/engine/sequence.py")
    block = load("nanovllm.engine.block_manager", "nanovllm/engine/block_manager.py")
    scheduler = load("nanovllm.engine.scheduler", "nanovllm/engine/scheduler.py")
    result = {
        "Config": config.Config,
        "SamplingParams": sampling.SamplingParams,
        "Sequence": sequence.Sequence,
        "SequenceStatus": sequence.SequenceStatus,
        "BlockManager": block.BlockManager,
        "Scheduler": scheduler.Scheduler,
    }
    for key in [
        key for key in sys.modules if key == "nanovllm" or key.startswith("nanovllm.")
    ]:
        sys.modules.pop(key, None)
    sys.modules.update(saved_modules)
    if saved_transformers is None:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = saved_transformers
    return result


class NanoVLLMBackend:
    """Run one real nano-vLLM generate call and emit host-observed traces."""

    _VALID_PREFIX = {"on", "off"}
    _VALID_SCHEDULER = {"upstream_dynamic", "fixed_batch"}

    def __init__(
        self,
        model_path: str,
        engine_config: dict[str, Any],
        prefix_cache_mode: str,
        scheduler_mode: str,
        seed: int,
        emit: Callable[[str, dict[str, Any]], None],
        measurement_profile: str = "detailed",
    ) -> None:
        if measurement_profile not in {"minimal", "detailed"}:
            raise ValueError(f"invalid measurement_profile: {measurement_profile}")
        if measurement_profile == "minimal" and (
            prefix_cache_mode != "on" or scheduler_mode != "upstream_dynamic"
        ):
            raise ValueError(
                "minimal profile supports only prefix_cache_mode='on' and "
                "scheduler_mode='upstream_dynamic'"
            )
        if prefix_cache_mode not in self._VALID_PREFIX:
            raise ValueError(f"invalid prefix_cache_mode: {prefix_cache_mode}")
        if scheduler_mode not in self._VALID_SCHEDULER:
            raise ValueError(f"invalid scheduler_mode: {scheduler_mode}")
        self.model_path = model_path
        self.engine_config = dict(engine_config)
        self.prefix_cache_mode = prefix_cache_mode
        self.scheduler_mode = scheduler_mode
        self.measurement_profile = measurement_profile
        self.seed = seed
        self.emit = emit
        self.engine: Any = None
        self._cleanup_registered = False
        self._closed = False
        self._clock_id = str(uuid.uuid4())
        self._step_id = 0
        self._step_state: dict[str, Any] | None = None
        self._requests: dict[str, dict[str, Any]] = {}
        self._first_observed: dict[str, dict[str, Any]] = {}
        self._request_output_ids: dict[str, list[int]] = {}
        self._enqueue_ns: dict[str, int] = {}
        self._first_scheduled_ns: dict[str, int] = {}
        self._finish_ns: dict[str, int] = {}
        self._outputs: list[dict[str, Any]] = []
        self._source_hashes: dict[str, str] = {}

    @staticmethod
    def _stamp() -> tuple[str, int]:
        return datetime.now(UTC).isoformat(), time.monotonic_ns()

    def _event(self, table: str, **row: Any) -> None:
        utc, mono = self._stamp()
        payload = {
            "timestamp_utc": utc,
            "event_time_utc": utc,
            "monotonic_ns": mono,
            "clock_id": self._clock_id,
            "process_id": __import__("os").getpid(),
            "event_id": str(uuid.uuid4()),
            **row,
        }
        for key in [key for key, value in payload.items() if value is None]:
            payload.pop(key)
        self.emit(table, payload)

    def prepare(self) -> dict[str, Any]:
        if self.engine is not None:
            raise RuntimeError("backend already prepared")
        root = _upstream_root()
        self._source_hashes = verify_pinned_source(root)
        # GPU-bound imports occur only after the source pin is verified.
        engine_config = dict(self.engine_config)
        engine_config.setdefault("tensor_parallel_size", 1)
        try:
            import torch

            self._original_torch_dtype = torch.get_default_dtype()
            self._original_torch_device = torch.get_default_device()
        except (ImportError, AttributeError):
            self._original_torch_dtype = None
            self._original_torch_device = None
        config_module = importlib.import_module("nanovllm.config")
        accepted = {
            field.name
            for field in importlib.import_module("dataclasses").fields(
                config_module.Config
            )
        } - {"model"}
        unknown = set(self.engine_config) - accepted
        if unknown:
            raise ValueError(
                f"unsupported nano-vLLM engine config keys: {sorted(unknown)}"
            )
        llm_module = importlib.import_module("nanovllm.llm")
        try:
            self.engine = llm_module.LLM(self.model_path, **engine_config)
            self._warm_and_reset()
            # Reset sampling RNG after warmup, so each measured run starts at seed.
            import torch

            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            if self.measurement_profile == "detailed":
                self._patch_live_engine()
        except BaseException:
            self._cleanup_failed_prepare()
            raise
        actual_config = self.engine.model_runner.config
        runner = self.engine.model_runner
        kv_cache = runner.kv_cache
        kv_cache_bytes = int(kv_cache.numel() * kv_cache.element_size())
        graph_sizes = (
            list(getattr(runner, "graph_bs", [])) if not runner.enforce_eager else []
        )
        return {
            "engine": "nano-vllm",
            "engine_revision": PINNED_REVISION,
            "engine_source_hashes": self._source_hashes,
            "effective_config": {
                **engine_config,
                "model_dtype": str(actual_config.hf_config.dtype),
                "effective_max_model_len": actual_config.max_model_len,
                "kvcache_block_size": actual_config.kvcache_block_size,
                "num_kvcache_blocks": actual_config.num_kvcache_blocks,
                "kv_cache_dtype": str(kv_cache.dtype),
                "kv_cache_capacity_bytes": kv_cache_bytes,
                "cuda_graphs_configured": not runner.enforce_eager,
                "cuda_graph_capture_batch_sizes": graph_sizes,
                "sampling_rng": "torch CPU and CUDA seeded after warmup",
            },
            "prefix_cache_mode": self.prefix_cache_mode,
            "scheduler_mode": self.scheduler_mode,
            "scheduler_max_num_seqs": self.engine.scheduler.max_num_seqs,
            "upstream_dynamic_admission_limit": (
                "per-schedule only; running set may exceed max_num_seqs"
            ),
            "measurement_profile": self.measurement_profile,
            "primary_complete": self.measurement_profile == "detailed",
            "capabilities": {
                "real_nano_vllm": True,
                "detailed_trace_hooks": self.measurement_profile == "detailed",
                "primary_trace_hooks": [
                    "steps",
                    "step_requests",
                    "cache_events",
                    "scheduler_events",
                    "token_events",
                ]
                if self.measurement_profile == "detailed"
                else [],
                "prefix_cache_toggle": self.prefix_cache_mode in self._VALID_PREFIX,
                "fixed_wave_admission": self.scheduler_mode == "fixed_batch",
                "per_step_gpu_timestamps": False,
                "gpu_generated_token_timestamps": False,
            },
            "field_status": {
                "detailed_trace_hooks": {
                    "status": "supported"
                    if self.measurement_profile == "detailed"
                    else "not_enabled",
                    "reason": "host hooks installed after warmup"
                    if self.measurement_profile == "detailed"
                    else "minimal overhead profile skips detailed hooks",
                },
                "gpu_generated_token_timestamp": {
                    "status": "unsupported",
                    "reason": (
                        "pinned engine exposes sampled token IDs on host, "
                        "not per-token GPU event timestamps"
                    ),
                },
                "gpu_step_duration": {
                    "status": "unsupported",
                    "reason": (
                        "baseline avoids CUDA event instrumentation and "
                        "per-step synchronization"
                    ),
                },
            },
        }

    def _warm_and_reset(self) -> None:
        """Warm one disjoint prompt, then invalidate and verify cache state."""
        engine = self.engine
        prompt = engine.tokenizer.encode("TAEI adapter disjoint warmup prompt.")
        sampling = importlib.import_module("nanovllm.sampling_params").SamplingParams
        engine.generate(
            [prompt],
            [sampling(temperature=0.6, max_tokens=1, ignore_eos=True)],
            use_tqdm=False,
        )
        scheduler = engine.scheduler
        manager = scheduler.block_manager
        if scheduler.waiting or scheduler.running:
            raise RuntimeError("nano-vLLM warmup did not drain scheduler")
        manager.hash_to_block_id.clear()
        manager.used_block_ids.clear()
        manager.free_block_ids = deque(range(len(manager.blocks)))
        for block in manager.blocks:
            block.ref_count = 0
            block.hash = -1
            block.token_ids = []
        if (
            manager.hash_to_block_id
            or manager.used_block_ids
            or len(manager.free_block_ids) != len(manager.blocks)
        ):
            raise RuntimeError("nano-vLLM cache reset verification failed")

    def _cleanup_failed_prepare(self) -> None:
        engine = self.engine
        if engine is not None:
            self._restore()
            with contextlib.suppress(AttributeError, ValueError):
                atexit.unregister(engine.exit)
            with contextlib.suppress(Exception):
                engine.exit()
            self.engine = None
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
            except Exception:
                pass
            try:
                if self._original_torch_device is not None:
                    torch.set_default_device(self._original_torch_device)
                if self._original_torch_dtype is not None:
                    torch.set_default_dtype(self._original_torch_dtype)
            except Exception:
                pass

    def synchronize(self) -> None:
        if self.engine is None:
            raise RuntimeError("backend is not prepared")
        import torch

        torch.cuda.synchronize()

    def _patch_live_engine(self) -> None:
        engine = self.engine
        scheduler = engine.scheduler
        block_manager = scheduler.block_manager
        self._originals: list[tuple[Any, str, Any]] = []
        self._seq_to_request_id: dict[int, str] = {}
        self._pending_request_ids: Any = iter(())
        self._cache_origin_by_block: dict[int, str] = {}
        self._cache_reuse_by_seq: dict[int, dict[str, int]] = {}
        self._cache_reuse_reported: set[int] = set()
        self._recompute_pending: set[str] = set()

        def replace(obj: Any, name: str, wrapper: Callable[..., Any]) -> None:
            original = getattr(obj, name)
            self._originals.append((obj, name, original))
            setattr(obj, name, wrapper(original))

        def cache_call(name: str) -> Callable[..., Any]:
            def decorate(original: Callable[..., Any]) -> Callable[..., Any]:
                def call(*args: Any, **kwargs: Any) -> Any:
                    seq = args[0] if args else None
                    hash_range = None
                    if name == "hash_blocks" and seq is not None:
                        block_size = block_manager.block_size
                        hash_start = seq.num_cached_tokens // block_size
                        hash_end = (
                            seq.num_cached_tokens + seq.num_scheduled_tokens
                        ) // block_size
                        hash_range = range(hash_start, hash_end)
                    before = self._block_snapshot(block_manager)
                    result = original(*args, **kwargs)
                    if name == "allocate" and seq is not None:
                        cached_blocks = int(args[1])
                        request_id = self._request_id_for_seq(seq)
                        origins = [
                            self._cache_origin_by_block.get(block_id)
                            for block_id in seq.block_table[:cached_blocks]
                        ]
                        self._cache_reuse_by_seq[seq.seq_id] = {
                            "cross_request_prefix_tokens": sum(
                                block_manager.block_size
                                for origin in origins
                                if origin is not None and origin != request_id
                            ),
                            "same_request_cached_tokens": sum(
                                block_manager.block_size
                                for origin in origins
                                if origin == request_id
                            ),
                        }
                        self._cache_reuse_reported.discard(seq.seq_id)
                    if (
                        name == "hash_blocks"
                        and seq is not None
                        and hash_range is not None
                    ):
                        request_id = self._request_id_for_seq(seq)
                        if request_id is not None:
                            for index in hash_range:
                                if index < len(seq.block_table):
                                    self._cache_origin_by_block[
                                        seq.block_table[index]
                                    ] = request_id
                    after = self._block_snapshot(block_manager)
                    if name == "can_allocate" and self.prefix_cache_mode == "off":
                        return result
                    touched = set(getattr(seq, "block_table", []))
                    self._event(
                        "cache_events",
                        event_id=str(uuid.uuid4()),
                        event_type={
                            "can_allocate": "lookup",
                            "allocate": "allocate",
                            "deallocate": "free",
                            "may_append": "allocate",
                            "hash_blocks": "unknown",
                        }.get(name, "unknown"),
                        operation=name,
                        step_id=self._step_key() if self._step_id else None,
                        request_id=self._request_id_for_seq(seq),
                        result=result
                        if isinstance(result, (str, int, float, bool, type(None)))
                        else None,
                        eligible_prefix_blocks=max(seq.num_blocks - 1, 0)
                        if name == "can_allocate" and seq is not None
                        else None,
                        eligible_blocks=max(seq.num_blocks - 1, 0)
                        if name == "can_allocate" and seq is not None
                        else None,
                        prefix_hit_blocks=int(result)
                        if name == "can_allocate"
                        and isinstance(result, int)
                        and result >= 0
                        else None,
                        hit_blocks=int(result)
                        if name == "can_allocate"
                        and isinstance(result, int)
                        and result >= 0
                        else None,
                        miss_blocks=max(seq.num_blocks - 1 - result, 0)
                        if name == "can_allocate"
                        and seq is not None
                        and isinstance(result, int)
                        and result >= 0
                        else None,
                        block_id=next(iter(touched)) if len(touched) == 1 else None,
                        refcount=block_manager.blocks[next(iter(touched))].ref_count
                        if len(touched) == 1
                        else None,
                        touched_block_ids=sorted(touched),
                        block_states={
                            block_id: {
                                "ref_count": block_manager.blocks[block_id].ref_count,
                                "hash": block_manager.blocks[block_id].hash,
                            }
                            for block_id in sorted(touched)
                        },
                        before=before,
                        after=after,
                        prefix_cache_mode=self.prefix_cache_mode,
                    )
                    return result

                return call

            return decorate

        if self.prefix_cache_mode == "off":

            def no_lookup(original: Callable[..., Any]) -> Callable[..., Any]:
                def call(seq: Any) -> int:
                    needed = seq.num_blocks
                    free = len(block_manager.free_block_ids)
                    result = 0 if free >= needed else -1
                    self._event(
                        "cache_events",
                        event_id=str(uuid.uuid4()),
                        event_type="capacity_check",
                        operation="can_allocate",
                        step_id=self._step_key() if self._step_id else None,
                        request_id=self._request_id_for_seq(seq),
                        prefix_cache_mode="off",
                        eligible_blocks=needed,
                        result=result,
                    )
                    return result

                return call

            replace(block_manager, "can_allocate", no_lookup)
            replace(block_manager, "hash_blocks", lambda original: lambda seq: None)
        else:
            replace(block_manager, "can_allocate", cache_call("can_allocate"))
            replace(block_manager, "hash_blocks", cache_call("hash_blocks"))
        for name in ("allocate", "deallocate", "may_append"):
            replace(block_manager, name, cache_call(name))

        def schedule_wrap(original: Callable[..., Any]) -> Callable[..., Any]:
            def call(*args: Any, **kwargs: Any) -> Any:
                before_waiting = list(scheduler.waiting)
                before_running = list(scheduler.running)
                schedule_started = time.monotonic_ns()
                held_waiting = []
                if self.scheduler_mode == "fixed_batch":
                    self._fixed_prepare_wave(scheduler)
                    wave_ids = self._wave_members
                    eligible = [
                        seq for seq in scheduler.waiting if seq.seq_id in wave_ids
                    ]
                    held_waiting = [
                        seq for seq in scheduler.waiting if seq.seq_id not in wave_ids
                    ]
                    scheduler.waiting = deque(eligible)
                selected, prefill = original(*args, **kwargs)
                if self.scheduler_mode == "fixed_batch":
                    still_waiting = set(scheduler.waiting)
                    scheduler.waiting = deque(
                        seq
                        for seq in before_waiting
                        if seq in still_waiting or seq in held_waiting
                    )
                    self._fixed_note_scheduled(selected)
                scheduled_at = time.monotonic_ns()
                for seq in selected:
                    request_id = self._request_id_for_seq(seq)
                    if request_id is not None:
                        self._first_scheduled_ns.setdefault(request_id, scheduled_at)
                preempted = [seq for seq in before_running if seq in scheduler.waiting]
                for seq in preempted:
                    request_id = self._request_id_for_seq(seq)
                    if request_id is not None:
                        self._recompute_pending.add(request_id)
                sequence_snapshots = {}
                for seq in selected:
                    request_id = self._request_id_for_seq(seq)
                    cache_stats = self._cache_reuse_by_seq.get(seq.seq_id, {})
                    report_cache = (
                        prefill and seq.seq_id not in self._cache_reuse_reported
                    )
                    sequence_snapshots[seq.seq_id] = {
                        "num_scheduled_tokens": seq.num_scheduled_tokens,
                        "num_cached_tokens": seq.num_cached_tokens,
                        "num_tokens": seq.num_tokens,
                        "block_ids": list(seq.block_table),
                        "cross_request_prefix_tokens": (
                            cache_stats.get("cross_request_prefix_tokens", 0)
                            if report_cache
                            else 0
                        ),
                        "same_request_cached_tokens": (
                            cache_stats.get("same_request_cached_tokens", 0)
                            if report_cache
                            else 0
                        ),
                        "is_recompute": bool(
                            prefill and request_id in self._recompute_pending
                        ),
                    }
                    if report_cache:
                        self._cache_reuse_reported.add(seq.seq_id)
                    if prefill and request_id is not None:
                        self._recompute_pending.discard(request_id)
                state = {
                    "schedule_start_ns": schedule_started,
                    "schedule_end_ns": scheduled_at,
                    "seqs": selected,
                    "is_prefill": prefill,
                    "waiting_before": before_waiting,
                    "running_before": before_running,
                    "waiting_after": list(scheduler.waiting),
                    "running_after": list(scheduler.running),
                    "preempted": preempted,
                    "seq_snapshot": sequence_snapshots,
                }
                self._step_state = state
                self._event(
                    "scheduler_events",
                    event_type="schedule",
                    step_id=self._step_key(),
                    request_ids=[self._request_id_for_seq(seq) for seq in selected],
                    request_count=len(selected),
                    reason="prefill" if prefill else "decode",
                    wave_id=f"wave-{self._wave_id:04d}"
                    if self.scheduler_mode == "fixed_batch"
                    else None,
                    scheduler_mode=self.scheduler_mode,
                    is_prefill=prefill,
                    admitted_request_ids=[
                        self._request_id_for_seq(s) for s in selected
                    ],
                    waiting_before=[
                        self._request_id_for_seq(s) for s in before_waiting
                    ],
                    running_before=[
                        self._request_id_for_seq(s) for s in before_running
                    ],
                    queue_count_before=len(before_waiting),
                    running_count_before=len(before_running),
                    queue_after=len(scheduler.waiting),
                    running_after=len(scheduler.running),
                    active_count_after=len(scheduler.running),
                    preempted_request_ids=[
                        self._request_id_for_seq(seq)
                        for seq in before_running
                        if seq in scheduler.waiting
                    ],
                    active_count_may_exceed_max_num_seqs=(
                        self.scheduler_mode == "upstream_dynamic"
                        and len(scheduler.running) > scheduler.max_num_seqs
                    ),
                )
                return selected, prefill

            return call

        replace(scheduler, "schedule", schedule_wrap)

        def add_request_wrap(original: Callable[..., Any]) -> Callable[..., Any]:
            def call(prompt: Any, sampling_params: Any) -> None:
                original(prompt, sampling_params)
                seq = scheduler.waiting[-1]
                ids = getattr(self, "_pending_request_ids", None)
                request_id = (
                    next(ids, "__warmup__") if ids is not None else "__warmup__"
                )
                self._seq_to_request_id[seq.seq_id] = request_id
                if request_id != "__warmup__":
                    self._enqueue_ns[request_id] = time.monotonic_ns()
                    self._request_output_ids[request_id] = []
                    self._event(
                        "scheduler_events",
                        event_type="admit",
                        request_id=request_id,
                        request_ids=[request_id],
                        request_count=1,
                        reason="enqueue",
                        wave_id=None,
                        queue_wait_ns=None,
                    )

            return call

        replace(engine, "add_request", add_request_wrap)

        def call_wrap(original: Callable[..., Any]) -> Callable[..., Any]:
            def call(method: str, *args: Any, **kwargs: Any) -> Any:
                if method != "run":
                    return original(method, *args, **kwargs)
                started = time.monotonic_ns()
                result = original(method, *args, **kwargs)
                observed = time.monotonic_ns()
                state = self._step_state or {}
                seqs = state.get("seqs", [])
                snapshots = state.get("seq_snapshot", {})
                produced_by_seq = {}
                for seq, token_id in zip(seqs, result or [], strict=True):
                    request_id = self._request_id_for_seq(seq)
                    snap = snapshots.get(seq.seq_id, {})
                    is_discarded_prefill_sample = state.get("is_prefill") and snap.get(
                        "num_cached_tokens", 0
                    ) + snap.get("num_scheduled_tokens", 0) < snap.get("num_tokens", 0)
                    produced_by_seq[seq.seq_id] = int(not is_discarded_prefill_sample)
                    if request_id is not None and request_id != "__warmup__":
                        if not is_discarded_prefill_sample:
                            output_index = len(
                                self._request_output_ids.setdefault(request_id, [])
                            )
                            self._request_output_ids[request_id].append(int(token_id))
                            self._event(
                                "token_events",
                                request_id=request_id,
                                step_id=self._step_key(),
                                token_index=output_index,
                                token_id=int(token_id),
                                token_type="sampled_output",
                                token_origin="sampled_output",
                                gpu_generated_timestamp_ns=None,
                                host_observed_timestamp_ns=observed,
                            )
                            self._first_observed.setdefault(
                                request_id,
                                {
                                    "first_host_observed_token_ns": observed,
                                    "first_generated_token_ns": None,
                                },
                            )
                        else:
                            state["discarded_prefill_sample_count"] = (
                                state.get("discarded_prefill_sample_count", 0) + 1
                            )
                state["modelcall_start_ns"] = started
                state["modelcall_end_ns"] = observed
                state["sampled_token_ids"] = result
                batch_size = len(seqs)
                model_runner = self.engine.model_runner
                if (
                    state.get("is_prefill")
                    or model_runner.enforce_eager
                    or batch_size > 512
                ):
                    state["graph_padded_batch_size"] = None
                else:
                    state["graph_padded_batch_size"] = next(
                        (size for size in model_runner.graph_bs if size >= batch_size),
                        None,
                    )
                state["produced_by_seq"] = produced_by_seq
                state["produced_token_count"] = sum(produced_by_seq.values())
                return result

            return call

        replace(engine.model_runner, "call", call_wrap)

        def step_wrap(original: Callable[..., Any]) -> Callable[..., Any]:
            def call(*args: Any, **kwargs: Any) -> Any:
                self._step_id += 1
                step_start = time.monotonic_ns()
                result = original(*args, **kwargs)
                step_end = time.monotonic_ns()
                state = self._step_state or {}
                seqs = state.get("seqs", [])
                snapshots = state.get("seq_snapshot", {})
                request_ids = [self._request_id_for_seq(seq) for seq in seqs]
                if not all(
                    rid is not None and rid != "__warmup__" for rid in request_ids
                ):
                    raise RuntimeError(
                        "measured scheduler step has unresolved request identity"
                    )
                is_prefill = bool(state.get("is_prefill"))
                token_count = (
                    sum(snap["num_scheduled_tokens"] for snap in snapshots.values())
                    if is_prefill
                    else len(seqs)
                )
                produced_count = state.get("produced_token_count", 0)
                fresh_prefill_count = sum(
                    snap["num_scheduled_tokens"]
                    for snap in snapshots.values()
                    if is_prefill and not snap.get("is_recompute")
                )
                recompute_count = sum(
                    snap["num_scheduled_tokens"]
                    for snap in snapshots.values()
                    if is_prefill and snap.get("is_recompute")
                )
                cross_request_reused = (
                    sum(
                        snap.get("cross_request_prefix_tokens", 0)
                        for snap in snapshots.values()
                    )
                    if is_prefill
                    else 0
                )
                step_key = self._step_key()
                self._event(
                    "steps",
                    step_id=step_key,
                    step_index=self._step_id,
                    phase="prefill" if is_prefill else "decode",
                    request_ids=request_ids,
                    contexts_before={
                        self._request_id_for_seq(seq): snapshots[seq.seq_id][
                            "num_tokens"
                        ]
                        for seq in seqs
                    },
                    contexts_after={
                        self._request_id_for_seq(seq): len(seq) for seq in seqs
                    },
                    scheduled_tokens=token_count,
                    produced_tokens=produced_count,
                    fresh_prefill_tokens=fresh_prefill_count,
                    recompute_tokens=recompute_count,
                    prefix_reused_tokens=cross_request_reused,
                    same_request_cached_tokens=sum(
                        snap.get("same_request_cached_tokens", 0)
                        for snap in snapshots.values()
                    ),
                    discarded_prefill_samples=state.get(
                        "discarded_prefill_sample_count", 0
                    ),
                    actual_batch_size=len(seqs),
                    batch_size=len(seqs),
                    graph_padded_batch_size=state.get("graph_padded_batch_size"),
                    queue_before=len(state.get("waiting_before", [])),
                    queue_after=len(self.engine.scheduler.waiting),
                    running_before=len(state.get("running_before", [])),
                    running_after=len(self.engine.scheduler.running),
                    schedule_start_ns=state.get("schedule_start_ns"),
                    schedule_end_ns=state.get("schedule_end_ns"),
                    modelcall_start_ns=state.get("modelcall_start_ns"),
                    modelcall_end_ns=state.get("modelcall_end_ns"),
                    postprocess_start_ns=state.get("modelcall_end_ns"),
                    postprocess_end_ns=step_end,
                    step_host_start_ns=step_start,
                    step_host_end_ns=step_end,
                    schedule_start_monotonic_ns=state.get("schedule_start_ns"),
                    schedule_end_monotonic_ns=state.get("schedule_end_ns"),
                    model_call_start_monotonic_ns=state.get("modelcall_start_ns"),
                    model_call_end_monotonic_ns=state.get("modelcall_end_ns"),
                    postprocess_start_monotonic_ns=state.get("modelcall_end_ns"),
                    postprocess_end_monotonic_ns=step_end,
                    gpu_event_duration_ns=None,
                )
                for sequence_index, seq in enumerate(seqs):
                    request_id = self._request_id_for_seq(seq)
                    snap = snapshots.get(seq.seq_id, {})
                    self._event(
                        "step_requests",
                        step_id=step_key,
                        request_id=request_id,
                        relation="prefill" if is_prefill else "decode",
                        token_count=snap.get("num_scheduled_tokens", 0),
                        produced_tokens=state.get("produced_by_seq", {}).get(
                            seq.seq_id, 0
                        ),
                        fresh_prefill_tokens=(
                            snap.get("num_scheduled_tokens", 0)
                            if is_prefill and not snap.get("is_recompute")
                            else 0
                        ),
                        recompute_tokens=(
                            snap.get("num_scheduled_tokens", 0)
                            if is_prefill and snap.get("is_recompute")
                            else 0
                        ),
                        sequence_index=sequence_index,
                        context_tokens_before=snap.get("num_tokens"),
                        context_tokens_after=len(seq),
                        private_kv_cached_tokens=snap.get("num_cached_tokens", 0),
                        prefix_reused_tokens=snap.get("cross_request_prefix_tokens", 0),
                        scheduled_tokens=snap.get("num_scheduled_tokens", 0),
                        block_ids=snap.get("block_ids", []),
                    )
                for seq_id, completion_ids in result[0]:
                    request_id = self._seq_to_request_id.get(seq_id)
                    if request_id is not None:
                        self._finish_ns[request_id] = step_end
                        request = self._requests.get(request_id, {})
                        reason = "length"
                        if len(completion_ids) < int(
                            request.get("requested_output_tokens", len(completion_ids))
                        ):
                            if (
                                not request.get("ignore_eos")
                                and completion_ids
                                and completion_ids[-1] == self.engine.scheduler.eos
                            ):
                                reason = "eos"
                            else:
                                reason = "unknown"
                        self._event(
                            "scheduler_events",
                            event_type="complete",
                            request_id=request_id,
                            request_ids=[request_id],
                            request_count=1,
                            reason=reason,
                            wave_id=f"wave-{self._wave_id:04d}"
                            if self.scheduler_mode == "fixed_batch"
                            else None,
                            queue_wait_ns=None,
                        )
                self._step_state = None
                return result

            return call

        replace(engine, "step", step_wrap)

    def _block_snapshot(self, manager: Any) -> dict[str, Any]:
        """Return O(1) allocator counters; never copy the capacity-sized block map."""
        return {
            "capacity_blocks": len(manager.blocks),
            "free_block_count": len(manager.free_block_ids),
            "used_block_count": len(manager.used_block_ids),
            "hashed_block_count": len(manager.hash_to_block_id),
            "free_queue_head": manager.free_block_ids[0]
            if manager.free_block_ids
            else None,
        }

    def _step_key(self) -> str:
        return f"step-{self._step_id:06d}"

    def _request_id_for_seq(self, seq: Any) -> str | None:
        if seq is None:
            return None
        return getattr(self, "_seq_to_request_id", {}).get(seq.seq_id)

    def _fixed_prepare_wave(self, scheduler: Any) -> None:
        live = {seq.seq_id for seq in scheduler.waiting} | {
            seq.seq_id for seq in scheduler.running
        }
        if getattr(self, "_wave_active", False):
            self._wave_members.intersection_update(live)
            if self._wave_members:
                return
            self._wave_active = False
        # The full waiting deque remains intact; schedule_wrap temporarily filters
        # it to this wave and restores held entries before returning.
        members = list(scheduler.waiting)[: scheduler.max_num_seqs]
        self._wave_id = getattr(self, "_wave_number", 0)
        self._wave_number = self._wave_id + 1
        self._wave_members = {seq.seq_id for seq in members}
        self._wave_active = bool(members)

    def _fixed_note_scheduled(self, selected: list[Any]) -> None:
        self._wave_members.update(seq.seq_id for seq in selected)

    def _restore(self) -> None:
        for obj, name, original in reversed(getattr(self, "_originals", [])):
            setattr(obj, name, original)
        self._originals = []

    def generate(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.engine is None or self._closed:
            raise RuntimeError("backend must be prepared and open before generate")
        if self._requests:
            raise RuntimeError("backend supports one workload generate call")
        prompts: list[list[int]] = []
        params: list[Any] = []
        sampling = importlib.import_module("nanovllm.sampling_params").SamplingParams
        for request in requests:
            request_id = str(request["request_id"])
            self._requests[request_id] = dict(request)
            prompts.append([int(token) for token in request["input_token_ids"]])
            params.append(
                sampling(
                    temperature=float(request["temperature"]),
                    max_tokens=int(request["requested_output_tokens"]),
                    ignore_eos=bool(request["ignore_eos"]),
                )
            )
        if self.measurement_profile == "detailed":
            self._pending_request_ids = iter(
                str(request["request_id"]) for request in requests
            )
        started = time.monotonic_ns()
        # One upstream generate call receives the entire workload in every mode.
        upstream_outputs = self.engine.generate(prompts, params, use_tqdm=False)
        ended = time.monotonic_ns()
        if self.measurement_profile == "minimal":
            # Pinned generate preserves sequence order in its returned list; this is
            # the real upstream output, not a reconstructed or seeded substitute.
            for request, output in zip(requests, upstream_outputs, strict=True):
                request_id = str(request["request_id"])
                self._request_output_ids[request_id] = [
                    int(token) for token in output["token_ids"]
                ]
        records = []
        for request in requests:
            request_id = str(request["request_id"])
            emitted = self._request_output_ids.get(request_id, [])
            records.append(
                {
                    "request_id": request_id,
                    "input_token_ids": list(request["input_token_ids"]),
                    "output_token_ids": emitted,
                    "input_token_count": len(request["input_token_ids"]),
                    "output_token_count": len(emitted),
                    "actual_output_tokens": len(emitted),
                    "enqueue_monotonic_ns": self._enqueue_ns.get(request_id),
                    "first_scheduled_monotonic_ns": self._first_scheduled_ns.get(
                        request_id
                    ),
                    "first_generated_monotonic_ns": None,
                    "first_host_observed_monotonic_ns": self._first_observed.get(
                        request_id, {}
                    ).get("first_host_observed_token_ns"),
                    "last_generated_monotonic_ns": None,
                    "first_host_observed_token_ns": self._first_observed.get(
                        request_id, {}
                    ).get("first_host_observed_token_ns"),
                    "first_generated_token_ns": None,
                    "finish_monotonic_ns": self._finish_ns.get(request_id, ended),
                    "generate_start_monotonic_ns": started,
                    "generate_end_monotonic_ns": ended,
                    "finish_reason": "length"
                    if len(emitted) == int(request["requested_output_tokens"])
                    else "unknown",
                    "status": "success" if emitted else "incomplete",
                    "gpu_generated_timestamp_status": "unsupported",
                    "field_status": {
                        "first_generated_monotonic_ns": {
                            "status": "unsupported",
                            "reason": (
                                "pinned upstream exposes no GPU token "
                                "generation timestamp"
                            ),
                        },
                        "last_generated_monotonic_ns": {
                            "status": "unsupported",
                            "reason": (
                                "pinned upstream exposes no GPU token "
                                "generation timestamp"
                            ),
                        },
                        "first_host_observed_monotonic_ns": {
                            "status": "supported"
                            if self.measurement_profile == "detailed"
                            and request_id in self._first_observed
                            else "not_enabled",
                            "reason": "model runner returned sampled token ID on host"
                            if request_id in self._first_observed
                            else "minimal profile skips per-step host observation hook",
                        },
                    },
                }
            )
        return records

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        engine = self.engine
        self._restore()
        if engine is not None:
            with contextlib.suppress(AttributeError, ValueError):
                atexit.unregister(engine.exit)
            # The pinned exit method performs model-runner/NCCL shutdown. Avoid masking
            # a primary generation exception at the call site; surface close failures.
            engine.exit()
            self.engine = None
