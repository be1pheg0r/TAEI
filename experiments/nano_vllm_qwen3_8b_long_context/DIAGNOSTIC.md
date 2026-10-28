# Qwen3-8B cache-on diagnostic

This profile plans one attempt only: the first long-context matrix row, `L09`, with prefix cache `on`, `upstream_dynamic` scheduling, seed 42, and workload `qwen3_8b_long_context_set_02`. It retains the archived workload: 16 requests (8 hit targets plus 8 miss controls), prompts of 5,121–30,209 tokens, 282,640 total prompt tokens, 20,480 requested output tokens, and a 32,257-token maximum prompt-plus-output length. The eight hit targets use 1, 6, 15, 27, 43, 62, 85, and 112 shared 256-token blocks. Model/engine settings are Qwen3-8B, TP=2, BF16, 32,768 max context, eager execution, 16,384 max batched tokens, 8 max sequences, and 0.9 GPU memory utilization.

The uploaded archive `/root/qwen-3-8b-768-big.zip` recorded 96 attempts, 48 failures. Failed `runs.json` rows include `AssertionError` with an empty message at `prepare_or_generate`; the archived supervisor stderr contains no traceback. This is the observed reporting gap, not an established explanation for the assertion. The archive's first row, L09, is a failed cache-on/upstream-dynamic attempt with the workload/config above; it is the single diagnostic selection here. This run is for diagnosis, not a cache-vs-no-cache comparison or a cause claim.

Run from the repository root on the GPU host, after confirming the configured local model path resolves to the intended Qwen3-8B assets:

```bash
cd /root/TAEI
uv run --locked --extra inference python experiments/nano_vllm_qwen3_8b_long_context/run.py --config-name diagnostic mode=run
```

The config sets `dataset.run_limit: 1`; the actual planned matrix has 96 rows, but the production entrypoint slices it to the first row, L09. There is one measured `engine.generate` workload call. Engine warmup and the eight prefix-donor calls are setup outside the measured window, as in the source protocol. Hydra creates an invocation-specific output directory under `outputs/nano_vllm_qwen3_8b_long_context_cache_on_diagnostic/`; `config.yaml`, `run.log`, `hydra.log`, and `runs/L09/<attempt-id>/` remain there. On caught preparation/generation exceptions, `runs.json` now stores the full Python traceback and `traceback.txt` stores the same text; inherited child stdout/stderr continues through the run transcript. The run's final dataset summary reports one attempt. Do not rerun the measured command to obtain a second sample; inspect the created attempt/logs first.

No measured GPU run was performed in this workspace: `nvidia-smi` and `/dev/nvidia*` are unavailable, and the project environment reports CUDA unavailable. CPU regression tests exercise the real isolated worker/attempt production path using a controlled import-time exception; they verify traceback persistence, not the archived GPU failure or CUDA behavior.
