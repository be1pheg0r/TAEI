# Qwen3-8B cache-on diagnostic

This diagnostic profile plans exactly one launch (`D01`) with prefix cache `on`, `upstream_dynamic` scheduling, seed 42, and a separate diagnostic workload protocol. It contains 8 measured requests: four hit targets and four miss controls, each with exactly 13,000 input token IDs and a requested output budget of 1,000 tokens (104,000 input and 8,000 requested output tokens total). Hit prompts share 2,048, 4,096, 6,144, and 8,192-token prefixes with their corresponding cache donors; each donor adds one disjoint tail token to keep the shared prefix cacheable. Miss prompts have no donor and their first 256 token IDs differ from the corresponding hit donor, preventing a full leading-block match. The first cache intent pair is not a separately measured run: all eight requests are in the single measured workload call. Donor priming is setup outside that measured call. Model/engine settings remain Qwen3-8B, TP=2, BF16, 32,768 max context, eager execution, 16,384 max batched tokens, 8 max sequences, and 0.9 GPU memory utilization.

The diagnostic builder/protocol is separate from `qwen3_8b_long_context_32k_v1`; it does not change the historical experiment matrix or its workload lengths. Synthetic token IDs are not tokenizer-produced text, and this prepared-cache diagnostic is not a cache-off comparison, an end-to-end cache-population measurement, a performance study, or a guaranteed reproduction of the archived failure.

Historical context: the uploaded archive `/root/qwen-3-8b-768-big.zip` recorded 96 attempts, 48 failures. Failed `runs.json` rows include `AssertionError` with an empty message at `prepare_or_generate`; the archived supervisor stderr contains no traceback. This is the observed reporting gap, not an established explanation for the assertion. The diagnostic below uses a deliberately different workload from the archive and may not reproduce its failure. It is for diagnosis, not a cache-vs-no-cache comparison or a cause claim.

Run from the repository root on the GPU host, after confirming the configured local model path resolves to the intended Qwen3-8B assets:

```bash
cd /root/TAEI
uv run --locked --extra inference python experiments/nano_vllm_qwen3_8b_long_context/run.py --config-name diagnostic mode=run
```

The config sets `dataset.run_limit: 1`, and the diagnostic plan itself contains exactly one row, `D01`; the entrypoint therefore executes one attempt and one measured `engine.generate` workload call. Engine warmup and four prefix-donor calls are setup outside the measured window. Hydra creates an invocation-specific output directory under `outputs/nano_vllm_qwen3_8b_long_context_cache_on_diagnostic/`; `config.yaml`, `run.log`, `hydra.log`, and `runs/D01/<attempt-id>/` remain there. On caught preparation/generation exceptions, `runs.json` stores the full Python traceback and `traceback.txt` stores the same text; inherited child stdout/stderr continues through the run transcript. The run's final dataset summary reports one attempt. Do not rerun the measured command to obtain a second sample; inspect the created attempt/logs first.

No measured GPU run was performed in this workspace: `nvidia-smi` and `/dev/nvidia*` are unavailable, and the project environment reports CUDA unavailable. CPU regression tests exercise the real isolated worker/attempt production path using a controlled import-time exception; they verify traceback persistence, not the archived GPU failure or CUDA behavior.
