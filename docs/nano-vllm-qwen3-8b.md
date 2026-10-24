# Qwen3-8B RTX A4000 TP2 sibling profile

This model-identity derivative retains the `rtx_a4000_tp2` 24-run workload, scheduler/cache policy, 32-request batching, TP=2, eager mode, 30-minute attempt timeout, and attempt layout. It uses a separate experiment/output root and pins `Qwen/Qwen3-8B` to `b968826d9c46dd6066d109eabc6255188de91218`; its local model path defaults to `~/.cache/taei/models/Qwen3-8B` and can be set with `TAEI_QWEN3_8B_PATH`.

The pinned `config.json` identifies `Qwen3ForCausalLM`/`qwen3`, bfloat16, 36 layers, 32 attention heads, 8 KV heads, and `max_position_embeddings=40960`. The experiment's `max_model_len=4096` is within that limit. No 0.6B download identity is inherited.

From the repository root, prepare dependencies and acquire the pinned model outside measured runs:

```bash
uv sync --locked --extra inference
uv run --locked --extra inference python experiments/nano_vllm_dataset/run.py \
  profile@_global_=rtx_a4000_tp2_qwen3_8b mode=download
uv run --locked --extra inference python experiments/nano_vllm_dataset/run.py \
  profile@_global_=rtx_a4000_tp2_qwen3_8b mode=preflight
```

Inspect/plan without model acquisition or inference:

```bash
uv run --locked --extra inference python experiments/nano_vllm_dataset/run.py \
  profile@_global_=rtx_a4000_tp2_qwen3_8b mode=plan
```

After successful preflight on the intended GPU host, launch the complete matrix (no Hydra multirun):

```bash
uv run --locked --extra inference python experiments/nano_vllm_dataset/run.py \
  profile@_global_=rtx_a4000_tp2_qwen3_8b mode=run dataset.run_limit=24
```

Model acquisition and GPU execution are not performed by CPU-only plan/configuration checks. Set host-specific GPU selection in the shell; do not stop existing GPU processes to make room.
