# Hydra config layout

- `config.yaml` composes the default engine, model, and experiment groups.
- `engine/` holds backend/runtime settings; `nano_vllm.yaml` is the first backend adapter.
- `model/` identifies a checkpoint and its local path; model choices are independent of engine settings.
- `experiment/` holds a named run's inputs and generation/evaluation choices. `smoke.yaml` is only a placeholder.

The current tree is a configuration scaffold, not an inference entry point. `TAEI_QWEN3_8B_PATH` can override the local checkpoint path; nano-vllm expects a downloaded model directory. The model group records the upstream Qwen3-8B architecture metadata, while nano-vllm reads the actual `config.json` from that directory.

The initial engine settings target the two 16 GiB RTX A4000s shown on the inference server. Start the future runner with `CUDA_VISIBLE_DEVICES=1,2` so nano-vllm sees only those cards, then use tensor parallelism of 2. The `4096` context and four concurrent sequences are conservative starting values, not benchmarked limits; adjust after a real server-side smoke test. The four 8 GiB Quadro RTX 4000s are excluded. No GPU job is launched by these configs.

Hydra can select a group option and override values at launch, e.g. `model=qwen3_8b_dense engine.max_num_seqs=8`, once a Hydra-backed runner is added.
