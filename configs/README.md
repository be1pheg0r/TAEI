# Hydra config layout

- `config.yaml` composes the default engine, model, and experiment groups.
- `engine/` holds backend/runtime settings; `nano_vllm.yaml` is the first backend adapter.
- `model/` identifies a checkpoint and its local path; model choices are independent of engine settings.
- `experiment/` holds a named run's inputs and generation/evaluation choices. `smoke.yaml` is only a placeholder.

The current tree is a configuration scaffold, not an inference entry point. `TAEI_QWEN3_8B_PATH` can override the example checkpoint path. GPU sizing, model path, input data, evaluation, and ClearML tracking still need to be set or verified on the target server before a real run. No GPU job is launched by these configs.

Hydra can select a group option and override values at launch, e.g. `model=qwen3_8b_dense engine.max_num_seqs=8`, once a Hydra-backed runner is added.
