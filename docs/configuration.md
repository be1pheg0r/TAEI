# Configuration layout

- `configs/config.yaml` composes the default engine, model, and experiment groups.
- `configs/engine/` holds backend and runtime settings; `nano_vllm.yaml` is the initial backend adapter.
- `configs/model/` identifies a checkpoint and its local path; model choices are independent of engine settings.
- `configs/experiment/` holds a named run's inputs and generation/evaluation choices. `smoke.yaml` is a placeholder.

This tree is a configuration scaffold, not an inference entry point. Set `TAEI_QWEN3_8B_PATH` to a downloaded model directory when the default path does not apply. The model group records upstream Qwen3-8B architecture metadata; nano-vllm reads the model's actual `config.json` from the selected directory.

Engine values are starting points, not benchmarked limits. Select appropriate devices for the host and validate memory use, context length, and concurrency with a smoke test before a real run. These configs do not launch a GPU job.

Hydra can select a group option and override values at launch, for example `model=qwen3_8b_dense engine.max_num_seqs=8`, once a Hydra-backed runner is added.

## Machine-specific notes

Host hardware selection, device indices, and tested or intended per-host launch settings are kept in the ignored `private_specs/` directory and are not part of the general project documentation.
