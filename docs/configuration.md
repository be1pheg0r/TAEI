# Configuration and experiments

Hydra composes `configs/config.yaml` from the engine, model, and experiment groups. The default configuration now selects a CPU-safe dummy smoke experiment; it does not load a model, launch nano-vLLM, require a GPU, or connect to ClearML.

Run the default smoke experiment or the dedicated log-collection example from the repository root:

```bash
uv run python -m taei.run_experiment
uv run python -m taei.run_experiment experiment=dummy
uv run python -m taei.run_experiment experiment=dummy experiment.metrics.score=0.9
```

The dummy experiment emits each configured metric through `taei.logging` and returns the metric mapping. It exists to exercise experiment construction and log collection, not to benchmark inference. Hydra writes its run metadata under its normal output directory; use `hydra.run.dir=...` to override that behavior.

Configuration groups:

- `configs/engine/` contains backend settings. `nano_vllm.yaml` is available for future inference experiments; selecting it does not cause this dummy runner to import or start nano-vLLM.
- `configs/model/` contains checkpoint metadata. Set `TAEI_QWEN3_8B_PATH` when a real model workflow needs a different local path.
- `configs/experiment/` contains experiment classes and their inputs. Both `smoke.yaml` and `dummy.yaml` target `DummyExperiment` and run without external services.

The current experiment runner deliberately treats the engine/model groups as configuration only. A future inference experiment should implement `taei.experiments.base.Experiment` and provide a Hydra `_target_` plus constructor fields in its experiment YAML. Host-specific device settings belong in the ignored `private_specs/` directory, not general docs.
