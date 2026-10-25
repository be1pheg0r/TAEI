# nano-vLLM: six-set paired cache experiment

The active `six_set_paired_prefix_sweep_v1` protocol runs six independently seeded workload sets, seeds 42 and 43, and four treatments (prefix cache ON/OFF × `upstream_dynamic`/`fixed_batch`). That is **48 engine launches and 768 measured requests per model**. The Qwen3-0.6B and Qwen3-8B entrypoints are separate and write under distinct Hydra experiment roots. Each workload/seed pair is identical across all four treatments and both models; the six sets use distinct generated token IDs and lengths.

## Workload design

Each workload has eight hit/miss pairs (16 measured requests). For hit targets, one prompt length is sampled in each of eight strata spanning 512–4096 tokens. All input lengths are greater than 256 and not multiples of the 256-token KV block. Shared block counts are sampled from each prompt's eligible full blocks, rather than being fixed by length stratum; the eight positive counts and ratios are pairwise distinct within each workload. Every matching miss has zero shared prefix. Per-pair requested output lengths are independently sampled from 16, 32, 64, 128, and 256 tokens. Inputs and requested output lengths are shared across treatments/models for each set and seed.

Cache-ON hit targets use one untimed donor request each: the selected full shared prefix plus a disjoint tail token. Donors are generated outside the measured window and excluded from the 16 measured requests. Cache-off mode runs the same priming inputs without retaining cache owners. Each measured run makes one `generate` call; fixed batching is an internal admission policy.

## Grid size

Per model: 48 launches × 16 measured requests = 768 measured request executions; 384 untimed donor requests and 48 warmup adapter requests; **1,200 adapter-level requests** overall. Across both model entrypoints: 96 launches, 1,536 measured request executions, and 2,400 adapter-level requests. These are adapter request counts, not model forward/decode counts.

## Launching

From the repository root on the GPU host, acquire dependencies once. Each entrypoint defaults to `mode=plan`; specify `mode=run` to start inference. `--detach` is supported.

```bash
uv sync --locked --extra inference
uv run --locked --extra inference python experiments/nano_vllm_qwen3_0_6b/run.py mode=run
uv run --locked --extra inference python experiments/nano_vllm_qwen3_8b/run.py mode=run
```

Model locations can be set with `TAEI_QWEN3_0_6B_PATH` and `TAEI_QWEN3_8B_PATH`. The configs bind the matching repository and immutable revision explicitly. Both use TP2, eager execution, max sequence count 8, max batched tokens 16384, context limit 4352, and block size 256. The experiment names and output roots are distinct. Host-specific GPU selection belongs in the user's shell.

The legacy `experiments/nano_vllm_dataset/run.py` entrypoint remains available and shares the expanded workload matrix. Optional `mode=download` performs model acquisition; `mode=plan` is model-free and does not download or infer.

## Artifacts and evaluation status

A model-free generated grid manifest and CSV are stored at:

- `/root/.hermes/cache/scratch/nano-expanded-grid-manifest.json`
- `/root/.hermes/cache/scratch/nano-expanded-grid-runs.csv`

They enumerate the workload builder's run matrix, tokenized request IDs, lengths, per-request outputs, shared-block counts, and planned hit ratios. Planned ratios are not observed metrics.

The run path does not require preflight, source checks, schema validation, QC, calibration, or acceptance gates. Artifacts remain `validation_status=disabled`, `primary_status=not_evaluated`, and `qc.valid=null`. Actual per-request and run-level cache metrics, when collected, derive from initial prompt allocation reuse. Successful generation is not reported as scientific acceptance.

This change was source-only: tests, linters, preflight, configuration validation, QC, model downloads, and GPU inference were not run. No GPU/model execution is claimed.
