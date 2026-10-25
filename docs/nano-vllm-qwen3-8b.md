# Qwen3-8B standalone cache experiment

This dedicated entrypoint uses the shared six-set paired protocol: six workload sets × seeds 42/43 × four cache/scheduler treatments, or **48 engine launches and 768 measured requests**. Each launch contains eight hit/miss pairs, lengths sampled across non-overlapping strata from 512–4096 (never block-aligned), randomly selected eligible shared block counts with distinct positive counts and ratios within each workload, zero reuse targets on misses, and varied requested output lengths (16/32/64/128/256).

The immutable model revision is `b968826d9c46dd6066d109eabc6255188de91218` for `Qwen/Qwen3-8B`. Its model path is `${TAEI_QWEN3_8B_PATH}` or `~/.cache/taei/models/Qwen3-8B`. The configuration explicitly sets model and engine paths, model repo/revision, and leaves inherited upstream config metadata null. It uses TP2/eager, `max_num_seqs=8`, `max_num_batched_tokens=16384`, `max_model_len=4352`, and 256-token KV blocks.

From the repository root on the GPU host:

```bash
uv sync --locked --extra inference
uv run --locked --extra inference python experiments/nano_vllm_qwen3_8b/run.py mode=plan
uv run --locked --extra inference python experiments/nano_vllm_qwen3_8b/run.py mode=run
```

`plan` is the default and does not download or infer. Model acquisition is separately requested with `mode=download`; GPU selection is host-specific. The Qwen3-0.6B experiment is a separate entrypoint and Hydra output root.

The model-free generated matrix is at `/root/.hermes/cache/scratch/nano-expanded-grid-manifest.json` and `/root/.hermes/cache/scratch/nano-expanded-grid-runs.csv`. No validation, QC, calibration, or acceptance status is implied: artifacts remain not evaluated. This was source-only work; no tests, linters, preflight, model downloads, or GPU inference were run.
