# nano-vLLM pilot dataset

This experiment follows Outline document `b0d87b1f-2006-4d92-8fbf-2efc72375b3c`, revision 16 (updated 2026-10-23). The implementation mapping version is **1.0.0**; document provenance and mapping version are copied to each run manifest. Do not infer GPU results from CPU preflight or plan.

## CPU-only plan, download and preflight

Install with `uv sync`. Plan does not import nano-vLLM or PyTorch. Preflight is read-only; it never downloads. The Hugging Face downloader is a base dependency, so CPU-only preparation does not require the `inference` extra.

```bash
uv run python experiments/nano_vllm_dataset/run.py --cfg job
uv run python experiments/nano_vllm_dataset/run.py mode=plan
uv run python experiments/nano_vllm_dataset/run.py mode=download
```

The default model is pinned to `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`. Its default destination is `~/.cache/taei/models/Qwen3-0.6B`; set `TAEI_QWEN3_0_6B_PATH` or `dataset.model_path=...` to select another path. `download.enabled`, `download.repo_id`, and `download.revision` are explicit controls. The downloader resolves the requested revision to an immutable Hub commit and uses that commit for the snapshot.

`mode=download` works without CUDA and writes `model-download.json` in the Hydra output directory and `.taei-model-download.json` beside the model assets. The marker records repository/revision/status and SHA-256 plus size for small config/tokenizer/index assets. Large weight files are recorded by name and size only, without rereading multi-GB files for hashing. A marker is `complete` only after config, tokenizer, safetensors and indexed-shard validation. Interrupted downloads can resume only for the same repository and requested revision. Non-empty incomplete unowned directories are preserved and rejected; complete local custom models are reused without claiming Hub provenance. A different pin cannot take over a downloader-owned directory.

`mode=run` and `mode=overhead` download only when read-only preflight reports model-only blockers. If preflight also finds a non-model blocker (such as unavailable CUDA/inference runtime), they fail before transferring model weights. After downloading, they run preflight again and proceed only if ready. `mode=plan` and `mode=preflight` never download.

Plan does not import nano-vLLM or PyTorch. Preflight does not load the model or construct a CUDA context; it imports PyTorch only for its lightweight `torch.cuda.is_available()` check.

```bash
uv run python experiments/nano_vllm_dataset/run.py --cfg job
uv run python experiments/nano_vllm_dataset/run.py mode=plan
TAEI_QWEN3_0_6B_PATH=/local/models/Qwen3-0.6B \
  uv run python experiments/nano_vllm_dataset/run.py mode=preflight
```

Preflight reads local model metadata and host hardware through read-only collectors. It exits nonzero and preserves `preflight.json` if the local model path or visible GPU capability is absent. The effective model path defaults to `~/.cache/taei/models/Qwen3-0.6B`; override with `TAEI_QWEN3_0_6B_PATH` or `dataset.model_path=...`. Actual architecture, dimensions, hashes, and weight dtype come from the selected local snapshot. Inherited Qwen3-8B labels are nulled, not presented as facts.

Plan writes `plan.json` in Hydra's output directory: 24 fixed run IDs in seeded randomized execution order, 12 conditions, 2 repeats (seeds 42/43), 32 requests/run, 768 planned executions, 835,584 logical prompt tokens, and 245,760 requested output tokens. Per-run prompt/output totals: S=4,096/4,096; L=65,536/16,384; M=34,816/10,240. Token-ID workloads are deterministic per workload seed and reused across cache/scheduler treatments. M contains sixteen short and sixteen long prompts; all long prompts share exactly their first 1,024 IDs and have unique suffixes. S is input/output 128/128, L 2,048/512, M mixes 128-token short outputs with 512-token long outputs.

Pair IDs group the four treatments for each workload and seed. Split group IDs keep all treatments and both seeds of a workload family together, preventing repeat/family leakage into downstream train/test splits.

## Exact GPU run

Run on the intended local GPU host only after preflight succeeds, with the model local and pinned inference extra installed:

```bash
uv sync --extra inference
export TAEI_QWEN3_0_6B_PATH=/local/models/Qwen3-0.6B
uv run python experiments/nano_vllm_dataset/run.py \
  mode=run dataset.model_path="$TAEI_QWEN3_0_6B_PATH" dataset.run_limit=24
```

`dataset.run_limit=1` is a setup check, not the complete pilot. Do not use Hydra multirun to represent the matrix: orchestration owns execution order and identity. Every measured row makes exactly one `engine.generate` call with all 32 requests. Fixed-batch admission waves happen within that call. Each attempt writes a unique UUID directory; failures are retained and retries never overwrite them.

The effective pilot config is asserted: TP=1, max_num_seqs=8, max_num_batched_tokens=16,384, max_model_len=4,096, CUDA graphs enabled (`enforce_eager=false`), gpu_memory_utilization=0.9. `max_num_seqs=8` is an upstream **per-schedule selected batch** cap in dynamic mode; the upstream running set can exceed eight across steps. Fixed mode admits waves <=8 without replacing completed members until the wave drains. Neither is a global active-request cap.

## Paired trace-overhead diagnostic

Before primary timing review, run a separate detailed-vs-minimal diagnostic on the same local GPU/model/config:

```bash
uv run python experiments/nano_vllm_dataset/run.py mode=overhead
```

This writes a standalone report and supporting files to a unique directory under `dataset.overhead_output_root`; the exact report path is saved in the command's `results.json`. It is not one of the 24 pilot runs. Default diagnostic scope is S, seed 42, prefix cache on, upstream dynamic scheduler. The report records actual token input/output identity, profile/config/source identity, paired wall-time deltas/ratios, and files/hashes. It does not invent a pass threshold. When re-running the pilot, pass the exact report path:

```bash
uv run python experiments/nano_vllm_dataset/run.py \
  mode=run dataset.calibration_report=/absolute/path/to/overhead.json dataset.run_limit=24
```

Only the matching S/seed-42/cache-on/upstream-dynamic treatment is within this diagnostic's scope, and even that gate stays `review_required` until a human reviews its magnitude/variability. It cannot calibrate the other workload, repeat, cache, or scheduler conditions. The report alone never promotes a row to `accepted`.

## Timing and primary gate

The measured interval starts after `backend.synchronize()` and ends after matching synchronization around one real `generate` call. It includes upstream scheduling, model execution, sampling, and tokenizer decoding before return. Model load, disjoint warm-up/cache reset, workload tokenization, and buffered trace flush are excluded. Host timestamps are observations, not GPU execution instants. Per-token GPU timestamps and per-step GPU duration remain unsupported absent genuine instrumentation. Host monotonic timestamps share a process clock domain; source identifiers remain recorded, and recorder-domain/process/receipt-time fields make joins explicit.

Telemetry includes explicit before/after snapshots outside the timed window plus periodic samples. The `inference` extra installs `nvidia-ml-py` for read-only NVML access. Device support and permissions remain conditional; missing values carry field status/reason, never zero. Periodic samples are not per-step estimates. No counter reset, clock/power change, or remote execution occurs.

**Primary analysis is gated pending a paired minimal-vs-detailed trace overhead diagnostic.** Successful inference is not primary-accepted until that diagnostic is reviewed, required traces/counts/joins pass QC, and calibration is explicitly recorded. Partial or empty traces are incomplete/rejected. No forecast is generated.

The current collector preserves matching calibration evidence as `review_required` and does not implement a review-approval/promote command. Therefore raw inference collection can succeed while primary-analysis acceptance remains pending; the raw data must not be relabeled as accepted automatically.

## Dataset/QC

Each immutable attempt contains `manifest.json`, `hardware.json`, `environment.json`, `model.json`, `workload.json`, one-object `runs.json`, and JSONL `requests`, `steps`, `step_requests`, `cache_events`, `scheduler_events`, `token_events`, `telemetry`. SHA-256 and byte lengths cover every persisted file except the manifest. QC verifies hashes, strict JSONL, required primary trace files, and request/step joins. `runs.json` is an object, not an array.

`src/taei/nano_dataset/schema.py` supplies the versioned JSON Schema and field dictionary. Dictionary entries identify role (input/label/diagnostic), source/hook, availability, and null reasons. Each run directory persists `schema.json`, `field_dictionary.json`, and `schema_validation.json`; nullable missing capabilities carry field status/reason.

### Versioned source-to-artifact mapping (1.0.0)

| Revision-16 requirement | Dataset fields / artifacts | Capture and acceptance rule |
|---|---|---|
| 12 conditions × 2 seeds, randomized order, stable run IDs | `runs.condition_id`, `pair_id`, `repeat_id`, `run_order`, `attempt_id`; `workload.json`, `requests.jsonl` | Matrix builder; immutable per-attempt directories; exact totals checked in tests |
| S/L/M token-ID inputs, sampling, shared-prefix boundary | `workload.requests[].input_token_ids`, `requested_output_tokens`, `temperature`, `ignore_eos`, `prefix_family_id`, `prefix_group_id`; workload summary | Deterministic local token-ID builder; mode treatments reuse exact workload+seed IDs; pinned upstream chain semantics noted in the plan |
| One real generate window, throughput/latency labels | `runs.metrics`, request start/finish fields, `timing_policy` | Host monotonic brackets around one real `generate`; output includes upstream decode time; null for unavailable timings |
| Per-step scheduling, request links, cache, token, telemetry evidence | `steps.jsonl`, `step_requests.jsonl`, `scheduler_events.jsonl`, `cache_events.jsonl`, `token_events.jsonl`, `telemetry.jsonl` | Pinned hooks; actual IDs/order; missing required table or invalid join rejects primary completeness |
| Hardware, software, model and reproducibility context | `hardware.json`, `environment.json`, `model.json`, stable corresponding IDs | Read-only named collectors and local model hashes; field-specific status/reason; secret allowlist |
| Strict dataset integrity, joins and calibration | `manifest.files`, `schema_validation.json`, `runs.timing_policy.primary_analysis_gate` | SHA-256 exact-byte hashes, JSON Schema validation, QC join checks; overhead evidence applies only to matching scope and remains review-required |

Manifest records the source document ID/revision/update and mapping version. The source document body was not archived here, so no content hash is claimed.

## Local verification limits

Hydra composition, deterministic workloads/totals, CPU preflight, metadata collectors, schema API, hashes/joins, and pinned-source CPU scheduler/cache tests are CPU checks. This environment has not performed GPU inference. Full 24-run GPU acceptance, model/GPU/software homogeneity, and trace-overhead calibration remain unverified until executed and reviewed on the user's GPU host.
