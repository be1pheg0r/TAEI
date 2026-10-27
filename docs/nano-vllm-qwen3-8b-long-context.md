# Qwen3-8B long-context cache experiment

This is a separate, GPU-only sibling to the existing Qwen3-8B experiment; it does not change the original profile or workload. It uses the immutable `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`. The exact revision's config declares `max_position_embeddings=40960` with `rope_scaling=null`; the model's native pretraining context is 32,768 tokens. The 40,960 metadata value is not treated as a native-context guarantee, and the model card's YaRN extension is not enabled. This experiment deliberately sets nano-vLLM `max_model_len=32768`, so a 50k prompt is outside this profile. At the pinned nano-vLLM revision, `Config.__post_init__` clamps configured `max_model_len` to the loaded model config's `max_position_embeddings`, but the pinned `generate` path does not itself reject prompt-plus-output requests beyond that configured value. The new plan builder therefore enforces `prompt_length + requested_output_tokens <= 32768` before running. Separately, the scheduler enforces `max_num_seqs` per selected step and `max_num_batched_tokens` for prefill, allowing chunked prefill for the first scheduled sequence. These configured bounds are not proof that a particular GPU has enough KV-cache memory.

## Exact workload schedule

Each set/seed has eight matched hit/miss pairs (16 measured requests). Each hit shares a block-aligned prefix with its untimed donor; the paired miss has the same prompt length and output budget but no matching first block. All token sequences are deterministic by workload-set and seed.

| Pair | Prompt tokens | Output budget | Shared blocks | Shared tokens | Target reuse |
|---|---:|---:|---:|---:|---:|
| 01 | 5,121 | 512 | 1 | 256 | 4.999% |
| 02 | 8,705 | 512 | 6 | 1,536 | 17.645% |
| 03 | 12,289 | 1,024 | 15 | 3,840 | 31.247% |
| 04 | 15,873 | 1,024 | 27 | 6,912 | 43.546% |
| 05 | 19,457 | 1,024 | 43 | 11,008 | 56.576% |
| 06 | 23,041 | 2,048 | 62 | 15,872 | 68.886% |
| 07 | 26,625 | 2,048 | 85 | 21,760 | 81.728% |
| 08 | 30,209 | 2,048 | 112 | 28,672 | 94.912% |

Percentages are planned targets, rounded by full 256-token blocks, not observed cache hits. Miss targets are zero; cache-off cannot reuse prefixes. Each donor contains exactly the shared prefix plus one disjoint tail token. Priming processes 89,864 donor prompt tokens per launch outside measurement; cache-on temporarily retains 89,856 prefix tokens until matching allocations, increasing KV demand. The measured window excludes this preparation cost. Reuse fraction is coupled to prompt length and output budget: this is not a Cartesian sweep, and their individual effects cannot be separated from these observations alone.

The largest requested sequence is `30,209 + 2,048 = 32,257` tokens, 511 below configured `max_model_len=32,768`. The original six sets × two seeds × four treatments remain **48 engine launches, 768 measured requests, 13,566,720 logical prompt tokens and 983,040 requested output tokens**. Their run IDs and shuffled order remain unchanged.

An additional 48 launches repeat the exact input token IDs, donor prefixes, seeds and four treatments with **one requested output token per measured request**. Their run IDs start with `P`, their workload IDs end in `_prefill_only`, and they follow the original 48 runs in the same treatment order. These add 768 measured requests, 13,566,720 logical prompt tokens and 768 requested output tokens. The whole plan therefore has **96 launches, 1,536 measured requests, 27,133,440 logical prompt tokens and 983,808 requested output tokens**. Each launch also performs eight untimed donor requests and one adapter warmup request: 768 donors and 96 warmups across the full plan, excluded from measured request counts.

The one-token variant avoids a long output phase, but requests are still batched and long prefill may be chunked; it is not a guaranteed single-request, single-step or pure GPU-only timing experiment. The first output token may be produced on the final prefill step. GPU generation and actual phase traces remain unverified.

The deterministic original run order is shuffled with seed 20261023. `max_num_seqs=8` and `max_num_batched_tokens=16,384`; the pinned scheduler chunks prefill when an individual prompt exceeds the remaining step-token budget. `max_num_batched_tokens` is a per-scheduler-step limit, not a bound on the sum of all queued prompt tokens. Also, upstream dynamic scheduling applies `max_num_seqs` to each scheduled batch, not a hard cap on the full running set; fixed-batch mode uses waves bounded by eight. This means cache/KV demand can exceed an eight-sequence assumption under dynamic scheduling.

The workload is deliberately capped at 32K, not the nominal 40,960 position limit. Even under the 32K cap, real GPU KV-cache capacity and execution feasibility are hardware-dependent and remain unverified until a GPU run. Use `mode=plan` to inspect the saved exact schedule; planning is CPU-only and does not acquire model files or run inference.

## Commands

From the repository root on the selected, compatible GPU host:

```bash
uv sync --locked --extra inference
uv run --locked --extra inference python experiments/nano_vllm_qwen3_8b_long_context/run.py mode=plan
uv run --locked --extra inference python experiments/nano_vllm_qwen3_8b_long_context/run.py mode=run
```

Model acquisition is explicit and outside the measurement window:

```bash
uv run --locked --extra inference python experiments/nano_vllm_qwen3_8b_long_context/run.py mode=download
```

No GPU was launched while creating this profile. The plan does not establish KV capacity, successful model loading, generation, or dataset acceptance; automatic QC/acceptance remains disabled just as in the shared collector.
