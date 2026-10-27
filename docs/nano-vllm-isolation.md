# nano-vLLM isolated attempts

Each measured attempt runs in a fresh session/process group with a hard per-attempt timeout (`dataset.attempt_timeout_s`). A non-success attempt stops the series immediately; its immutable attempt directory and `runs.json` remain the diagnostic record. Timeouts produce uniquely named supervisor diagnostics. On timeout or parent interruption, the supervisor sends TERM to the entire process group, waits briefly, then KILLs the group; this handles children that ignore TERM and descendants even if the worker leader exits.

The child inherits the composed `logging` settings and calls the shared `taei.logging.configure_logging` API. The outer Hydra runtime remains responsible for lifecycle, output capture, and archives.

## Eager-mode hardware profile

The `rtx_a4000_tp2` derivative enables `engine.enforce_eager: true`, disabling CUDA-graph capture/replay. This is a conservative compatibility/performance tradeoff: it can increase launch overhead and reduce throughput, so results from it are not directly equivalent to the source TP4 profile. It is a mitigation, not proof of an NCCL root cause. Overhead calibration is optional and does not gate collection. If performed, it must match the effective eager configuration to be applicable.

Run `uv run --locked --extra inference python experiments/nano_vllm_dataset/run.py profile@_global_=rtx_a4000_tp2 mode=plan --cfg job` to inspect the composed profile without inference.
