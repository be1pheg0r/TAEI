# Detached experiment runs

Foreground is the default and remains attached to the terminal. The opt-in
`--detach` flag starts a new Python session, then the invoking process exits:

```bash
python experiments/smoke/run.py --detach
python experiments/dummy/run.py --detach metrics.score=0.9
python experiments/nano_vllm_dataset/run.py --detach mode=plan
```

Place `--detach` before Hydra overrides. The launcher removes this flag and,
unless an output override is supplied, adds `hydra.run.dir` (or
`hydra.sweep.dir` for multirun) pointing to the launch directory. Other
arguments, environment, Python executable, and working directory are preserved.
The experiment procedure and CUDA initialization do not run in the launcher
parent. `--detach` is not implied by closing a
terminal and is not enabled for normal foreground runs.

## Durable launch files

Each launch gets a collision-resistant directory under the invocation working
directory:

```text
outputs/background-runs/<UTC timestamp>-<id>/
├── launch.json
├── stdout.log
├── stderr.log
├── config.yaml
├── run.log
├── hydra.log
├── results.json
├── logs.zip
├── runs/           # nano-vLLM attempts in mode=run
└── overhead/       # calibration artifacts in mode=overhead
```

`launch.json` records the launch status, child PID, command (not environment
values), working directory, timestamps, and absolute log paths. Launcher files
and Hydra/runtime artifacts share this one invocation directory. An explicit
`hydra.run.dir=/path/to/invocation` is honored and receives all files there;
use a concrete path, not an unresolved `${...}` expression. Existing
`launch.json` records are never overwritten. In multirun, launch files live at
the sweep root and each Hydra job has its own subdirectory with its config,
results and attempts. Plan and help/config inspection modes do not perform
GPU inference; only artifacts actually produced by a mode are present.

Follow output while it runs:

```bash
tail -f outputs/background-runs/<launch-id>/stdout.log
tail -f outputs/background-runs/<launch-id>/stderr.log
```

Inspect metadata and whether the recorded PID currently exists:

```bash
python -m taei.launch status outputs/background-runs/<launch-id>/launch.json
```

The status command reports `pid_alive`; it is a liveness check, not a process
exit code or proof that the PID was not later reused. Finished runs retain
`status: "started"` in their immutable launch record; check `pid_alive` and the
Hydra/runtime artifacts for completion. To request termination of the detached
process group on POSIX, use the recorded PID (the detached session leader is
also its process-group ID):

```bash
kill -INT -- -<PID>
```

SIGINT requests the normal Python interrupt/cleanup path, including cleanup of
supervised GPU attempt groups. `SIGTERM`/`SIGKILL` can bypass that cleanup;
forced termination can leave workers running and incomplete
logs/artifacts. `start_new_session` prevents terminal hangup (`SIGHUP`) from
reaching the detached process group. It does not protect a process from an
explicit group signal, machine shutdown, resource manager, or service manager
that kills the user's processes at logout. For those policies, use an
administrator-approved scheduler/service manager such as `systemd-run` rather
than assuming a terminal-detached process survives logout.

The launcher inherits the caller's environment to preserve Hydra and runtime
semantics, but never writes environment contents to `launch.json`; avoid
putting secrets in command-line arguments because the exact command is recorded.
