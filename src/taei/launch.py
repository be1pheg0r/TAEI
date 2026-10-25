"""Opt-in detached experiment launching and launch status inspection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

DETACH_FLAG = "--detach"


def maybe_launch_detached(script_path: str | Path) -> bool:
    """Launch this script in a new session when ``--detach`` is present.

    The current process only creates the launch record and child process;
    it never calls the experiment procedure or initializes CUDA itself.
    Returns True in the launcher process, which should then exit without
    calling the Hydra entrypoint.
    """
    arguments = list(sys.argv[1:])
    if DETACH_FLAG not in arguments:
        return False
    if arguments.count(DETACH_FLAG) != 1:
        raise ValueError(f"{DETACH_FLAG} may be specified only once")

    cwd = Path.cwd()
    script = Path(script_path)
    if not script.is_absolute():
        script = cwd / script
    if not script.is_file():
        raise FileNotFoundError(f"experiment script not found: {script}")

    child_overrides = [arg for arg in arguments if arg != DETACH_FLAG]
    multirun = (
        "-m" in child_overrides
        or "--multirun" in child_overrides
        or any(
            arg
            in {
                "hydra.mode=MULTIRUN",
                'hydra.mode="MULTIRUN"',
                "hydra.mode='MULTIRUN'",
            }
            for arg in child_overrides
        )
    )
    output_key = "hydra.sweep.dir" if multirun else "hydra.run.dir"
    explicit_output = next(
        (
            arg.split("=", 1)[1]
            for arg in reversed(child_overrides)
            if arg.startswith(f"{output_key}=")
        ),
        None,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    if explicit_output is not None:
        # Honor the user's Hydra output override and co-locate launch files there.
        output_value = explicit_output
        if (
            len(output_value) >= 2
            and output_value[0] == output_value[-1]
            and output_value[0] in {"'", '"'}
        ):
            output_value = output_value[1:-1]
        if not output_value or "${" in output_value:
            raise ValueError(
                f"{DETACH_FLAG} requires a concrete {output_key} override; "
                "omit it to use a unique invocation directory"
            )
        launch_dir = Path(output_value).expanduser()
        if not launch_dir.is_absolute():
            launch_dir = cwd / launch_dir
        launch_dir.mkdir(parents=True, exist_ok=True)
    else:
        launch_root = cwd / "outputs" / "background-runs"
        launch_root.mkdir(parents=True, exist_ok=True)
        launch_dir = launch_root / f"{timestamp}-{uuid.uuid4().hex[:8]}"
        launch_dir.mkdir()
        child_overrides.append(f"{output_key}={launch_dir}")
    child_arguments = [sys.argv[0], *child_overrides]
    stdout_path = launch_dir / "stdout.log"
    stderr_path = launch_dir / "stderr.log"
    metadata_path = launch_dir / "launch.json"
    # Reserve exclusively: a second launch must not overwrite another PID/record.
    with metadata_path.open("x", encoding="utf-8"):
        pass
    metadata: dict[str, object] = {
        "status": "starting",
        "pid": None,
        "command": [sys.executable, *child_arguments],
        "cwd": str(cwd),
        "launch_dir": str(launch_dir),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "created_at": datetime.now(UTC).isoformat(),
    }

    def save_metadata() -> None:
        temporary_path = metadata_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(metadata_path)

    save_metadata()
    child_env = os.environ.copy()
    child_env["TAEI_DETACHED_PROGRESS"] = "1"
    child_env["PYTHONUNBUFFERED"] = "1"
    try:
        with (
            stdout_path.open("ab", buffering=0) as stdout_file,
            stderr_path.open("ab", buffering=0) as stderr_file,
        ):
            child = subprocess.Popen(
                [sys.executable, *child_arguments],
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=cwd,
                env=child_env,
                start_new_session=True,
            )
    except OSError as exc:
        metadata.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        save_metadata()
        raise RuntimeError(
            f"could not start detached experiment; see {metadata_path}"
        ) from exc

    metadata.update({"status": "started", "pid": child.pid})
    save_metadata()
    print(
        f"Detached experiment started: pid={child.pid}\n"
        f"Launch record: {metadata_path}\n"
        f"stdout: {stdout_path}\nstderr: {stderr_path}",
        flush=True,
    )
    return True


def launch_status(metadata_path: str | Path) -> dict[str, object]:
    """Read launch metadata and report whether its recorded PID exists."""
    path = Path(metadata_path).expanduser().resolve()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    pid = metadata.get("pid")
    alive = False
    if isinstance(pid, int) and pid > 0:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
        else:
            alive = True
    return {**metadata, "metadata_path": str(path), "pid_alive": alive}


def main(argv: Sequence[str] | None = None) -> int:
    """Support ``python -m taei.launch status LAUNCH.json``."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2 or arguments[0] != "status":
        print("usage: python -m taei.launch status LAUNCH.json", file=sys.stderr)
        return 2
    try:
        print(json.dumps(launch_status(arguments[1]), ensure_ascii=False, indent=2))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read launch metadata: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
