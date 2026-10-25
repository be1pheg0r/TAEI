"""Run each GPU attempt in a supervised, disposable process group."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any


def run_isolated(payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """Execute worker payload; timeout kills its complete process group."""
    with tempfile.TemporaryDirectory(prefix="taei-attempt-") as tmp:
        request = Path(tmp) / "request.json"
        response = Path(tmp) / "response.json"
        request.write_text(json.dumps(payload), encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "taei.nano_dataset.isolation",
                str(request),
                str(response),
            ],
            start_new_session=True,
        )
        try:
            _, _ = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as timeout_error:
            _stop_group(process.pid, process, 0.5)
            diagnostic = Path(
                payload.get(
                    "diagnostic_path",
                    Path(payload.get("output_root", ".")) / "supervisor-timeout.json",
                )
            )
            diagnostic = _write_timeout_diagnostic(
                diagnostic,
                timeout_s,
                str(payload.get("run_spec", {}).get("run_id", "unknown")),
                "unknown",
            )
            raise TimeoutError(
                f"attempt subprocess exceeded {timeout_s:g}s; diagnostic={diagnostic}"
            ) from timeout_error
        finally:
            _stop_group(process.pid, process, 0.5)
        result = json.loads(response.read_text(encoding="utf-8"))
        return result


def _write_timeout_diagnostic(
    path: Path, timeout_s: float, run_id: str, attempt_id: str
) -> Path:
    """Persist each timeout independently so repeated failures never overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = path.with_name(f"{path.stem}-{run_id}-{uuid.uuid4()}.json")
    record.write_text(
        json.dumps(
            {
                "status": "timeout",
                "timeout_s": timeout_s,
                "run_id": run_id,
                "attempt_id": attempt_id,
            }
        ),
        encoding="utf-8",
    )
    return record


def _stop_group(pgid: int, process: subprocess.Popen, grace_s: float) -> None:
    """Terminate every process in the session, even if its leader exited."""
    with suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    with suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=1)


def _worker(request: Path, response: Path) -> None:
    from taei.logging import configure_logging

    data = json.loads(request.read_text(encoding="utf-8"))
    configure_logging(config=data.pop("logging_config", {}))
    data.pop("diagnostic_path", None)
    from .orchestrator import run_attempt

    result = run_attempt(**data)
    response.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    _worker(Path(sys.argv[1]), Path(sys.argv[2]))
