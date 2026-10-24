from __future__ import annotations

import json
import os
import pty
import subprocess
import sys
import time
from pathlib import Path

import pytest

from taei.launch import maybe_launch_detached


def test_foreground_without_flag_does_not_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run.py", "metrics.value=9"])
    assert maybe_launch_detached("run.py") is False


def test_detached_launch_rejects_missing_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_script = tmp_path / "missing.py"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(missing_script), "--detach"])

    with pytest.raises(FileNotFoundError, match="experiment script not found"):
        maybe_launch_detached(missing_script)


def test_detached_launch_strips_only_flag_and_records_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path('worker-result.txt').write_text(repr(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(worker), "--detach", "x=1", "--help"])

    assert maybe_launch_detached(worker) is True
    metadata_files = list(
        (tmp_path / "outputs" / "background-runs").glob("*/launch.json")
    )
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["pid"] > 0
    assert metadata["status"] == "started"
    assert Path(metadata["stdout"]).is_file()
    assert Path(metadata["stderr"]).is_file()
    result = tmp_path / "worker-result.txt"
    for _ in range(100):
        if result.exists():
            break
        import time

        time.sleep(0.02)
    child_args = json.loads(result.read_text(encoding="utf-8").replace("'", '"'))
    assert child_args[:2] == ["x=1", "--help"]
    assert len(child_args) == 3
    assert child_args[2].startswith("hydra.run.dir=")
    assert Path(child_args[2].split("=", 1)[1]) == metadata_files[0].parent


def test_launch_directory_is_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text("pass\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(worker), "--detach"])
    assert maybe_launch_detached(worker) is True
    first = next((tmp_path / "outputs" / "background-runs").glob("*/launch.json"))
    monkeypatch.setattr(sys, "argv", [str(worker), "--detach"])
    assert maybe_launch_detached(worker) is True
    launches = list((tmp_path / "outputs" / "background-runs").glob("*/launch.json"))
    assert len({path.resolve() for path in launches}) == 2
    assert first in launches


def test_detached_launch_does_not_overwrite_existing_launch(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text("pass\n", encoding="utf-8")
    output = tmp_path / "invocation"
    output.mkdir()
    record = output / "launch.json"
    record.write_text('{"original": true}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", [str(worker), "--detach", f"hydra.run.dir={output}"]
    )
    with pytest.raises(FileExistsError):
        maybe_launch_detached(worker)
    assert json.loads(record.read_text()) == {"original": True}


def test_detached_launch_rejects_unresolved_output_directory(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text("pass\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", [str(worker), "--detach", "hydra.run.dir=outputs/${now:%Y}"]
    )
    with pytest.raises(ValueError, match="concrete"):
        maybe_launch_detached(worker)
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("experiment", ["smoke", "nano_vllm_dataset"])
@pytest.mark.parametrize("explicit_output", [False, True])
def test_actual_detached_experiment_co_locates_artifacts(
    tmp_path, experiment, explicit_output
):
    script = Path(__file__).resolve().parents[1] / "experiments" / experiment / "run.py"
    arguments = [sys.executable, str(script), "--detach"]
    if experiment == "nano_vllm_dataset":
        arguments.append("mode=plan")
    if explicit_output:
        arguments.append(f"hydra.run.dir={tmp_path / 'invocation'}")
    launch = subprocess.run(
        arguments, cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert launch.returncode == 0, launch.stderr
    records = list(tmp_path.rglob("launch.json"))
    assert len(records) == 1
    output = records[0].parent
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not (output / "logs.zip").exists():
        time.sleep(0.05)
    assert (output / "logs.zip").is_file(), (output / "stderr.log").read_text()
    for name in ("config.yaml", "run.log", "results.json", "stdout.log", "stderr.log"):
        assert (output / name).is_file(), name
    if explicit_output:
        assert output == tmp_path / "invocation"
    if experiment == "nano_vllm_dataset":
        from omegaconf import OmegaConf

        config = OmegaConf.load(output / "config.yaml")
        assert Path(config.dataset.output_root) == output / "runs"
        assert Path(config.dataset.overhead_output_root) == output / "overhead"
        assert (output / "plan.json").is_file()


@pytest.mark.skipif(os.name != "posix", reason="detached sessions are POSIX-specific")
def test_detached_child_and_descendant_survive_controlling_pty_close(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "worker.py"
    descendant_code = (
        "import time; from pathlib import Path; time.sleep(.25); "
        "Path('descendant-complete').write_text('yes')"
    )
    worker.write_text(
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "Path('worker-started').write_text('yes')\n"
        f"child = subprocess.Popen([sys.executable, '-c', {descendant_code!r}])\n"
        "Path('descendant-pid').write_text(str(child.pid))\n"
        "child.wait()\n"
        "Path('worker-complete').write_text('yes')\n",
        encoding="utf-8",
    )

    child_pid, master_fd = pty.fork()
    if child_pid == 0:
        os.chdir(tmp_path)
        sys.argv = [str(worker), "--detach"]
        maybe_launch_detached(worker)
        os._exit(0)

    launch_dir = tmp_path / "outputs" / "background-runs"
    deadline = time.monotonic() + 5
    metadata_path = None
    while time.monotonic() < deadline and not (tmp_path / "worker-started").exists():
        matches = list(launch_dir.glob("*/launch.json")) if launch_dir.exists() else []
        if matches:
            metadata_path = matches[0]
        time.sleep(0.02)
    assert metadata_path is not None
    assert (tmp_path / "worker-started").exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    os.close(master_fd)
    os.waitpid(child_pid, 0)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not (tmp_path / "worker-complete").exists():
        time.sleep(0.02)

    assert (tmp_path / "worker-started").exists()
    assert (tmp_path / "descendant-complete").exists()
    assert (tmp_path / "worker-complete").exists()
    assert int((tmp_path / "descendant-pid").read_text(encoding="utf-8")) > 0
    assert metadata["pid"] > 0
