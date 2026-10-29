"""GPU waiter uses the observed free-memory sum, not mocked GPU inference."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/wait-gpu-pair.sh"


def test_waiter_retries_invalid_and_exact_threshold_then_launches_once(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state = tmp_path / "state"
    out = tmp_path / "launch"
    nvidia = bin_dir / "nvidia-smi"
    nvidia.write_text("""#!/usr/bin/env bash
n=0; [[ -f "$STATE" ]] && n=$(<"$STATE"); n=$((n+1)); printf %s "$n" > "$STATE"
case "$n" in
1) exit 1;;
2) printf 'GPU-aaaa, N/A\\nGPU-bbbb, 16000\\n';;
3) printf 'GPU-aaaa, 16000\\nGPU-bbbb, 14720\\n';;
4) printf 'GPU-aaaa, 1000\\nGPU-bbbb, 15900\\nGPU-cccc, 15000\\n';;
esac
""")
    uv = bin_dir / "uv"
    uv.write_text("""#!/usr/bin/env bash
printf '%s\\n' "$CUDA_VISIBLE_DEVICES" "$*" >> "$OUT"
exit 17
""")
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/usr/bin/env bash\nexit 0\n")
    for item in (nvidia, uv, sleep):
        item.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STATE": str(state),
        "OUT": str(out),
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 17, result.stderr
    assert state.read_text() == "4"
    lines = out.read_text().splitlines()
    assert lines[0] == "GPU-bbbb,GPU-cccc"
    assert lines[1] == (
        "run --locked --extra inference python "
        "experiments/nano_vllm_qwen3_8b_long_context/run.py "
        "--config-name request_scaling mode=run --detach"
    )
    assert len(lines) == 2
    assert "30720" in result.stdout and "30900" in result.stdout


def test_waiter_rejects_missing_or_duplicate_rows_and_accepts_unbalanced_pair(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text("""#!/usr/bin/env bash
n=0; [[ -f "$STATE" ]] && n=$(<"$STATE"); n=$((n+1)); printf %s "$n" > "$STATE"
case "$n" in
1) printf 'GPU-aaaa, 32000\\n';;
2) printf 'GPU-aaaa, 20000\\nGPU-aaaa, 20000\\n';;
3) printf 'GPU-aaaa, 30720\\nGPU-bbbb, 1\\n';;
esac
""")
    (bin_dir / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "uv").write_text(
        '#!/usr/bin/env bash\nprintf "%s" "$CUDA_VISIBLE_DEVICES" > "$OUT"\n'
    )
    for path in bin_dir.iterdir():
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STATE": str(tmp_path / "state"),
        "OUT": str(tmp_path / "out"),
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "state").read_text() == "3"
    assert (tmp_path / "out").read_text() == "GPU-aaaa,GPU-bbbb"
