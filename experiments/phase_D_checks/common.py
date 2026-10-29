"""Чтение попыток локальных фаз Qwen3-0.6B для разборов фазы D (без GPU)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "qwen-3-06b-local"
OUT = Path(__file__).resolve().parent / "out"
BLOCK = 256
KV_BLOCKS = 186


def jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def attempts(phase: str, pattern: str) -> Iterator[tuple[dict, Path]]:
    """Successful attempts of ``phase`` whose run id matches ``pattern``."""
    for f in sorted((DATA / f"phase_{phase}" / "runs").glob(f"{pattern}/*/runs.json")):
        run = json.loads(f.read_text())
        run = run[0] if isinstance(run, list) else run
        if run["status"] == "success":
            yield run, f.parent


def decode_steps(attempt: Path, skip: int = 0) -> list[dict]:
    """Decode steps as the loader sees them (``nano_time.data``): host step time,
    B and Σℓ before the step; the first ``skip`` decode steps are dropped."""
    rows = []
    for s in jsonl(attempt / "steps.jsonl"):
        if s["phase"] != "decode":
            continue
        rows.append(
            {
                "T": (s["step_host_end_ns"] - s["step_host_start_ns"]) * 1e-9,
                "B": s["actual_batch_size"],
                "sum_l": float(sum(s["contexts_before"].values())),
                "contexts": s["contexts_before"],
            }
        )
    return rows[skip:]


def table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    """Markdown table; ``cols`` — (key, header, format)."""
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "---|" * len(cols)
    body = [
        "| "
        + " | ".join(format(r[k], f) if r[k] is not None else "—" for k, _, f in cols)
        + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


def write(name: str, text: str) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(text, encoding="utf-8")
    print(text)
