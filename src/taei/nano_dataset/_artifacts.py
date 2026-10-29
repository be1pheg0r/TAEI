"""Shared byte-stable serialization helpers for dataset artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    """Encode JSON identically to the dataset identity hashing contract."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def write_json(path: Path, value: Any) -> None:
    """Write the pretty-printed JSON artifact format used by the collector."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact, one-object-per-line JSONL without changing row order."""
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                )
                + "\n"
            )
