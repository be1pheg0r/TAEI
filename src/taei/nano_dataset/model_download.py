"""Download and validate a pinned Hugging Face nano-vLLM model snapshot."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock
from huggingface_hub import HfApi, snapshot_download

PROVENANCE_FILENAME = ".taei-model-download.json"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SMALL_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer*.json",
    "tokenizer.model",
    "vocab*",
    "merges*",
    "added_tokens*.json",
    "special_tokens_map.json",
    "spiece.model",
    "*.tiktoken",
    "*.index.json",
)
_DOWNLOAD_PATTERNS = (*_SMALL_PATTERNS, "*.safetensors")


class ModelDownloadError(RuntimeError):
    """The requested local model snapshot is unavailable or incomplete."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_error(exc: BaseException) -> str:
    # Avoid persisting credential-bearing URLs and keep receipts bounded.
    text = re.sub(r"https?://\S+", "[URL omitted]", str(exc))
    text = re.sub(r"(?i)(token|authorization|password)=\S+", r"\1=[omitted]", text)
    return f"{type(exc).__name__}: {text[:400]}"


def _marker_path(destination: Path) -> Path:
    return destination / PROVENANCE_FILENAME


def _read_marker(destination: Path) -> dict[str, Any] | None:
    if _marker_path(destination).is_symlink():
        raise ModelDownloadError("refusing a symlinked model provenance marker")
    try:
        value = json.loads(_marker_path(destination).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _validate_snapshot(path: Path) -> list[str]:
    if not path.is_dir():
        return ["model directory missing"]
    reasons = []
    config_path = path / "config.json"
    if not config_path.is_file():
        reasons.append("config.json missing")
    else:
        try:
            if not isinstance(
                json.loads(config_path.read_text(encoding="utf-8")), dict
            ):
                reasons.append("config.json invalid")
        except (OSError, ValueError):
            reasons.append("config.json invalid")
    tokenizer_files = [
        item
        for pattern in (
            "tokenizer.json",
            "tokenizer.model",
            "vocab*",
            "merges*",
            "spiece.model",
            "*.tiktoken",
        )
        for item in path.glob(pattern)
        if item.is_file() and item.stat().st_size > 0
    ]
    if not tokenizer_files:
        reasons.append("tokenizer assets missing")
    index_path = path / "model.safetensors.index.json"
    weights = [item for item in path.glob("*.safetensors") if item.is_file()]
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("empty weight_map")
            missing = sorted(set(weight_map.values()) - {item.name for item in weights})
            if missing:
                reasons.append("indexed safetensors shards missing")
        except (OSError, ValueError, KeyError, TypeError):
            reasons.append("safetensors index invalid")
    elif not weights:
        reasons.append("safetensors weights missing")
    if weights and any(item.stat().st_size == 0 for item in weights):
        reasons.append("empty safetensors file")
    return reasons


def _asset_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for pattern in _SMALL_PATTERNS:
        for item in path.glob(pattern):
            if item.is_file() and item.name != PROVENANCE_FILENAME:
                records[item.name] = {
                    "sha256": _sha256(item),
                    "bytes": item.stat().st_size,
                }
    return dict(sorted(records.items()))


def _weight_records(path: Path) -> list[dict[str, Any]]:
    # Large weight files are not reread for hashing; Hub transport verification applies.
    return [
        {"name": item.name, "bytes": item.stat().st_size, "sha256": None}
        for item in sorted(path.glob("*.safetensors"))
        if item.is_file()
    ]


def _write_marker(path: Path, marker: dict[str, Any]) -> None:
    marker_path = _marker_path(path)
    temporary = marker_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker_path)


def _check_owned_snapshot(destination: Path, repo_id: str, revision: str) -> None:
    marker = _read_marker(destination)
    if marker is None:
        if _marker_path(destination).exists():
            raise ModelDownloadError("model provenance marker is invalid")
        return
    if marker.get("repo_id") != repo_id or marker.get("requested_revision") != revision:
        raise ModelDownloadError("model directory is owned by a different download pin")
    if marker.get("status") == "complete" and (
        marker.get("files") != _asset_records(destination)
        or marker.get("weight_files") != _weight_records(destination)
    ):
        raise ModelDownloadError("completed model assets were modified; refusing reuse")


def ensure_model(
    model_path: str | Path,
    repo_id: str,
    revision: str,
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """Ensure a complete model, downloading only into empty or owned partial paths."""
    destination = Path(model_path).expanduser().resolve()
    if not enabled:
        _check_owned_snapshot(destination, repo_id, revision)
        if not _validate_snapshot(destination):
            return {"status": "reused_local", "path": str(destination), "repo_id": None}
        raise ModelDownloadError("download disabled and local model is incomplete")

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(destination.parent / f".{destination.name}.taei-download.lock"))
    with lock:
        _check_owned_snapshot(destination, repo_id, revision)
        marker = _read_marker(destination) if destination.exists() else None
        exists = destination.exists()
        nonempty = exists and any(destination.iterdir())
        matching_owned = bool(
            marker
            and marker.get("repo_id") == repo_id
            and marker.get("requested_revision") == revision
            and marker.get("status") in {"downloading", "failed"}
        )
        if nonempty and not marker:
            if not _validate_snapshot(destination):
                return {
                    "status": "reused_local",
                    "path": str(destination),
                    "repo_id": None,
                }
            raise ModelDownloadError(
                "refusing non-empty unowned incomplete model directory"
            )
        if nonempty and marker and not matching_owned:
            if marker.get("status") == "complete" and not _validate_snapshot(
                destination
            ):
                return {
                    "status": "reused_local",
                    "path": str(destination),
                    "repo_id": marker.get("repo_id"),
                    "resolved_revision": marker.get("resolved_revision"),
                }
            raise ModelDownloadError("model directory is owned by a different download")
        if (
            marker
            and marker.get("status") == "complete"
            and not _validate_snapshot(destination)
        ):
            return {
                "status": "complete",
                "path": str(destination),
                "repo_id": marker.get("repo_id"),
                "resolved_revision": marker.get("resolved_revision"),
            }
        if nonempty and not matching_owned:
            raise ModelDownloadError("refusing non-empty unowned model directory")

        destination.mkdir(parents=True, exist_ok=True)
        pending: dict[str, Any] = {
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_revision": marker.get("resolved_revision")
            if matching_owned
            else None,
            "status": "downloading",
            "files": {},
            "weight_files": [],
            "weight_hash_policy": (
                "sizes only; large weights are not reread for hashing"
            ),
            "updated_at": _now(),
        }
        if (
            matching_owned
            and isinstance(pending["resolved_revision"], str)
            and not _SHA40.fullmatch(pending["resolved_revision"])
        ):
            pending["resolved_revision"] = None
        _write_marker(destination, pending)
        try:
            resolved = pending["resolved_revision"]
            if not resolved:
                info = HfApi().model_info(repo_id=repo_id, revision=revision)
                resolved = info.sha
            if not isinstance(resolved, str) or not _SHA40.fullmatch(resolved):
                raise ModelDownloadError(
                    "Hub did not resolve revision to a full commit hash"
                )
            pending["resolved_revision"] = resolved
            _write_marker(destination, pending)
            snapshot_download(
                repo_id=repo_id,
                revision=resolved,
                local_dir=str(destination),
                allow_patterns=list(_DOWNLOAD_PATTERNS),
            )
            problems = _validate_snapshot(destination)
            if problems:
                raise ModelDownloadError(
                    "downloaded snapshot incomplete: " + ", ".join(problems)
                )
            pending.update(
                status="complete",
                files=_asset_records(destination),
                weight_files=_weight_records(destination),
                updated_at=_now(),
            )
            _write_marker(destination, pending)
            return {
                "status": "complete",
                "path": str(destination),
                "repo_id": repo_id,
                "requested_revision": revision,
                "resolved_revision": resolved,
                "provenance_path": str(_marker_path(destination)),
            }
        except Exception as exc:
            pending.update(status="failed", error=_safe_error(exc), updated_at=_now())
            with contextlib.suppress(OSError):
                _write_marker(destination, pending)
            if isinstance(exc, ModelDownloadError):
                raise
            raise ModelDownloadError(_safe_error(exc)) from exc
