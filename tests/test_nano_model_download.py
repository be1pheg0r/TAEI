from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from taei.nano_dataset import model_download

REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@pytest.fixture(autouse=True)
def offline_hub_metadata(monkeypatch):
    """Unit tests never need network merely to resolve the pinned commit."""
    monkeypatch.setattr(
        model_download,
        "HfApi",
        lambda: SimpleNamespace(
            model_info=lambda **kwargs: SimpleNamespace(sha=REVISION)
        ),
    )


def make_model(path: Path, *, weight: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    if weight:
        (path / "model.safetensors").write_bytes(b"tiny fixture")


def test_download_fetches_pinned_snapshot_and_writes_complete_provenance(
    tmp_path, monkeypatch
):
    destination = tmp_path / "model"
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        make_model(destination)
        return str(destination)

    monkeypatch.setattr(model_download, "snapshot_download", download)
    receipt = model_download.ensure_model(destination, "Qwen/Qwen3-0.6B", REVISION)
    marker = json.loads((destination / ".taei-model-download.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["resolved_revision"] == REVISION
    assert len(calls) == 1
    assert calls[0]["revision"] == REVISION
    assert calls[0]["local_dir"] == str(destination)
    assert marker["repo_id"] == "Qwen/Qwen3-0.6B"
    assert marker["requested_revision"] == REVISION
    assert marker["resolved_revision"] == REVISION
    assert marker["status"] == "complete"
    assert marker["files"]["config.json"]["bytes"] > 0
    assert len(marker["files"]["config.json"]["sha256"]) == 64


def test_complete_local_model_reuses_without_network(tmp_path, monkeypatch):
    destination = tmp_path / "local"
    make_model(destination)
    monkeypatch.setattr(
        model_download, "snapshot_download", lambda **_: pytest.fail("downloaded")
    )
    receipt = model_download.ensure_model(destination, "Qwen/Qwen3-0.6B", REVISION)
    assert receipt["status"] == "reused_local"


def test_nonempty_unowned_directory_is_preserved_and_rejected(tmp_path, monkeypatch):
    destination = tmp_path / "custom"
    destination.mkdir()
    (destination / "notes.txt").write_text("user data")
    monkeypatch.setattr(
        model_download, "snapshot_download", lambda **_: pytest.fail("downloaded")
    )
    with pytest.raises(model_download.ModelDownloadError):
        model_download.ensure_model(destination, "Qwen/Qwen3-0.6B", REVISION)
    assert (destination / "notes.txt").read_text() == "user data"


def test_partial_owned_download_failure_is_recorded_and_can_resume(
    tmp_path, monkeypatch
):
    destination = tmp_path / "partial"
    calls = 0

    def interrupted(**_):
        nonlocal calls
        calls += 1
        make_model(destination, weight=False)
        raise OSError("network interrupted")

    monkeypatch.setattr(model_download, "snapshot_download", interrupted)
    with pytest.raises(model_download.ModelDownloadError):
        model_download.ensure_model(destination, "Qwen/Qwen3-0.6B", REVISION)
    failed = json.loads((destination / ".taei-model-download.json").read_text())
    assert failed["status"] == "failed"
    assert "network interrupted" in failed["error"]

    def resumed(**_):
        nonlocal calls
        calls += 1
        (destination / "model.safetensors").write_bytes(b"weights")
        return str(destination)

    monkeypatch.setattr(model_download, "snapshot_download", resumed)
    receipt = model_download.ensure_model(destination, "Qwen/Qwen3-0.6B", REVISION)
    assert receipt["status"] == "complete"
    assert calls == 2


def test_disabled_download_does_not_contact_hub(tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_download, "snapshot_download", lambda **_: pytest.fail("downloaded")
    )
    with pytest.raises(model_download.ModelDownloadError):
        model_download.ensure_model(
            tmp_path / "missing", "Qwen/Qwen3-0.6B", REVISION, enabled=False
        )


def test_missing_weight_and_index_shard_are_not_ready(tmp_path, monkeypatch):
    destination = tmp_path / "broken"

    def download(**_):
        (destination / "config.json").write_text("{}")
        (destination / "tokenizer.json").write_text("{}")
        (destination / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "x": "model-00001-of-00002.safetensors",
                        "y": "model-00002-of-00002.safetensors",
                    }
                }
            )
        )
        (destination / "model-00001-of-00002.safetensors").write_bytes(b"x")
        return str(destination)

    monkeypatch.setattr(model_download, "snapshot_download", download)
    with pytest.raises(model_download.ModelDownloadError):
        model_download.ensure_model(destination, "Qwen/Qwen3-0.6B", REVISION)
    marker = json.loads((destination / ".taei-model-download.json").read_text())
    assert marker["status"] == "failed"
    assert "indexed safetensors shards missing" in marker["error"]


def test_completed_download_rejects_different_pin_without_overwrite(
    tmp_path, monkeypatch
):
    destination = tmp_path / "owned"
    monkeypatch.setattr(
        model_download, "snapshot_download", lambda **_: make_model(destination)
    )
    model_download.ensure_model(destination, "Qwen/Qwen3-0.6B", REVISION)
    before = (destination / model_download.PROVENANCE_FILENAME).read_bytes()
    with pytest.raises(model_download.ModelDownloadError, match="different"):
        model_download.ensure_model(destination, "Qwen/Qwen3-0.6B", "a" * 40)
    assert (destination / model_download.PROVENANCE_FILENAME).read_bytes() == before


def test_completed_download_rejects_modified_config(tmp_path, monkeypatch):
    destination = tmp_path / "owned"
    monkeypatch.setattr(
        model_download, "snapshot_download", lambda **_: make_model(destination)
    )
    model_download.ensure_model(destination, "Qwen/Qwen3-0.6B", REVISION)
    (destination / "config.json").write_text('{"model_type":"changed"}')
    with pytest.raises(model_download.ModelDownloadError, match="modified"):
        model_download.ensure_model(destination, "Qwen/Qwen3-0.6B", REVISION)


def test_tokenizer_config_without_tokenizer_assets_is_incomplete(tmp_path):
    make_model(tmp_path)
    (tmp_path / "tokenizer.json").unlink()
    assert "tokenizer assets missing" in model_download._validate_snapshot(tmp_path)


def load_experiment():
    import importlib.util

    script = (
        Path(__file__).resolve().parents[1] / "experiments/nano_vllm_dataset/run.py"
    )
    spec = importlib.util.spec_from_file_location("nano_dataset_run", script)
    assert spec is not None and spec.loader is not None
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    return experiment


def test_cli_download_mode_bypasses_gpu_readiness(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from taei.nano_dataset import orchestrator

    experiment = load_experiment()

    destination = tmp_path / "model"
    make_model(destination)
    monkeypatch.setattr(
        experiment,
        "ensure_model",
        lambda *args, **kwargs: {"status": "complete", "path": str(destination)},
    )
    monkeypatch.setattr(
        orchestrator,
        "preflight",
        lambda _: {"ready": False, "blocking_reasons": ["torch_cuda_unavailable"]},
    )
    monkeypatch.setattr(
        experiment.HydraConfig,
        "get",
        lambda: type(
            "H", (), {"runtime": type("R", (), {"output_dir": str(tmp_path / "out")})()}
        )(),
    )
    cfg = OmegaConf.load("experiments/nano_vllm_dataset/config.yaml")
    cfg.mode = "download"
    cfg.dataset.model_path = str(destination)
    result = experiment.run(cfg)
    assert result["mode"] == "download"
    assert result["status"] == "complete"


def test_download_cli_persists_failure_receipt(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    experiment = load_experiment()
    cfg = OmegaConf.load("experiments/nano_vllm_dataset/config.yaml")
    cfg.dataset.model_path = str(tmp_path / "model")

    def fail(*args, **kwargs):
        raise model_download.ModelDownloadError("offline")

    monkeypatch.setattr(experiment, "ensure_model", fail)
    with pytest.raises(model_download.ModelDownloadError):
        experiment._download_model(cfg, tmp_path / "run")
    report = json.loads((tmp_path / "run" / "model-download.json").read_text())
    assert report["status"] == "failed"
    assert report["error_type"] == "ModelDownloadError"


def test_inference_path_downloads_model_only_blockers_then_rechecks(
    tmp_path, monkeypatch
):
    from omegaconf import OmegaConf

    experiment = load_experiment()
    cfg = OmegaConf.load("experiments/nano_vllm_dataset/config.yaml")
    cfg.dataset.model_path = str(tmp_path / "model")
    calls = []
    reports = iter(
        [
            {"ready": False, "blocking_reasons": ["model_path_missing"]},
            {"ready": True, "blocking_reasons": []},
        ]
    )
    monkeypatch.setattr(experiment, "preflight", lambda _: next(reports))
    monkeypatch.setattr(
        experiment,
        "ensure_model",
        lambda *args, **kwargs: calls.append(args) or {"status": "complete"},
    )
    report = experiment._prepare_for_inference(cfg, tmp_path / "output", "run")
    assert report["ready"] is True
    assert len(calls) == 1
    assert (tmp_path / "output" / "model-download.json").is_file()


def test_inference_path_does_not_download_when_environment_is_blocked(
    tmp_path, monkeypatch
):
    from omegaconf import OmegaConf

    experiment = load_experiment()
    cfg = OmegaConf.load("experiments/nano_vllm_dataset/config.yaml")
    calls = []
    monkeypatch.setattr(
        experiment,
        "preflight",
        lambda _: {
            "ready": False,
            "blocking_reasons": ["model_path_missing", "torch_cuda_unavailable"],
        },
    )
    monkeypatch.setattr(
        experiment,
        "ensure_model",
        lambda *args, **kwargs: calls.append(args),
    )
    with pytest.raises(RuntimeError, match="torch_cuda_unavailable"):
        experiment._prepare_for_inference(cfg, tmp_path / "output", "run")
    assert calls == []


def test_ready_preflight_still_validates_owned_snapshot_pin(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    experiment = load_experiment()
    cfg = OmegaConf.load("experiments/nano_vllm_dataset/config.yaml")
    cfg.dataset.model_path = str(tmp_path / "model")
    monkeypatch.setattr(
        experiment, "preflight", lambda _: {"ready": True, "blocking_reasons": []}
    )

    def reject_pin(*args, **kwargs):
        raise model_download.ModelDownloadError("different pinned revision")

    monkeypatch.setattr(experiment, "ensure_model", reject_pin)
    with pytest.raises(model_download.ModelDownloadError, match="different"):
        experiment._prepare_for_inference(cfg, tmp_path / "output", "run")
