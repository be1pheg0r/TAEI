"""CLI для планирования и сбора датасета экспериментов nano-vLLM.

Точка входа Hydra проверяет неизменяемые ограничения пилотного движка и пути
модели, затем поддерживает режимы: построить план; сформировать безопасный
отчет preflight по среде, модели и CUDA; выполнить парную диагностику накладных
расходов трассировки после preflight; либо выполнить заданную часть запланированных
реальных запусков. Режим запусков проверяет хэши отчета калибровки, передает
соответствующее области свидетельство каждой неизменяемой попытке, сохраняет
сводки выполнения/QC и сводку датасета. Общая среда отвечает за конфигурацию,
логи, метрики и ClearML; preflight не считается доказательством инференса.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig

from taei.nano_dataset.isolation import run_isolated
from taei.nano_dataset.model_download import ensure_model
from taei.nano_dataset.orchestrator import build_plan, preflight
from taei.runtime import run_experiment

PILOT_ENGINE = {
    "tensor_parallel_size": 4,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 16384,
    "max_model_len": 4096,
    "enforce_eager": False,
    "gpu_memory_utilization": 0.9,
}
ENGINE_PROFILES = {
    "source_tp4": {**PILOT_ENGINE},
    "rtx_a4000_tp2": {
        **PILOT_ENGINE,
        "tensor_parallel_size": 2,
        "enforce_eager": True,
    },
    "rtx_a4000_tp2_qwen3_8b": {
        **PILOT_ENGINE,
        "tensor_parallel_size": 2,
        "enforce_eager": True,
    },
}


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_calibration_report(path: str | None) -> dict | None:
    if not path:
        return None
    report_path = Path(path).expanduser().resolve()
    report_bytes = report_path.read_bytes()
    manifest_path = report_path.with_name("overhead-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(report_bytes).hexdigest() != manifest.get("sha256"):
        raise ValueError("overhead report SHA-256 does not match its manifest")
    report = json.loads(report_bytes)
    for name, record in manifest.get("supporting_files", {}).items():
        raw = (report_path.parent / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record.get("sha256"):
            raise ValueError(f"overhead supporting-file SHA-256 mismatch: {name}")
    return report


def _check_engine_config(cfg: DictConfig) -> dict:
    profile = dict(cfg.hardware_profile)
    profile_id = profile.get("profile_id")
    expected_engine = ENGINE_PROFILES.get(profile_id)
    if expected_engine is None:
        raise ValueError(f"unsupported hardware profile: {profile_id!r}")
    actual = {key: cfg.engine.get(key) for key in expected_engine}
    if actual != expected_engine:
        raise ValueError(
            f"{profile_id} engine invariants differ: "
            f"expected={expected_engine}, actual={actual}"
        )
    expected_identity = {
        "source_tp4": ("nano_vllm_dataset", "outputs/nano_vllm_dataset/runs", False),
        "rtx_a4000_tp2": (
            "nano_vllm_dataset_rtx_a4000_tp2",
            "outputs/nano_vllm_dataset_rtx_a4000_tp2/runs",
            True,
        ),
        "rtx_a4000_tp2_qwen3_8b": (
            "nano_vllm_dataset_rtx_a4000_tp2_qwen3_8b",
            "outputs/nano_vllm_dataset_rtx_a4000_tp2_qwen3_8b/runs",
            True,
        ),
    }[profile_id]
    identity = (
        cfg.experiment.name,
        cfg.dataset.output_root,
        bool(profile.get("hardware_deviation")),
    )
    if identity != expected_identity:
        raise ValueError(
            f"{profile_id} identity invariant differs: "
            f"expected={expected_identity}, actual={identity}"
        )
    model_path = str(cfg.dataset.model_path)
    if cfg.engine.get("model") != model_path or cfg.model.get("path") != model_path:
        raise ValueError("engine/model paths must match selected local model snapshot")
    if cfg.model.get("upstream_config") is not None:
        raise ValueError(
            "stale inherited model upstream_config cannot label the selected model"
        )
    if profile_id == "rtx_a4000_tp2_qwen3_8b":
        expected_model = (
            "Qwen3-8B",
            "Qwen/Qwen3-8B",
            "b968826d9c46dd6066d109eabc6255188de91218",
            "Qwen/Qwen3-8B",
        )
        actual_model = (
            cfg.model.name,
            cfg.model.huggingface_id,
            cfg.model.revision,
            cfg.download.repo_id,
        )
        if actual_model != expected_model:
            raise ValueError(
                "Qwen3-8B profile model identity differs: "
                f"expected={expected_model}, actual={actual_model}"
            )
        if int(cfg.engine.max_model_len) > 40960:
            raise ValueError("Qwen3-8B max_model_len exceeds pinned config limit 40960")
    return {
        key: cfg.engine[key] for key in cfg.engine if key not in {"backend", "model"}
    }


def _download_model(cfg: DictConfig, output_dir: Path) -> dict:
    settings = cfg.download
    try:
        receipt = ensure_model(
            cfg.dataset.model_path,
            str(settings.repo_id),
            str(settings.revision),
            enabled=bool(settings.enabled),
        )
    except Exception as exc:
        _save_json(
            output_dir / "model-download.json",
            {
                "status": "failed",
                "path": str(Path(cfg.dataset.model_path).expanduser()),
                "repo_id": str(settings.repo_id),
                "requested_revision": str(settings.revision),
                "error_type": type(exc).__name__,
            },
        )
        raise
    _save_json(output_dir / "model-download.json", receipt)
    return receipt


def _prepare_for_inference(cfg: DictConfig, output_dir: Path, label: str) -> dict:
    model_path = str(Path(cfg.dataset.model_path).expanduser().resolve())
    report = preflight(model_path)
    _save_json(output_dir / "preflight.json", report)
    if report["ready"]:
        _download_model(cfg, output_dir)
        return report
    model_blockers = {
        "model_path_missing",
        "model_config_missing",
        "model_weights_missing",
        "tokenizer_assets_missing",
    }
    blockers = set(report["blocking_reasons"])
    if blockers and blockers <= model_blockers:
        _download_model(cfg, output_dir)
        report = preflight(model_path)
        _save_json(output_dir / "preflight.json", report)
    if not report["ready"]:
        raise RuntimeError(
            f"{label} preflight blocked: {report['blocking_reasons']}; "
            f"report={output_dir / 'preflight.json'}"
        )
    return report


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> dict:
    return run_experiment(cfg, run)


def run(cfg: DictConfig) -> dict:
    mode = str(cfg.mode)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    model_path = str(Path(cfg.dataset.model_path).expanduser().resolve())
    if mode == "plan":
        plan = build_plan(int(cfg.dataset.order_seed))
        plan["configuration"] = {
            "engine": _check_engine_config(cfg),
            "model_path": model_path,
            "experiment_profile": dict(cfg.hardware_profile),
        }
        _save_json(output_dir / "plan.json", plan)
        return {
            "mode": mode,
            "run_count": plan["run_count"],
            "request_count": plan["request_count"],
            "logical_prompt_tokens": plan["logical_prompt_tokens"],
            "requested_output_tokens": plan["requested_output_tokens"],
            "plan_path": str(output_dir / "plan.json"),
        }
    if mode == "download":
        return {"mode": mode, **_download_model(cfg, output_dir)}
    if mode == "preflight":
        report = preflight(model_path)
        _save_json(output_dir / "preflight.json", report)
        if not report["ready"]:
            detail = f"preflight blocked: {report['blocking_reasons']}"
            detail += f"; report={output_dir / 'preflight.json'}"
            raise RuntimeError(detail)
        return {
            "mode": mode,
            "ready": True,
            "preflight_path": str(output_dir / "preflight.json"),
        }
    if mode == "overhead":
        engine_config = _check_engine_config(cfg)
        report = _prepare_for_inference(cfg, output_dir, "overhead")
        from taei.nano_dataset.overhead import measure_overhead

        diagnostic = measure_overhead(
            model_path,
            engine_config,
            Path(cfg.dataset.overhead_output_root) / str(uuid.uuid4()),
            workload_name=str(cfg.dataset.overhead_workload),
            seed=int(cfg.dataset.overhead_seed),
            pair_count=int(cfg.dataset.overhead_pair_count),
        )
        return {
            "mode": mode,
            "diagnostic_id": diagnostic["diagnostic_id"],
            "status": diagnostic["status"],
            "calibration_applicable": diagnostic["calibration_applicable"],
            "report_path": diagnostic["path"],
        }
    if mode != "run":
        raise ValueError(
            f"unknown mode {mode!r}; expected plan, preflight, download, "
            "overhead, or run"
        )
    engine_config = _check_engine_config(cfg)
    report = _prepare_for_inference(cfg, output_dir, "run")
    calibration = _load_calibration_report(cfg.dataset.calibration_report)
    plan = build_plan(int(cfg.dataset.order_seed))
    limit = int(cfg.dataset.run_limit)
    if limit < 0 or limit > 24:
        raise ValueError("dataset.run_limit must be between 0 and 24")
    selected = plan["matrix"][: limit or 24]
    summaries = []
    for item in selected:
        logger.info(
            "dataset run={} order={} condition={}",
            item["run_id"],
            item["run_order"],
            item["condition_id"],
        )
        attempt_summary = run_isolated(
            {
                "run_spec": item,
                "model_path": model_path,
                "engine_config": engine_config,
                "output_root": cfg.dataset.output_root,
                "experiment_profile": dict(cfg.hardware_profile),
                "telemetry_interval_s": float(cfg.dataset.telemetry_interval_s),
                "calibration_report": calibration,
                "calibration_report_path": str(cfg.dataset.calibration_report)
                if cfg.dataset.calibration_report
                else None,
                "logging_config": {
                    "logging": dict(cfg.get("logging", {})),
                    "clearml": dict(cfg.get("clearml", {})),
                },
            },
            timeout_s=float(cfg.dataset.attempt_timeout_s),
        )
        summaries.append(attempt_summary)
        if attempt_summary.get("status") != "success":
            raise RuntimeError(
                f"attempt failed; artifact={attempt_summary.get('path', 'unknown')}"
            )
    identities = {
        key: sorted({item[key] for item in summaries if item.get(key) is not None})
        for key in ("hardware_id", "environment_id", "model_id")
    }
    result = {
        "mode": mode,
        "attempt_count": len(summaries),
        "runs": summaries,
        "accepted_primary_count": sum(
            item["primary_status"] == "accepted" for item in summaries
        ),
        "homogeneity": {
            "identity_values": identities,
            "all_stable_ids_constant": bool(summaries)
            and all(len(values) == 1 for values in identities.values()),
        },
        "execution_evidence": {
            "backend_constructed_count": sum(
                bool(item.get("backend_constructed")) for item in summaries
            ),
            "real_generate_completed_count": sum(
                bool(item.get("real_generate_completed")) for item in summaries
            ),
            "gpu_validation_claimed": False,
        },
    }
    _save_json(output_dir / "dataset-summary.json", result)
    return result


if __name__ == "__main__":
    main()
