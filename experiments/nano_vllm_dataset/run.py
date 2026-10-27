"""Hydra entrypoint for paired nano-vLLM cache experiments, without validation gates."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from taei import logging as project_logging
from taei.launch import maybe_launch_detached
from taei.nano_dataset.isolation import run_isolated
from taei.nano_dataset.model_download import ensure_model
from taei.nano_dataset.orchestrator import build_plan
from taei.runtime import run_experiment


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _engine_config(cfg: DictConfig) -> dict:
    """Forward the chosen engine settings without fixed-profile invariants."""
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


def _prepare_model(cfg: DictConfig, output_dir: Path) -> None:
    """Acquire a missing model when enabled; leave existing local assets alone."""
    if cfg.download.enabled and not Path(cfg.dataset.model_path).expanduser().exists():
        _download_model(cfg, output_dir)


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> dict:
    return run_experiment(cfg, run)


def run(cfg: DictConfig) -> dict:
    mode = {
        "plan": "plan",
        "download": "download",
        "preflight": "preflight",
        "overhead": "overhead",
        "run": "run",
    }[str(cfg.mode)]
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    model_path = str(Path(cfg.dataset.model_path).expanduser().resolve())
    engine_config = _engine_config(cfg)
    order_seed = int(cfg.dataset.order_seed)
    protocol = str(cfg.dataset.workload_protocol)
    context_cap = int(cfg.engine.max_model_len)
    max_num_seqs = int(cfg.engine.max_num_seqs)
    max_num_batched_tokens = int(cfg.engine.max_num_batched_tokens)
    if mode == "plan":
        plan = build_plan(
            order_seed,
            workload_protocol=protocol,
            context_cap=context_cap,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        plan["configuration"] = {
            "engine": engine_config,
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
            "schedule": plan.get("schedule"),
            "plan_path": str(output_dir / "plan.json"),
        }
    if mode == "download":
        return {"mode": mode, **_download_model(cfg, output_dir)}
    if mode == "preflight":
        # Retain the old CLI spelling without executing environment/model checks.
        report = {"status": "disabled", "ready": None, "checks_performed": False}
        _save_json(output_dir / "preflight.json", report)
        return {"mode": mode, **report}
    if mode == "overhead":
        _prepare_model(cfg, output_dir)
        from taei.nano_dataset.overhead import measure_overhead

        pair_count = int(cfg.dataset.overhead_pair_count)
        with project_logging.experiment_progress(
            pair_count * 2, "Calibration profiles"
        ) as bar:

            def report_progress(stage: str, outcome: str | None) -> None:
                if outcome is None:
                    bar.stage(stage)
                else:
                    bar.advance(outcome)

            diagnostic = measure_overhead(
                model_path,
                engine_config,
                Path(cfg.dataset.overhead_output_root),
                workload_name=str(cfg.dataset.overhead_workload),
                seed=int(cfg.dataset.overhead_seed),
                pair_count=pair_count,
                progress_callback=report_progress,
            )
        return {
            "mode": mode,
            "diagnostic_id": diagnostic["diagnostic_id"],
            "status": diagnostic["status"],
            "calibration_applicable": diagnostic["calibration_applicable"],
            "report_path": diagnostic["path"],
        }
    _prepare_model(cfg, output_dir)
    plan = build_plan(
        order_seed,
        workload_protocol=protocol,
        context_cap=context_cap,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    limit = int(cfg.dataset.run_limit)
    selected = plan["matrix"][:limit] if limit > 0 else plan["matrix"]
    summaries = []
    with project_logging.experiment_progress(len(selected), "Dataset attempts") as bar:
        for index, item in enumerate(selected, start=1):
            bar.stage(
                f"Attempt {index}/{len(selected)}: waiting · {item['run_id'][:24]}"
            )
            attempt_summary = run_isolated(
                {
                    "run_spec": item,
                    "model_path": model_path,
                    "engine_config": engine_config,
                    "output_root": cfg.dataset.output_root,
                    "experiment_profile": dict(cfg.hardware_profile),
                    "telemetry_interval_s": float(cfg.dataset.telemetry_interval_s),
                    "logging_config": {
                        "logging": dict(cfg.get("logging", {})),
                        "clearml": dict(cfg.get("clearml", {})),
                    },
                },
                timeout_s=float(cfg.dataset.attempt_timeout_s),
            )
            summaries.append(attempt_summary)
            bar.advance(str(attempt_summary.get("status", "unknown")))
    result = {
        "mode": mode,
        "attempt_count": len(summaries),
        "runs": summaries,
        "validation_status": "disabled",
        "accepted_primary_count": 0,
        "failed_attempt_count": sum(
            item.get("status") != "success" for item in summaries
        ),
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
    if maybe_launch_detached(__file__):
        raise SystemExit(0)
    main()
