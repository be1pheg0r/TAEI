"""Проверяет точную сборку пилотной Hydra-конфигурации nano-vLLM.

Проверяются разрешенные параметры движка и модели, а также отсутствие
устаревших метаданных, которые могут неверно обозначить выбранную модель или
нарушить ограничения фиксированного пилотного запуска.
"""

import subprocess
import sys

from omegaconf import OmegaConf


def _compose(*overrides):
    return subprocess.run(
        [
            sys.executable,
            "experiments/nano_vllm_dataset/run.py",
            "--cfg",
            "job",
            *overrides,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_hydra_composes_exact_source_tp4_pilot():
    text = _compose()
    assert "tensor_parallel_size: 4" in text
    assert "max_num_seqs: 8" in text
    assert "max_num_batched_tokens: 16384" in text
    assert "max_model_len: 4096" in text
    assert "enforce_eager: false" in text
    assert "huggingface_id: null" in text
    assert "architecture: null" in text


def test_hydra_composes_distinct_rtx_a4000_tp2_profile():
    text = _compose("profile@_global_=rtx_a4000_tp2")
    assert "tensor_parallel_size: 2" in text
    assert "profile_id: rtx_a4000_tp2" in text
    assert "name: nano_vllm_dataset_rtx_a4000_tp2" in text
    assert "output_root: ${hydra:runtime.output_dir}/runs" in text
    assert "hardware_deviation: true" in text


def test_tp2_overrides_cannot_be_misrepresented_as_source_pilot():
    for override in (
        "engine.tensor_parallel_size=4",
        "dataset.output_root=outputs/nano_vllm_dataset/runs",
    ):
        result = subprocess.run(
            [
                sys.executable,
                "experiments/nano_vllm_dataset/run.py",
                "mode=plan",
                "profile@_global_=rtx_a4000_tp2",
                override,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert (
            "profile" in result.stderr.lower() or "invariant" in result.stderr.lower()
        )


def test_plan_invocations_keep_config_and_outputs_in_their_hydra_directory(tmp_path):
    invocation_dirs = [tmp_path / "first", tmp_path / "second"]
    configured_roots = []
    configured_overhead_roots = []
    for output_dir in invocation_dirs:
        result = subprocess.run(
            [
                sys.executable,
                "experiments/nano_vllm_dataset/run.py",
                "mode=plan",
                f"hydra.run.dir={output_dir}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "Finished experiment: nano_vllm_dataset" in result.stderr
        config = OmegaConf.load(output_dir / "config.yaml")
        configured_roots.append(config.dataset.output_root)
        configured_overhead_roots.append(config.dataset.overhead_output_root)
        assert (output_dir / "plan.json").is_file()
        assert (output_dir / "results.json").is_file()

    assert configured_roots == [str(path / "runs") for path in invocation_dirs]
    assert configured_overhead_roots == [
        str(path / "overhead") for path in invocation_dirs
    ]
    assert configured_roots[0] != configured_roots[1]


def test_run_summary_and_supervised_attempt_use_invocation_scoped_runs(
    tmp_path, monkeypatch
):
    import importlib.util
    import json
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    experiment_path = "experiments/nano_vllm_dataset"
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(experiment_path).resolve())
    ):
        config = compose(
            config_name="config",
            overrides=["mode=run", "dataset.run_limit=1"],
            return_hydra_config=True,
        )
    invocation = tmp_path / "invocation"
    config.hydra.runtime.output_dir = str(invocation)
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: config.hydra))
    spec = importlib.util.spec_from_file_location(
        "nano_vllm_dataset_run", Path(experiment_path, "run.py").resolve()
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_prepare_for_inference", lambda *args: {})
    monkeypatch.setattr(module, "_load_calibration_report", lambda _path: None)
    run_spec = {
        "run_id": "run-fixture",
        "run_order": 1,
        "condition_id": "condition-fixture",
    }
    monkeypatch.setattr(module, "build_plan", lambda _seed: {"matrix": [run_spec]})
    seen_paths = []

    def fake_run_isolated(payload, *, timeout_s):
        attempt_path = (
            Path(payload["output_root"]) / run_spec["run_id"] / "attempt-fixture"
        )
        attempt_path.mkdir(parents=True)
        seen_paths.append(attempt_path)
        return {
            "run_id": run_spec["run_id"],
            "attempt_id": "attempt-fixture",
            "path": str(attempt_path),
            "status": "success",
            "primary_status": "rejected",
            "hardware_id": "H-fixture",
            "environment_id": "E-fixture",
            "model_id": "M-fixture",
            "backend_constructed": True,
            "real_generate_completed": True,
        }

    monkeypatch.setattr(module, "run_isolated", fake_run_isolated)
    result = module.run(config)

    expected_root = invocation / "runs"
    assert seen_paths == [expected_root / "run-fixture" / "attempt-fixture"]
    assert result["runs"][0]["path"] == str(seen_paths[0])
    summary = json.loads((invocation / "dataset-summary.json").read_text())
    assert summary["runs"][0]["path"] == str(seen_paths[0])
