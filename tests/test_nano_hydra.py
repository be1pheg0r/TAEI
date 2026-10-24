"""Проверяет точную сборку пилотной Hydra-конфигурации nano-vLLM.

Проверяются разрешенные параметры движка и модели, а также отсутствие
устаревших метаданных, которые могут неверно обозначить выбранную модель или
нарушить ограничения фиксированного пилотного запуска.
"""

import subprocess
import sys


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
    assert "outputs/nano_vllm_dataset_rtx_a4000_tp2/runs" in text
    assert "hardware_deviation: true" in text


def test_qwen3_8b_profile_keeps_tp2_experiment_and_uses_its_own_model():
    text = _compose("profile@_global_=rtx_a4000_tp2_qwen3_8b")
    assert "profile_id: rtx_a4000_tp2_qwen3_8b" in text
    assert "experiment_name: nano_vllm_dataset_rtx_a4000_tp2_qwen3_8b" in text
    assert "outputs/nano_vllm_dataset_rtx_a4000_tp2_qwen3_8b/runs" in text
    assert "TAEI_QWEN3_8B_PATH" in text
    assert "Qwen/Qwen3-8B" in text
    assert "b968826d9c46dd6066d109eabc6255188de91218" in text
    assert "name: Qwen3-8B" in text
    assert "huggingface_id: Qwen/Qwen3-8B" in text
    assert "tensor_parallel_size: 2" in text
    assert "max_num_seqs: 8" in text
    assert "max_num_batched_tokens: 16384" in text
    assert "max_model_len: 4096" in text
    assert "enforce_eager: true" in text
    assert "attempt_timeout_s: 1800" in text
    assert "run_limit: 0" in text
    assert "repo_id: Qwen/Qwen3-8B" in text
    assert "Qwen3-0.6B" not in text
    assert "c1899de289a04d12100db370d81485cdf75e47ca" not in text


def test_qwen3_8b_profile_plan_runs_without_model_assets():
    result = subprocess.run(
        [
            sys.executable,
            "experiments/nano_vllm_dataset/run.py",
            "profile@_global_=rtx_a4000_tp2_qwen3_8b",
            "mode=plan",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "metric run_count=24" in result.stderr


def test_qwen3_8b_model_identity_cannot_be_overridden():
    result = subprocess.run(
        [
            sys.executable,
            "experiments/nano_vllm_dataset/run.py",
            "profile@_global_=rtx_a4000_tp2_qwen3_8b",
            "mode=plan",
            "download.repo_id=Qwen/Qwen3-0.6B",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "model identity" in result.stderr.lower()


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
