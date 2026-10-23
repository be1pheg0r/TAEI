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
            "profile" in result.stderr.lower()
            or "invariant" in result.stderr.lower()
        )
