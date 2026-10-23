"""Проверяет точную сборку пилотной Hydra-конфигурации nano-vLLM.

Проверяются разрешенные параметры движка и модели, а также отсутствие
устаревших метаданных, которые могут неверно обозначить выбранную модель или
нарушить ограничения фиксированного пилотного запуска.
"""

import subprocess
import sys


def test_hydra_composes_exact_pilot_and_no_stale_8b_metadata():
    result = subprocess.run(
        [sys.executable, "experiments/nano_vllm_dataset/run.py", "--cfg", "job"],
        check=True,
        capture_output=True,
        text=True,
    )
    text = result.stdout
    assert "tensor_parallel_size: 4" in text
    assert "max_num_seqs: 8" in text
    assert "max_num_batched_tokens: 16384" in text
    assert "max_model_len: 4096" in text
    assert "enforce_eager: false" in text
    assert "huggingface_id: null" in text
    assert "architecture: null" in text
