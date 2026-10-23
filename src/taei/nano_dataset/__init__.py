"""Сбор воспроизводимых нагрузок и данных экспериментов с nano-vLLM.

Пакет предоставляет построение матрицы запусков, детерминированных запросов,
адаптер движка, сбор метаданных и трасс, проверку артефактов и схему датасета.
Из этого модуля экспортируются основные функции построения матрицы и нагрузок.
"""

from .workload import build_run_matrix, build_workload, validate_shared_prefix_boundary

__all__ = ["build_run_matrix", "build_workload", "validate_shared_prefix_boundary"]
