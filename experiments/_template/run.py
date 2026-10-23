"""Точка входа Hydra и заменяемый шаблон детерминированного эксперимента.

Функция ``main`` собирает конфигурацию эксперимента из соседних файлов и
передает запуск общей среде для журналирования, сохранения результатов и
управления задачей. Callback ``run`` демонстрирует требуемый интерфейс:
читает параметры числа итераций и значения и возвращает синтетические метрики.
Внешних операций он не выполняет и предназначен для замены при копировании
папки шаблона для нового эксперимента.
"""

import hydra
from omegaconf import DictConfig

from taei.runtime import run_experiment


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> dict:
    return run_experiment(cfg, run)


def run(cfg: DictConfig) -> dict:
    """Runnable placeholder scenario to replace in a copied experiment folder."""
    iterations = int(cfg.metrics.iterations)
    value = float(cfg.metrics.value)
    # Replace this deterministic calculation with the experiment's real work.
    metrics = {
        "iterations_completed": iterations,
        "synthetic_total": iterations * value,
    }
    return metrics


if __name__ == "__main__":
    main()
