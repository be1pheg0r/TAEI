"""CPU-only smoke-эксперимент Hydra для проверки локального запуска.

Точка входа использует общую среду экспериментов. Callback читает заданное
число smoke-проверок, выполняет небольшую операцию подсчета на чистом Python
и возвращает разрешенную конфигурацию метрик с фактическим числом завершенных
проверок. Скрипт позволяет проверить конфигурацию, журналирование и артефакты
без сети, загрузки модели или использования GPU.
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from taei.runtime import run_experiment


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> dict:
    return run_experiment(cfg, run)


def run(cfg: DictConfig) -> dict:
    """Run a tiny local sanity check and return its configured count."""
    checks = int(cfg.metrics.smoke_checks)
    # Pure Python arithmetic keeps this smoke path CPU-only and network-free.
    completed = sum(1 for _ in range(checks))
    configured = OmegaConf.to_container(cfg.metrics, resolve=True)
    configured["smoke_checks"] = completed
    return configured


if __name__ == "__main__":
    main()
