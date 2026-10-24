"""Пример эксперимента под управлением Hydra, возвращающий заданные метрики.

Точка входа передает выполнение общей среде экспериментов. Callback ``run``
разрешает раздел ``metrics`` собранной Hydra-конфигурации в обычное Python-
отображение и возвращает значения без изменений. Это позволяет проверять
переопределения конфигурации и сохранение результатов/логов без затратных
вычислений и внешних побочных эффектов.
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from taei.launch import maybe_launch_detached
from taei.runtime import run_experiment


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> dict:
    return run_experiment(cfg, run)


def run(cfg: DictConfig) -> dict:
    """Return the configured deterministic example metrics."""
    metrics = OmegaConf.to_container(cfg.metrics, resolve=True)
    return metrics


if __name__ == "__main__":
    if maybe_launch_detached(__file__):
        raise SystemExit(0)
    main()
