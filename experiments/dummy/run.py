import hydra
from omegaconf import DictConfig, OmegaConf

from taei.runtime import run_experiment


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> dict:
    return run_experiment(cfg, run)


def run(cfg: DictConfig) -> dict:
    """Return the configured deterministic example metrics."""
    metrics = OmegaConf.to_container(cfg.metrics, resolve=True)
    return metrics


if __name__ == "__main__":
    main()
