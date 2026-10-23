import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from taei.logging import configure_logging


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    configure_logging(level=config.logging.level)
    experiment = instantiate(config.experiment)
    experiment.run()


if __name__ == "__main__":
    main()
