import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from taei.logging import configure_logging


def initialize_clearml(config: DictConfig) -> None:
    """Create a ClearML task when enabled, so the logging sink has a target."""
    if not config.clearml.enabled:
        return

    from clearml import Task

    if Task.current_task() is None:
        Task.init(
            project_name=config.clearml.project_name,
            task_name=config.experiment.name,
            auto_connect_frameworks=False,
        )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    initialize_clearml(config)
    configure_logging(config=config)
    experiment = instantiate(config.experiment)
    experiment.run()


if __name__ == "__main__":
    main()
