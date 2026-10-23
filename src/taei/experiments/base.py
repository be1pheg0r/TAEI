from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig


class Experiment(ABC):
    """Base class for Hydra-instantiated experiment runs."""

    def __init__(self, name: str, metrics: DictConfig, description: str = "") -> None:
        self.name = name
        self.metrics = metrics
        self.description = description

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Run the experiment and return its collected metrics."""
