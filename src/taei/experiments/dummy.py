from typing import Any

from loguru import logger

from taei.experiments.base import Experiment


class DummyExperiment(Experiment):
    """Emit configured metrics to demonstrate logger collection."""

    def run(self) -> dict[str, Any]:
        metrics = dict(self.metrics)
        logger.info("experiment={} status=started", self.name)
        for name, value in metrics.items():
            logger.info("metric {}={}", name, value)
        logger.info("experiment={} status=finished", self.name)
        return metrics
