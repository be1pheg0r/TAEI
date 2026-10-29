"""Qwen3-0.6B experiment entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from taei.launch import maybe_launch_detached
from taei.runtime import run_experiment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.nano_vllm_dataset.run import run


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> dict:
    return run_experiment(cfg, run)


if __name__ == "__main__":
    if maybe_launch_detached(__file__):
        raise SystemExit(0)
    main()
