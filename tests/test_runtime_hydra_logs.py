"""Проверяет разделение логов Hydra и общей записи runtime при реальном запуске.

Интеграционный тест запускает эксперимент через Hydra и проверяет, что в его
каталоге framework-лог Hydra отделен от объединенного захвата ``run.log``.
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path


def test_real_hydra_cli_separates_hydra_and_combined_logs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    (experiment_dir / "config.yaml").write_text(
        "defaults: [base, _self_]\n"
        "hydra:\n  searchpath: [pkg://taei.conf]\n"
        "experiment: {name: cli-logs, description: logging integration}\n",
        encoding="utf-8",
    )
    (experiment_dir / "run.py").write_text(
        "import logging\n"
        "from loguru import logger\n"
        "import hydra\n"
        "from taei.runtime import run_experiment\n"
        "@hydra.main(version_base='1.3', config_path='.', config_name='config')\n"
        "def main(config):\n"
        "    def callback(_):\n"
        "        logging.warning('STDLIB_MARKER')\n"
        "        logger.warning('LOGURU_MARKER')\n"
        "        print('PRINT_MARKER')\n"
        "        return {}\n"
        "    run_experiment(config, callback)\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(experiment_dir / "run.py")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output_dirs = list((tmp_path / "outputs" / "cli-logs").iterdir())
    assert len(output_dirs) == 1
    output_dir = output_dirs[0]
    run_log = (output_dir / "run.log").read_text(encoding="utf-8")
    hydra_log = (output_dir / "hydra.log").read_text(encoding="utf-8")
    assert run_log.count("STDLIB_MARKER") == 1
    assert run_log.count("LOGURU_MARKER") == 1
    assert run_log.count("PRINT_MARKER") == 1
    assert "STDLIB_MARKER" in hydra_log
    with zipfile.ZipFile(output_dir / "logs.zip") as archive:
        assert set(archive.namelist()) == {"hydra.log", "run.log"}
        assert archive.read("run.log").decode("utf-8") == run_log
        assert archive.read("hydra.log").decode("utf-8") == hydra_log
