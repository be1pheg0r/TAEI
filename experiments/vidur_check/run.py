"""Проверка законов модели времени на профилях Vidur; результат — ``vidur.json``.

Читает профили операций из ``data/vidur`` (``fetch.sh``), паспорта GPU из
``specs.yaml`` и наши коэффициенты из запуска модели времени (только чтение) и
пишет ``runs/<время>/vidur.json`` — документ отдельной ветки отчёта
(``kind: "vidur"``). Модель времени, её данные и ветки не меняются.
"""

import json
import subprocess
import sys
from pathlib import Path

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig

from taei.nano_time.report_data import clean
from taei.runtime import run_experiment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.vidur_check.analysis import analyze  # noqa: E402

LABEL = "Vidur: операции (A40 / A100 / H100)"
CAVEATS = [
    "Проверяются законы на уровне отдельных операций одного слоя на одном "
    "TP-шарде. Планировщика, накладных шага, Python-хоста и обмена между GPU в "
    "данных нет, поэтому прямого сравнения с нашими T_total нет.",
    "Берутся строки с наименьшим TP модели (везде TP = 1) и медианы "
    "time_stats.<операция>.median; prefill — один запрос в шаге. Строки с "
    "медианой 0 (ниже разрешения профиля, 1 мкс) отброшены: относительная "
    "ошибка на них не определена; их число — в таблице покрытия.",
    "Паспортные пики BF16 (без sparsity) и пропускная способность памяти — с "
    "источником в таблице GPU. Для A100 и H100 взяты варианты SXM: в Vidur "
    "есть профили сети a100_dgx и h100_dgx.",
    "Наша b — коэффициент при L²(1 − R²) = 2(C·C′ + C′²/2), поэтому из формы G1 "
    "b = β/2; a — линейная часть на токен (у Vidur — сумма операций mlp.csv).",
]


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> dict:
    return run_experiment(cfg, run)


def _commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _ours(specs: dict, run_json: Path) -> list[dict]:
    doc = json.loads(run_json.read_text(encoding="utf-8"))
    coef = {
        m["label"]: {**m["prefill_model"]["coef"], **m["decode_model"]["coef"]}
        for m in doc["models"]
    }
    return [{**o, "coef": coef[o["label"]]} for o in specs["ours"]]


def run(cfg: DictConfig) -> dict:
    out = Path(HydraConfig.get().runtime.output_dir)
    root = REPO / cfg.vidur_root
    specs = yaml.safe_load((REPO / cfg.specs).read_text(encoding="utf-8"))
    ours = _ours(specs, REPO / cfg.ours_run)
    result = analyze(root, specs, ours)
    commit = _commit(root)
    doc = clean(
        {
            "kind": "vidur",
            "label": LABEL,
            "meta": {
                "source": "github.com/microsoft/vidur, data/profiling (MIT)",
                "commit": commit,
                "specs": specs["gpus"],
                "ours_run": str(cfg.ours_run),
            },
            "caveats": CAVEATS,
            **result,
        }
    )
    path = out / "vidur.json"
    path.write_text(
        json.dumps(doc, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    logger.info(
        "vidur: {} GPU × model, commit {} -> {}",
        len(result["coverage"]),
        commit[:7],
        path,
    )
    return {"vidur_json": str(path), "pairs": len(result["coverage"]), "commit": commit}


if __name__ == "__main__":
    main()
