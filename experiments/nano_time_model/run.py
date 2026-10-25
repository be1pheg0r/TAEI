"""Подбор и проверка модели времени nano-vLLM по готовым прогонам.

Модель A (prefill) и выбранные способы decode подбираются на прогонах модели
обучения по одному стратифицированному разбиению 50/50 и без подгонки
применяются к каждой модели переноса. В каталог запуска ``runs/<время>/``
пишутся ``time_model.json`` (версионированный формат: коэффициенты, точки k,
метрики, разбиение, таблицы для графиков), ``report.html`` по общему шаблону и
копия кода; сводная ``runs/index.html`` перестраивается. GPU не используется.
"""

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from taei.launch import maybe_launch_detached
from taei.nano_time import data as nano_data
from taei.nano_time import report, report_data
from taei.nano_time.decode import select
from taei.nano_time.evaluate import TRAIN_SOURCE, evaluate
from taei.runtime import run_experiment

REPO = Path(__file__).resolve().parents[2]


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> dict:
    return run_experiment(cfg, run)


def _notes(ev, train, transfers) -> list[str]:
    ratio = train.runs["steps_seconds"] / train.runs["wall_seconds"]
    notes = [
        "Сумма измеренных шагов \\(\\sum_s T_s\\) — "
        f"{ratio.min():.2%}…{ratio.max():.2%} от "
        "\\(T^{\\mathrm{wall}}\\): промежутки между шагами и время до и после "
        "цикла моделью не описываются и входят в ошибку способов, которые "
        "суммируют шаги.",
        "Телеметрия пишется раз в ~1 с, а шаг длится доли секунды: частота шага — "
        "ближайшая выборка, а не среднее по шагу.",
    ]
    for name, method in ev.methods.items():
        if hasattr(method, "curves"):
            notes.append(
                f"У способа {name} train-прогоны сами задают узлы кривой, поэтому его "
                "ошибка на train занижена по построению; честная оценка — test. "
                "Прогоны с одинаковым \\(x\\) (например, разные планировщики при "
                "одном \\(p_{\\mathrm{decode}}\\)) кривая не различает."
            )
    for label, data in transfers.items():
        if (data.prefill_requests["C"] == 0).all():
            notes.append(
                f"{label}: кэш выключен (\\(R = 0\\) у всех запросов) — зависимость от "
                "\\(R\\) "
                "на этой модели проверить нельзя."
            )
        notes.append(
            f"{label}: {len(data.runs)} прогон(ов), {len(data.prefill_steps)} "
            f"prefill- и {len(data.decode_steps)} decode-шагов; оценка "
            "\\(\\hat T_{\\mathrm{total}}\\) — "
            f"{len(data.runs)} точка(и)."
        )
    return notes


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001 - recorded as text
        return f"unavailable: {type(error).__name__}"


def run(cfg: DictConfig) -> dict:
    out = Path(HydraConfig.get().runtime.output_dir)
    train = nano_data.load(
        REPO / cfg.train_data.runs_dir, cfg.train_data.label, cfg.train_data.run_ids
    )
    transfers = {
        t.label: nano_data.load(
            REPO / t.runs_dir, t.label, list(t.run_ids) if t.run_ids else None
        )
        for t in cfg.transfer_data or []
    }
    logger.info(
        "loaded {}: {} runs, {} prefill and {} decode steps; transfer: {}",
        cfg.train_data.label,
        len(train.runs),
        len(train.prefill_steps),
        len(train.decode_steps),
        list(transfers),
    )

    methods = select(cfg.decode)
    fit_kwargs = {"seed": int(cfg.seed), "trials": int(cfg.optuna.trials)}
    ev = evaluate(train, transfers, methods, cfg.fit, int(cfg.seed), fit_kwargs)
    meta = {
        "meta": {
            "experiment": cfg.experiment.name,
            "created_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "git_commit": _git_commit(),
            "fit": cfg.fit,
            "decode": cfg.decode,
            "seed": int(cfg.seed),
            "run_dir": out.name,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        "labels": {TRAIN_SOURCE: cfg.train_data.label},
    }
    document = report_data.build(ev, meta, _notes(ev, train, transfers))
    json_path = out / "time_model.json"
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    report_path = report.write_report(json_path)
    index_path = report.write_index(out.parent)
    code = out / "code"
    shutil.copytree(
        REPO / "src/taei/nano_time",
        code / "taei_nano_time",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        Path(__file__).parent,
        code / "experiment",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    logger.info("report: {} · index: {}", report_path, index_path)

    metrics = {"report": str(report_path), "index": str(index_path)}
    for row in document["metrics"]:
        name, _, kind = row["what"].partition(": ")
        if kind == "T_total per run" and row["part"] in ("test", "transfer"):
            key = "test" if row["part"] == "test" else row["source"].replace(" ", "_")
            metrics[f"{key}_{name}_mean_abs_err"] = row["mean_abs"]
    return metrics


if __name__ == "__main__":
    if maybe_launch_detached(__file__):
        raise SystemExit(0)
    main()
