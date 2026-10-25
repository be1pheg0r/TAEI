"""Подбор и проверка модели времени nano-vLLM по готовым прогонам.

Модель A (prefill) и выбранные способы decode подбираются на прогонах первой
модели по одному стратифицированному разбиению 50/50 и без подгонки
применяются ко второй модели. В каталог запуска ``runs/<время>/`` пишутся
``report.html``, ``time_model.json`` (коэффициенты, точки k, метрики, разбиение,
пометки) и копия кода, которым они получены. GPU не используется.
"""

import json
import shutil
import subprocess
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from taei.launch import maybe_launch_detached
from taei.nano_time import data as nano_data
from taei.nano_time import report
from taei.nano_time.decode import select
from taei.nano_time.evaluate import evaluate
from taei.runtime import run_experiment

REPO = Path(__file__).resolve().parents[2]


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> dict:
    return run_experiment(cfg, run)


def _notes(ev, train, transfer) -> list[str]:
    notes = []
    ratio = train.runs["steps_seconds"] / train.runs["wall_seconds"]
    notes.append(
        f"Сумма измеренных шагов составляет {ratio.min():.2%}–{ratio.max():.2%} от "
        "wall_seconds: промежутки между шагами и ~13 мс до и после цикла моделью не "
        "описываются и входят в ошибку способа additive."
    )
    notes.append(
        "Телеметрия пишется раз в ~1 с, а decode-шаг длится ~0,06–0,3 с: частота шага "
        "— "
        "ближайшая выборка, а не среднее по шагу."
    )
    if "multiplicative" in ev.methods:
        notes.append(
            "Ошибка multiplicative на train занижена по построению: train-прогоны "
            "сами задают "
            "узлы k, а остаток на train — это разброс между прогонами с одинаковым "
            "p_decode "
            "(разные планировщики), которые объединены медианой и которые k не "
            "различает. "
            "Честная оценка — только test."
        )
    if transfer is not None:
        if (transfer.prefill_requests["C"] == 0).all():
            notes.append(
                "У переноса кэш выключен (R = 0 у всех запросов): зависимость от R на "
                "второй "
                "модели проверить нельзя."
            )
        notes.append(
            f"У переноса {len(transfer.prefill_steps)} prefill-шагов и "
            f"{len(transfer.decode_steps)} decode-шагов в одном прогоне: оценка "
            f"переноса "
            "по прогону — одна точка."
        )
    mixed = 0  # nano-vllm schedules prefill and decode in separate steps
    notes.append(f"Смешанных prefill+decode шагов в данных нет ({mixed}).")
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
    transfer = None
    if cfg.transfer_data and cfg.transfer_data.runs_dir:
        transfer = nano_data.load(
            REPO / cfg.transfer_data.runs_dir,
            cfg.transfer_data.label,
            list(cfg.transfer_data.run_ids or []) or None,
        )
    logger.info(
        "loaded {} train-model runs, {} prefill and {} decode steps",
        len(train.runs),
        len(train.prefill_steps),
        len(train.decode_steps),
    )

    methods = select(cfg.decode)
    fit_kwargs = {"seed": int(cfg.seed), "trials": int(cfg.optuna.trials)}
    ev = evaluate(train, transfer, methods, cfg.fit, int(cfg.seed), fit_kwargs)

    notes = _notes(ev, train, transfer)
    test = ev.runs[ev.runs["part"] == "test"]
    by_method = test.assign(a=test["rel_err"].abs()).groupby("method")["a"].mean()
    summary = (
        f"Подбор: {cfg.train_data.label}, {len(train.runs)} прогонов "
        f"(train {len(ev.split['train'])}, test {len(ev.split['test'])}, seed "
        f"{cfg.seed}); "
        f"fit={cfg.fit}, decode={cfg.decode}. Средняя |ошибка| T_total на test: "
        + ", ".join(f"{m} {v:.1%}" for m, v in by_method.items())
        + f". Перенос без подгонки: {cfg.transfer_data.label}."
    )
    meta = {
        "summary": summary,
        "runs": train.runs,
        "prefill_requests": train.prefill_requests,
        "transfer_requests": transfer.prefill_requests if transfer else None,
        "transfer_runs": transfer.runs if transfer else None,
        "transfer_label": cfg.transfer_data.label,
    }
    (out / "report.html").write_text(report.render(ev, meta, notes), encoding="utf-8")

    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "git_commit": _git_commit(),
        "split": ev.split,
        "prefill_model_A": ev.prefill.coef,
        "prefill_model_A_cache_split": ev.prefill_cache_split.coef,
        "decode_methods": {name: m.describe() for name, m in ev.methods.items()},
        "metrics": ev.metrics,
        "runs": json.loads(
            ev.runs.drop(columns=[], errors="ignore").to_json(orient="records")
        ),
        "paired_test_comparison": report.paired_comparison(test),
        "notes": notes,
    }
    (out / "time_model.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
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
    logger.info("report: {}", out / "report.html")

    metrics = {"report": str(out / "report.html")}
    for name, value in by_method.items():
        metrics[f"test_mean_abs_err_{name}"] = float(value)
    tr = ev.runs[ev.runs["part"] == "transfer"]
    for _, row in tr.iterrows():
        metrics[f"transfer_{row['run_id']}_rel_err_{row['method']}"] = float(
            row["rel_err"]
        )
    a_test = [
        m for m in ev.metrics if m["what"] == "A: prefill step" and m["part"] == "test"
    ]
    if a_test:
        metrics["test_mean_abs_err_prefill_step"] = float(a_test[0]["mean_abs"])
    return {
        k: (float(v) if isinstance(v, np.floating) else v) for k, v in metrics.items()
    }


if __name__ == "__main__":
    if maybe_launch_detached(__file__):
        raise SystemExit(0)
    main()
