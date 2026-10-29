"""Подбор и проверка модели времени nano-vLLM по готовым прогонам.

Для каждой модели из конфига модель A (prefill) и decode-шаг подбираются на её
прогонах по своему стратифицированному разбиению 50/50 и без подгонки
применяются к её наборам переноса. В каталог запуска ``runs/<время>/`` пишутся
``time_model.json`` (версионированный формат: коэффициенты, метрики, разбиения,
цены токенов, таблицы для графиков), ``report.html`` по общему шаблону и копия
кода; сводная ``runs/index.html`` перестраивается. GPU не используется.
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
from taei.nano_time.evaluate import TRAIN_SOURCE, evaluate
from taei.nano_time.local_report import build_local
from taei.nano_time.prefill_forms import analyze as analyze_prefill_forms
from taei.runtime import run_experiment

REPO = Path(__file__).resolve().parents[2]


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> dict:
    return run_experiment(cfg, run)


def _notes(label, train, transfers, attempts) -> list[str]:
    ratio = train.runs["steps_seconds"] / train.runs["wall_seconds"]
    notes = [
        "Сумма измеренных шагов \\(\\sum_s T_s\\) — "
        f"{ratio.min():.2%}…{ratio.max():.2%} от "
        "\\(T^{\\mathrm{wall}}\\): промежутки между шагами и время до и после "
        "цикла моделью не описываются и входят в ошибку \\(T_{\\mathrm{total}}\\).",
        "В проверке шага ожидание в TTFT по модели взято из трассы (реальный "
        "порядок шагов); у локальной 0.6B очередь предсказывает модель «от N».",
    ]
    failed = attempts[attempts["status"] != "success"]
    if len(failed):
        kinds = failed.groupby(["cache", "error"]).size()
        text = ", ".join(f"кэш {c}: {n} × {e}" for (c, e), n in kinds.items())
        notes.append(
            f"{label}: {len(failed)} из {len(attempts)} попыток упали ({text}); "
            "в данных только удачные."
        )
    lost = train.runs[train.runs["lost_tokens"] > 0]
    if len(lost):
        notes.append(
            f"{label}: в {len(lost)} прогонах сборщик потерял вытесненный запрос "
            f"(−{int(lost['lost_tokens'].iloc[0])} токенов). Шаги трассы верны, "
            "поэтому \\(T_{\\mathrm{total}}\\) сравнивается честно; в TTPS–TTFT "
            "потерянный запрос не входит."
        )
    if (train.prefill_requests["C"] == 0).all():
        notes.append(f"{label}: кэша нет и нарезки нет, \\(R = 0\\) у всех запросов.")
    elif (train.runs["cache"] == "on").sum() == 0:
        notes.append(
            f"{label}: попаданий в кэш нет (\\(R_{{\\text{{кэш}}}} = 0\\)); "
            "\\(R_{\\text{нарезка}} > 0\\) только у вторых кусков chunked prefill "
            "(свой готовый KV), поэтому разбиения «train без кэша, test с кэшем» нет."
        )
    state = lambda d: (  # noqa: E731
        f"SM {d.runs['gpu_sm_mhz'].median():.0f} МГц, загрузка GPU "
        f"{d.runs['gpu_util'].median():.0f}%, мощность "
        f"{d.runs['gpu_power_w'].median():.0f} Вт, "
        f"KV-кэш {d.runs['kv_blocks'].median():.0f} блоков"
    )
    for name, data in transfers.items():
        notes.append(
            f"Состояние машины (медианы по видимым GPU): {name} — {state(data)}; "
            f"{label} — {state(train)}. Если GPU у переноса заняты чужими задачами "
            "(загрузка 100%, предел мощности, мало KV-блоков из-за чужой памяти), "
            "ошибка переноса — это разница машин, а не модели."
        )
        if (data.prefill_requests["C_cache"] == 0).all():
            notes.append(
                f"{name}: кэш выключен (\\(R_{{\\text{{кэш}}}} = 0\\)) — "
                "зависимость от кэша на этом наборе проверить нельзя."
            )
        notes.append(
            f"{name}: {len(data.runs)} прогон(ов), {len(data.prefill_steps)} "
            f"prefill- и {len(data.decode_steps)} decode-шагов."
        )
    return notes


def _local_notes(local: dict) -> list[str]:
    c = local["conditions"]
    return [
        f"Своя GPU ({c['gpu']}), чужих задач нет; частота SM зафиксирована, во всех "
        f"попытках {c['sm_clock_min']:.0f}–{c['sm_clock_max']:.0f} МГц. Попытки, где "
        "прошивка ноутбука срезала мощность (частота ниже фиксированной), сборщик "
        "отбрасывает и повторяет — в этом сборе таких не было.",
        "S4: приход запросов во времени. TTFT по модели — предсказанные шаги от "
        "первого шага после прихода запроса до шага его первого токена; само "
        "расписание (какие шаги и в каком порядке) взято из трассы.",
        "Prefill-формы для этой модели подбираются без множителя прогона (машина "
        "одна, у прогона сетки обычно один prefill-шаг) и проверяются "
        "leave-one-point-out: все повторы одной точки сетки уходят в test вместе.",
    ]


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


def _root(entry):
    dirs = entry.runs_dir
    return [REPO / d for d in dirs] if OmegaConf.is_list(dirs) else REPO / dirs


def _load(entry) -> nano_data.Dataset:
    ids = list(entry.run_ids) if entry.get("run_ids") else None
    series = list(entry.series) if entry.get("series") else None
    return nano_data.load(_root(entry), entry.label, ids, series)


def run(cfg: DictConfig) -> dict:
    out = Path(HydraConfig.get().runtime.output_dir)
    fit_kwargs = {"seed": int(cfg.seed), "trials": int(cfg.optuna.trials)}
    meta = {
        "experiment": cfg.experiment.name,
        "created_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "git_commit": _git_commit(),
        "fit": cfg.fit,
        "seed": int(cfg.seed),
        "run_dir": out.name,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    models = []
    for entry in cfg.models:
        train = _load(entry)
        transfers = {t.label: _load(t) for t in entry.transfer or []}
        logger.info(
            "loaded {}: {} runs, {} prefill and {} decode steps; transfer: {}",
            entry.label,
            len(train.runs),
            len(train.prefill_steps),
            len(train.decode_steps),
            list(transfers),
        )
        split_by = OmegaConf.to_container(entry.split_by)
        ev = evaluate(train, transfers, cfg.fit, int(cfg.seed), fit_kwargs, split_by)
        series = list(entry.series) if entry.get("series") else None
        notes = _notes(
            entry.label, train, transfers, nano_data.attempts(_root(entry), series)
        )
        forms = local = None
        if entry.get("local"):
            # Own GPU, one prefill step per grid run: forms without a run factor,
            # checked leave-one-point-out on the train grids.
            forms = analyze_prefill_forms(
                nano_data.subset(train, ev.split["train"]), "workload", False
            )
            labels = {TRAIN_SOURCE: entry.label}
            system = (
                OmegaConf.to_container(entry.system) if entry.get("system") else None
            )
            if system:
                for key in ("check_roots", "step_roots"):
                    system[key] = [REPO / r for r in system[key]]
            local = build_local(
                ev,
                [REPO / r for r in entry.local.roots],
                [REPO / p for p in entry.local.plans],
                REPO / entry.local.server_runs_dir,
                labels,
                system,
                OmegaConf.to_container(cfg.economics),
            )
            notes += _local_notes(local)
        caveats = {t.label: t.caveat for t in entry.transfer or [] if t.get("caveat")}
        window = OmegaConf.to_container(cfg.windows)
        models.append(
            report_data.build(
                ev,
                entry.label,
                meta,
                notes,
                forms,
                local,
                window,
                caveats,
                entry.get("caveat"),
            )
        )
        logger.info(
            "{}: prefill {} · decode {}", entry.label, ev.prefill.coef, ev.decode.coef
        )

    # Extra branches (for example the Vidur check) are finished documents of
    # their own experiments; they are appended after our models, unchanged.
    for path in cfg.get("extra_branches") or []:
        models.append(json.loads((REPO / path).read_text(encoding="utf-8")))
    document = report_data.document(meta, models)
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
    for model in models:
        key = model["label"].replace(" ", "_")
        for row in model.get("metrics", []):  # extra branches have none
            if row["what"] == "T_total per run" and row["part"] in ("test", "transfer"):
                part = row["part"] if row["part"] == "test" else "transfer"
                metrics[f"{key}_{part}_T_total_mean_abs_err"] = row["mean_abs"]
    return metrics


if __name__ == "__main__":
    if maybe_launch_detached(__file__):
        raise SystemExit(0)
    main()
