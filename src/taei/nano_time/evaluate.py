"""Подбор и оценка одной модели на одном разбиении; результат — таблицы для отчёта.

Модель A подбирается на prefill-шагах train-прогонов (и отдельно на прогонах
без кэша для проверки «train — без кэша, test — с кэшем», если прогоны с кэшем
есть), decode-шаг — на decode-шагах тех же train-прогонов. Время прогона —
сумма предсказаний по шагам; оценка — на test-прогонах, затем без подгонки на
каждом наборе переноса. Ошибка везде относительная: ``(предсказание − замер) /
замер``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import split as splits
from .data import Dataset
from .decode import DecodeModel, predict_total
from .fit import relative_errors, summary
from .prefill import PrefillModel

TRAIN_SOURCE = "train_model"


@dataclass
class Evaluation:
    split: dict[str, list[str]]
    cache_split: dict[str, list[str]] | None
    prefill: PrefillModel
    prefill_cache_split: PrefillModel | None
    decode: DecodeModel
    datasets: dict[str, Dataset]  # source label -> data (train model first)
    prefill_steps: pd.DataFrame  # every source, with predictions
    decode_steps: pd.DataFrame
    runs: pd.DataFrame  # one row per run
    metrics: list[dict] = field(default_factory=list)


def _parts(run_ids, split_map: dict[str, list[str]] | None, source: str) -> np.ndarray:
    if source != TRAIN_SOURCE:
        return np.array(["transfer"] * len(run_ids), dtype=object)
    lookup = {r: part for part, ids in (split_map or {}).items() for r in ids}
    return np.array([lookup.get(r, "unused") for r in run_ids], dtype=object)


def evaluate(
    train_data: Dataset,
    transfers: dict[str, Dataset],
    fit_method: str,
    seed: int,
    fit_kwargs: dict,
    split_by=splits.DEFAULT_KEYS,
) -> Evaluation:
    sp = splits.stratified(train_data.runs, seed, split_by)
    cache_sp = splits.cache_off_on(train_data.runs)
    pf = train_data.prefill_steps
    prefill = PrefillModel().fit(
        pf[pf["run_id"].isin(sp["train"])], fit_method, **fit_kwargs
    )
    prefill_cache = (
        PrefillModel().fit(
            pf[pf["run_id"].isin(cache_sp["train"])], fit_method, **fit_kwargs
        )
        if cache_sp
        else None
    )
    dc = train_data.decode_steps
    decode = DecodeModel().fit(
        dc[dc["run_id"].isin(sp["train"])], fit_method, **fit_kwargs
    )

    datasets = {TRAIN_SOURCE: train_data, **transfers}
    pf_rows, dc_rows, run_rows = [], [], []
    for source, data in datasets.items():
        steps = data.prefill_steps.copy()
        steps["source"] = source
        steps["part"] = _parts(steps["run_id"], sp, source)
        steps["pred"] = prefill.predict_steps(steps)
        steps["rel_err"] = relative_errors(steps["pred"], steps["T"])
        if prefill_cache is not None:
            steps["cache_part"] = _parts(steps["run_id"], cache_sp, source)
            steps["pred_cache_split"] = prefill_cache.predict_steps(steps)
            steps["rel_err_cache_split"] = relative_errors(
                steps["pred_cache_split"], steps["T"]
            )
        pf_rows.append(steps)

        dec = data.decode_steps.copy()
        dec["source"] = source
        dec["part"] = _parts(dec["run_id"], sp, source)
        dec["pred"] = decode.predict_steps(dec)
        dec["rel_err"] = relative_errors(dec["pred"], dec["T"])
        dc_rows.append(dec)

        runs = data.runs.set_index("run_id")
        ids = list(runs.index)
        frame = runs.assign(
            part=_parts(ids, sp, source),
            source=source,
            pred=predict_total(prefill, decode, data, ids).reindex(ids).to_numpy(),
            prefill_hat=prefill.predict_runs(data.prefill_steps)
            .reindex(ids)
            .to_numpy(),
        ).reset_index()
        frame["rel_err"] = relative_errors(frame["pred"], frame["wall_seconds"])
        run_rows.append(frame)

    ev = Evaluation(
        sp,
        cache_sp,
        prefill,
        prefill_cache,
        decode,
        datasets,
        pd.concat(pf_rows, ignore_index=True),
        pd.concat(dc_rows, ignore_index=True),
        pd.concat(run_rows, ignore_index=True),
    )
    ev.metrics = _metrics(ev)
    return ev


def _metrics(ev: Evaluation) -> list[dict]:
    rows = []

    def add(what, group, source, part, rel, coef):
        rel = pd.Series(rel).dropna()
        if len(rel):
            rows.append(
                {
                    "what": what,
                    "group": group,
                    "source": source,
                    "part": part,
                    **summary(rel),
                    "coef": coef,
                }
            )

    def by_part(frame):
        for (source, part), sub in frame.groupby(["source", "part"], sort=False):
            if part != "unused":
                yield source, part, sub

    pf = ev.prefill_steps
    for source, part, sub in by_part(pf):
        add("prefill step", "prefill", source, part, sub["rel_err"], ev.prefill.coef)
    if ev.prefill_cache_split is not None:
        train_pf = pf[pf["source"] == TRAIN_SOURCE]
        for part in ("train", "test"):
            add(
                "prefill step (train cache off, test cache on)",
                "prefill",
                TRAIN_SOURCE,
                part,
                train_pf.loc[train_pf["cache_part"] == part, "rel_err_cache_split"],
                ev.prefill_cache_split.coef,
            )
    run_pf = pf.groupby(["source", "run_id", "part"])[["pred", "T"]].sum().reset_index()
    for source, part, sub in by_part(run_pf):
        add(
            "prefill total per run",
            "prefill",
            source,
            part,
            relative_errors(sub["pred"], sub["T"]),
            ev.prefill.coef,
        )
    for source, part, sub in by_part(ev.decode_steps):
        add("decode step", "decode", source, part, sub["rel_err"], ev.decode.coef)
    coef = {**ev.prefill.coef, **ev.decode.coef}
    for source, part, sub in by_part(ev.runs):
        add("T_total per run", "total", source, part, sub["rel_err"], coef)
    return rows
