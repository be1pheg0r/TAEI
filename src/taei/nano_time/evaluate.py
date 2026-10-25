"""Подбор и оценка всех частей на одном разбиении; результат — таблицы для отчёта.

Модель A подбирается на prefill-шагах train-прогонов (и отдельно на прогонах
без кэша для проверки «train — без кэша, test — с кэшем»). Каждый выбранный
способ decode подбирается на тех же train-прогонах и оценивается на тех же
test-прогонах, а затем без подгонки применяется к каждой модели переноса.
Способы обрабатываются одинаково, по именам их не различают: необязательные
хуки способа — ``predict_steps`` (предсказание decode-шагов), ``diagnostics``
(дополнительные предсказания T_total по прогонам) и ``curves`` (кривые для
отчёта). Ошибка везде относительная: ``(предсказание − замер) / замер``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import split as splits
from .data import Dataset
from .fit import relative_errors, summary
from .prefill import PrefillModel

TRAIN_SOURCE = "train_model"


@dataclass
class Evaluation:
    split: dict[str, list[str]]
    cache_split: dict[str, list[str]]
    prefill: PrefillModel
    prefill_cache_split: PrefillModel
    methods: dict[str, object]
    datasets: dict[str, Dataset]  # source label -> data (train model first)
    prefill_steps: pd.DataFrame  # every source, with predictions
    decode_steps: pd.DataFrame
    runs: pd.DataFrame  # one row per (run, method)
    metrics: list[dict] = field(default_factory=list)


def _parts(run_ids, split_map: dict[str, list[str]], source: str) -> np.ndarray:
    if source != TRAIN_SOURCE:
        return np.array(["transfer"] * len(run_ids), dtype=object)
    lookup = {r: part for part, ids in split_map.items() for r in ids}
    return np.array([lookup.get(r, "unused") for r in run_ids], dtype=object)


def evaluate(
    train_data: Dataset,
    transfers: dict[str, Dataset],
    methods: dict[str, type],
    fit_method: str,
    seed: int,
    fit_kwargs: dict,
) -> Evaluation:
    sp = splits.stratified(train_data.runs, seed)
    cache_sp = splits.cache_off_on(train_data.runs)
    pf = train_data.prefill_steps
    prefill = PrefillModel().fit(
        pf[pf["run_id"].isin(sp["train"])], fit_method, **fit_kwargs
    )
    prefill_cache = PrefillModel().fit(
        pf[pf["run_id"].isin(cache_sp["train"])], fit_method, **fit_kwargs
    )
    fitted = {
        name: cls().fit(train_data, sp["train"], prefill, fit_method, **fit_kwargs)
        for name, cls in methods.items()
    }

    datasets = {TRAIN_SOURCE: train_data, **transfers}
    pf_rows, dc_rows, run_rows = [], [], []
    for source, data in datasets.items():
        steps = data.prefill_steps.copy()
        steps["source"] = source
        steps["part"] = _parts(steps["run_id"], sp, source)
        steps["cache_part"] = _parts(steps["run_id"], cache_sp, source)
        steps["pred"] = prefill.predict_steps(steps)
        steps["pred_cache_split"] = prefill_cache.predict_steps(steps)
        steps["rel_err"] = relative_errors(steps["pred"], steps["T"])
        steps["rel_err_cache_split"] = relative_errors(
            steps["pred_cache_split"], steps["T"]
        )
        pf_rows.append(steps)

        dec = data.decode_steps.copy()
        dec["source"] = source
        dec["part"] = _parts(dec["run_id"], sp, source)
        for name, method in fitted.items():
            if hasattr(method, "predict_steps"):
                dec[f"pred__{name}"] = method.predict_steps(dec)
                dec[f"rel_err__{name}"] = relative_errors(
                    dec[f"pred__{name}"], dec["T"]
                )
        dc_rows.append(dec)

        runs = data.runs.set_index("run_id")
        ids = list(runs.index)
        t_prefill_hat = prefill.predict_runs(data.prefill_steps).reindex(ids)
        for name, method in fitted.items():
            frame = runs.assign(
                method=name,
                part=_parts(ids, sp, source),
                source=source,
                pred=method.predict_total(data, ids).reindex(ids).to_numpy(),
                prefill_hat=t_prefill_hat.to_numpy(),
                flag=method.flags(data, ids).reindex(ids).to_numpy(),
            ).reset_index()
            frame["rel_err"] = relative_errors(frame["pred"], frame["wall_seconds"])
            if hasattr(method, "diagnostics"):
                for key, values in method.diagnostics(data, ids).items():
                    frame[f"diag__{key}"] = values.reindex(ids).to_numpy()
            run_rows.append(frame)

    ev = Evaluation(
        sp,
        cache_sp,
        prefill,
        prefill_cache,
        fitted,
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
        add("A: prefill step", "prefill", source, part, sub["rel_err"], ev.prefill.coef)
    train_pf = pf[pf["source"] == TRAIN_SOURCE]
    for part in ("train", "test"):
        add(
            "A: prefill step (train cache off, test cache on)",
            "prefill",
            TRAIN_SOURCE,
            part,
            train_pf.loc[train_pf["cache_part"] == part, "rel_err_cache_split"],
            ev.prefill_cache_split.coef,
        )
    run_pf = pf.groupby(["source", "run_id", "part"])[["pred", "T"]].sum().reset_index()
    for source, part, sub in by_part(run_pf):
        add(
            "A: prefill total per run",
            "prefill",
            source,
            part,
            relative_errors(sub["pred"], sub["T"]),
            ev.prefill.coef,
        )
    dc = ev.decode_steps
    for name, method in ev.methods.items():
        column = f"rel_err__{name}"
        if column in dc:
            for source, part, sub in by_part(dc):
                add(
                    f"{name}: decode step",
                    name,
                    source,
                    part,
                    sub[column],
                    method.describe()["coef"],
                )
    runs = ev.runs
    for name, method in ev.methods.items():
        coef = method.describe()["coef"]
        sub_runs = runs[runs["method"] == name]
        diag_cols = [c for c in sub_runs.columns if c.startswith("diag__")]
        for source, part, sub in by_part(sub_runs):
            add(f"{name}: T_total per run", name, source, part, sub["rel_err"], coef)
            for column in diag_cols:
                if sub[column].notna().any():
                    add(
                        f"{name} (diagnostic: {column[6:]})",
                        name,
                        source,
                        part,
                        relative_errors(sub[column], sub["wall_seconds"]),
                        {},
                    )
    return rows
