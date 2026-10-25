"""Подбор и оценка всех частей на одном разбиении; результат — таблицы для отчёта.

Модель A подбирается на prefill-шагах train-прогонов (и отдельно на прогонах
без кэша для проверки «train — без кэша, test — с кэшем»). Каждый выбранный
способ decode подбирается на тех же train-прогонах и оценивается на тех же
test-прогонах, а затем без подгонки применяется к прогонам второй модели.
Ошибка везде относительная: ``(предсказание − замер) / замер``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import split as splits
from .data import Dataset
from .fit import relative_errors, summary
from .prefill import PrefillModel


@dataclass
class Evaluation:
    split: dict[str, list[str]]
    prefill: PrefillModel
    prefill_cache_split: PrefillModel
    methods: dict[str, object]
    prefill_steps: pd.DataFrame  # train data + transfer, with predictions
    decode_steps: pd.DataFrame
    runs: pd.DataFrame  # per (run, method) predictions
    metrics: list[dict] = field(default_factory=list)


def _tag(run_ids, split_map: dict[str, list[str]]) -> pd.Series:
    lookup = {r: part for part, ids in split_map.items() for r in ids}
    return pd.Series([lookup.get(r, "transfer") for r in run_ids])


def evaluate(
    train_data: Dataset,
    transfer: Dataset | None,
    methods: dict[str, type],
    fit_method: str,
    seed: int,
    fit_kwargs: dict,
) -> Evaluation:
    sp = splits.stratified(train_data.runs, seed)
    pf = train_data.prefill_steps
    prefill = PrefillModel().fit(
        pf[pf["run_id"].isin(sp["train"])], fit_method, **fit_kwargs
    )
    cache_sp = splits.cache_off_on(train_data.runs)
    prefill_cache = PrefillModel().fit(
        pf[pf["run_id"].isin(cache_sp["train"])], fit_method, **fit_kwargs
    )

    fitted = {
        name: cls().fit(train_data, sp["train"], prefill, fit_method, **fit_kwargs)
        for name, cls in methods.items()
    }

    datasets = [("train_model", train_data)] + (
        [("transfer", transfer)] if transfer else []
    )
    pf_rows, dc_rows, run_rows = [], [], []
    for source, data in datasets:
        steps = data.prefill_steps.copy()
        steps["source"] = source
        steps["part"] = (
            _tag(steps["run_id"], sp).to_numpy()
            if source == "train_model"
            else "transfer"
        )
        steps["cache_part"] = (
            _tag(steps["run_id"], cache_sp).to_numpy()
            if source == "train_model"
            else "transfer"
        )
        steps["pred"] = prefill.predict_steps(steps)
        steps["pred_cache_split"] = prefill_cache.predict_steps(steps)
        steps["rel_err"] = relative_errors(steps["pred"], steps["T"])
        steps["rel_err_cache_split"] = relative_errors(
            steps["pred_cache_split"], steps["T"]
        )
        pf_rows.append(steps)

        dec = data.decode_steps.copy()
        dec["source"] = source
        dec["part"] = (
            _tag(dec["run_id"], sp).to_numpy()
            if source == "train_model"
            else "transfer"
        )
        if "additive" in fitted:
            dec["pred"] = fitted["additive"].predict_steps(dec)
            dec["rel_err"] = relative_errors(dec["pred"], dec["T"])
        dc_rows.append(dec)

        runs = data.runs.set_index("run_id")
        ids = list(runs.index)
        parts = (
            _tag(ids, sp).to_numpy()
            if source == "train_model"
            else np.array(["transfer"] * len(ids))
        )
        t_prefill_hat = prefill.predict_runs(data.prefill_steps).reindex(ids)
        for name, method in fitted.items():
            pred = method.predict_total(data, ids).reindex(ids)
            flags = method.flags(data, ids).reindex(ids)
            frame = runs.assign(
                method=name,
                part=parts,
                source=source,
                pred=pred.to_numpy(),
                prefill_hat=t_prefill_hat.to_numpy(),
                flag=flags.to_numpy(),
            ).reset_index()
            frame["rel_err"] = relative_errors(frame["pred"], frame["wall_seconds"])
            if name == "multiplicative":
                diag = method.predict_total_measured_prefill(data, ids).reindex(ids)
                frame["pred_measured_prefill"] = diag.to_numpy()
                frame["rel_err_measured_prefill"] = relative_errors(
                    frame["pred_measured_prefill"], frame["wall_seconds"]
                )
            run_rows.append(frame)

    ev = Evaluation(
        sp,
        prefill,
        prefill_cache,
        fitted,
        pd.concat(pf_rows, ignore_index=True),
        pd.concat(dc_rows, ignore_index=True),
        pd.concat(run_rows, ignore_index=True),
    )
    ev.metrics = _metrics(ev)
    return ev


def _metrics(ev: Evaluation) -> list[dict]:
    rows = []

    def add(what, part, rel, coef):
        if len(rel):
            rows.append({"what": what, "part": part, **summary(rel), "coef": coef})

    pf = ev.prefill_steps
    for part in ("train", "test", "transfer"):
        add(
            "A: prefill step",
            part,
            pf.loc[pf["part"] == part, "rel_err"],
            ev.prefill.coef,
        )
    for part in ("train", "test"):
        add(
            "A: prefill step (train cache off, test cache on)",
            part,
            pf.loc[pf["cache_part"] == part, "rel_err_cache_split"],
            ev.prefill_cache_split.coef,
        )
    # A at run level: T̂_prefill against the sum of measured prefill steps.
    run_pf = pf.groupby(["run_id", "part"])[["pred", "T"]].sum().reset_index()
    for part in ("train", "test", "transfer"):
        sub = run_pf[run_pf["part"] == part]
        add(
            "A: prefill total per run",
            part,
            relative_errors(sub["pred"], sub["T"]),
            ev.prefill.coef,
        )
    dc = ev.decode_steps
    if "rel_err" in dc:
        for part in ("train", "test", "transfer"):
            add(
                "additive: decode step",
                part,
                dc.loc[dc["part"] == part, "rel_err"],
                ev.methods["additive"].coef,
            )
    runs = ev.runs
    for name, method in ev.methods.items():
        coef = method.describe()
        for part in ("train", "test", "transfer"):
            sub = runs[(runs["method"] == name) & (runs["part"] == part)]
            add(
                f"{name}: T_total per run",
                part,
                sub["rel_err"],
                coef.get(
                    "decode_step_coef", {"k_points": len(coef.get("k_points", []))}
                ),
            )
            if name == "multiplicative" and len(sub):
                add(
                    "multiplicative (diagnostic, measured prefill × k_meas)",
                    part,
                    sub["rel_err_measured_prefill"],
                    {},
                )
    return rows
