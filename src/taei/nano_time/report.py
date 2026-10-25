"""Один HTML-файл с таблицами и графиками (PNG matplotlib, встроены base64).

Разделы: 1 — метрики и коэффициенты; 2 — additive против multiplicative на
одних test-прогонах; 3 — кривые k(p_decode); 4 — prefill (модель A); 5 — decode
(additive); 6 — ошибка от частоты SM; 7 — перенос на вторую модель; 8 —
покрытие диапазонов данных; в конце — что посчитать нельзя.
"""

from __future__ import annotations

import base64
import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .evaluate import Evaluation  # noqa: E402

COLORS = {
    "train": "#4C72B0",
    "test": "#DD8452",
    "transfer": "#55A868",
    "additive": "#4C72B0",
    "multiplicative": "#C44E52",
}


def paired_comparison(test: pd.DataFrame) -> dict | None:
    """Additive vs multiplicative |error| on the same test runs (Wilcoxon)."""
    from scipy import stats

    piv = test.pivot_table(index="run_id", columns="method", values="rel_err").abs()
    if not {"additive", "multiplicative"} <= set(piv.columns):
        return None
    add, mul = piv["additive"], piv["multiplicative"]
    return {
        "n": int(len(piv)),
        "additive_better": int((add < mul).sum()),
        "mean_diff": float((mul - add).mean()),
        "wilcoxon_p": float(stats.wilcoxon(add, mul).pvalue),
    }


def _img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return (
        f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
    )


def _table(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    return df.to_html(
        index=False,
        float_format=lambda v: floatfmt.format(v),
        border=0,
        classes="t",
        escape=True,
    )


def _binned(ax, x, y, bins, label, color):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    centers, med, lo, hi = [], [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() == 0:
            continue
        centers.append(np.median(x[sel]))
        med.append(np.median(y[sel]))
        lo.append(np.percentile(y[sel], 25))
        hi.append(np.percentile(y[sel], 75))
    ax.plot(centers, med, "o-", color=color, label=label)
    ax.fill_between(centers, lo, hi, color=color, alpha=0.2)


def _pred_vs_meas(ax, frame, x, y, color_by, title, unit="s"):
    for key, sub in frame.groupby(color_by):
        ax.scatter(
            sub[x], sub[y], s=10, alpha=0.6, label=str(key), color=COLORS.get(key)
        )
    lim = [min(frame[x].min(), frame[y].min()), max(frame[x].max(), frame[y].max())]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xlabel(f"measured, {unit}")
    ax.set_ylabel(f"predicted, {unit}")
    ax.set_title(title)
    ax.legend(fontsize=8)


def render(ev: Evaluation, meta: dict, notes: list[str]) -> str:
    parts: list[str] = []
    h = parts.append
    runs = ev.runs
    test = runs[runs["part"] == "test"]

    # 1. Metrics table.
    metrics = pd.DataFrame(ev.metrics)
    metrics["coef"] = metrics["coef"].map(
        lambda c: ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in c.items()
        )
    )
    metrics = metrics[
        [
            "what",
            "part",
            "n",
            "mean_abs",
            "median_abs",
            "max_abs",
            "mean_signed",
            "coef",
        ]
    ]
    h(
        "<h2>1. Ошибки и коэффициенты</h2>"
        "<p>Ошибка — относительная, (предсказание − замер)/замер; mean/median/max — "
        "по модулю, "
        "mean_signed — со знаком (минус = недобор). train/test — прогоны Qwen3-0.6B "
        "по одному "
        "разбиению; transfer — вторая модель без подгонки. Ошибка multiplicative на "
        "train "
        "занижена по построению: train-прогоны сами задают узлы k(p); остаётся только "
        "разброс прогонов с одинаковым p_decode.</p>"
    )
    h(_table(metrics))

    # 2. Additive vs multiplicative on the same test runs.
    h("<h2>2. Сложение против умножения на одних test-прогонах</h2>")
    if test["method"].nunique() >= 1:
        piv = test.pivot_table(index="run_id", columns="method", values="rel_err")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        piv.abs().plot.bar(ax=axes[0], color=[COLORS.get(c) for c in piv.columns])
        axes[0].set_ylabel("|relative error|")
        axes[0].set_title("test runs: |error| per run")
        for name, sub in test.groupby("method"):
            axes[1].scatter(
                sub["wall_seconds"],
                sub["pred"],
                label=name,
                color=COLORS.get(name),
                s=25,
            )
        lim = [
            test[["wall_seconds", "pred"]].min().min(),
            test[["wall_seconds", "pred"]].max().max(),
        ]
        axes[1].plot(lim, lim, "k--", lw=1, label="y = x")
        axes[1].set_xlabel("wall_seconds")
        axes[1].set_ylabel("T̂_total")
        axes[1].legend()
        axes[1].set_title("test runs: prediction vs wall_seconds")
        h(_img(fig))
        by = (
            test.assign(abs_err=test["rel_err"].abs())
            .groupby(["method", "scheduler", "cache"])["abs_err"]
            .agg(["mean", "median", "max", "count"])
            .reset_index()
        )
        h("<p>Разрез по планировщику и кэшу (|ошибка| на test):</p>" + _table(by))
        paired = paired_comparison(test)
        if paired:
            h(
                "<p>Парное сравнение на одних test-прогонах: "
                f"additive точнее в {paired['additive_better']} из {paired['n']}, "
                "средняя разница |ошибок| (multiplicative − additive) = "
                f"{paired['mean_diff']:+.3f}, "
                f"Wilcoxon p = {paired['wilcoxon_p']:.3f}.</p>"
            )
        flagged = test[test["flag"].astype(str) != ""]
        if len(flagged):
            h(
                "<p>Помеченные прогоны (p_decode вне диапазона train-точек k):</p>"
                + _table(flagged[["run_id", "method", "p_decode", "flag"]])
            )

    # 3. k(p) curves.
    if "multiplicative" in ev.methods:
        m = ev.methods["multiplicative"]
        h(
            "<h2>3. Кривая k(p_decode)</h2><p>Синяя — k по модели A "
            "(wall_seconds / T̂_prefill), оранжевая — диагностическая k по измеренному "
            "prefill (wall_seconds / Σ измеренных prefill-шагов). Кресты — "
            "test-прогоны "
            "поверх кривых.</p>"
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        grid = np.linspace(
            min(runs["p_decode"].min(), m.points["p"].min()),
            max(runs["p_decode"].max(), m.points["p"].max()),
            300,
        )
        ax.plot(
            m.points["p"],
            m.points["k"],
            "o",
            color="#4C72B0",
            label="train points, model A",
        )
        ax.plot(grid, m.k(grid), "-", color="#4C72B0", alpha=0.6)
        ax.plot(
            m.points_measured["p"],
            m.points_measured["k"],
            "s",
            color="#DD8452",
            label="train points, measured prefill",
        )
        ax.plot(
            grid,
            np.interp(grid, m.points_measured["p"], m.points_measured["k"]),
            "-",
            color="#DD8452",
            alpha=0.6,
        )
        mt = test[test["method"] == "multiplicative"]
        ax.plot(
            mt["p_decode"],
            mt["wall_seconds"] / mt["prefill_hat"],
            "x",
            color="#4C72B0",
            ms=9,
            label="test runs, model A",
        )
        ax.plot(
            mt["p_decode"],
            mt["wall_seconds"] / mt["prefill_seconds"],
            "x",
            color="#DD8452",
            ms=9,
            label="test runs, measured prefill",
        )
        ax.set_xlabel("p_decode = O / (P + O)")
        ax.set_ylabel("k")
        ax.legend(fontsize=8)
        h(_img(fig))

    # 4. Prefill.
    pf = ev.prefill_steps[ev.prefill_steps["source"] == "train_model"]
    reqs = meta["prefill_requests"]
    step_l = (
        reqs.groupby(["run_id", "step_id"])
        .agg(L_mean=("L", "mean"), L_sum=("L", "sum"), C_sum=("C", "sum"))
        .reset_index()
    )
    step_l["R_step"] = step_l["C_sum"] / step_l["L_sum"]
    pf = pf.merge(step_l, on=["run_id", "step_id"], how="left")
    h(
        "<h2>4. Prefill (модель A)</h2><p>По шагам. R шага = ΣC/ΣL по запросам шага, "
        "L — среднее по запросам шага.</p>"
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    _pred_vs_meas(
        axes[0],
        pf.assign(predicted=pf["pred"]),
        "T",
        "predicted",
        "part",
        "prefill step: predicted vs measured",
    )
    for part, sub in pf.groupby("part"):
        axes[1].scatter(
            sub["L_mean"], sub["rel_err"], s=12, color=COLORS[part], label=part
        )
        axes[2].scatter(
            sub["R_step"], sub["rel_err"], s=12, color=COLORS[part], label=part
        )
    for ax, xl in ((axes[1], "mean L of the step"), (axes[2], "R of the step (ΣC/ΣL)")):
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel(xl)
        ax.set_ylabel("relative error")
        ax.legend(fontsize=8)
    h(_img(fig))

    # 5. Decode (additive).
    dc = ev.decode_steps[ev.decode_steps["source"] == "train_model"]
    if "pred" in dc:
        h(
            "<h2>5. Decode (additive)</h2><p>По шагам; для наглядности на левом "
            "графике "
            "случайная выборка 4000 шагов. Справа — медиана и межквартильный размах "
            "ошибки "
            "по квантильным корзинам.</p>"
        )
        sample = dc.sample(min(4000, len(dc)), random_state=0)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        _pred_vs_meas(
            axes[0],
            sample.assign(predicted=sample["pred"]),
            "T",
            "predicted",
            "part",
            "decode step: predicted vs measured",
        )
        for part, sub in dc.groupby("part"):
            _binned(axes[1], sub["sum_l"], sub["rel_err"], 12, part, COLORS[part])
        for part, sub in dc.groupby("part"):
            b = sub.groupby("B")["rel_err"].median()
            axes[2].plot(b.index, b.values, "o-", color=COLORS[part], label=part)
        for ax, xl in ((axes[1], "Σℓ"), (axes[2], "B")):
            ax.axhline(0, color="k", lw=1)
            ax.set_xlabel(xl)
            ax.set_ylabel("relative error (median)")
            ax.legend(fontsize=8)
        h(_img(fig))
        by_sched = (
            dc.merge(meta["runs"][["run_id", "scheduler"]], on="run_id")
            .groupby(["scheduler", "part"])["rel_err"]
            .agg(mean_signed="mean", median="median")
            .reset_index()
        )
        h("<p>Ошибка decode-шагов по планировщику:</p>" + _table(by_sched))

    # 6. Frequency.
    h(
        "<h2>6. Ошибка от частоты SM</h2><p>Частота шага — среднее по двум GPU "
        "прогона в "
        "ближайшей выборке телеметрии (шаг выборки ~1 с, поэтому для коротких "
        "decode-шагов это "
        "приближение). Модели частоту не используют.</p>"
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for part, sub in pf.groupby("part"):
        axes[0].scatter(
            sub["sm_clock_mhz"], sub["rel_err"], s=12, color=COLORS[part], label=part
        )
    axes[0].set_title("prefill steps (model A)")
    if "pred" in dc:
        for part, sub in dc.groupby("part"):
            _binned(
                axes[1], sub["sm_clock_mhz"], sub["rel_err"], 10, part, COLORS[part]
            )
        axes[1].set_title("decode steps (additive)")
    for name, sub in runs[runs["source"] == "train_model"].groupby("method"):
        axes[2].scatter(
            sub["sm_clock_mhz_median"],
            sub["rel_err"],
            s=20,
            color=COLORS.get(name),
            label=name,
        )
    axes[2].set_title("runs: T_total error vs median SM clock")
    for ax in axes:
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("SM clock, MHz")
        ax.set_ylabel("relative error")
        ax.legend(fontsize=8)
    h(_img(fig))

    # 7. Transfer.
    tr = runs[runs["source"] == "transfer"]
    label = html.escape(meta.get("transfer_label", "вторую модель"))
    h(f"<h2>7. Перенос на {label}</h2>")
    if len(tr):
        cols = [
            "run_id",
            "method",
            "wall_seconds",
            "pred",
            "rel_err",
            "prefill_seconds",
            "prefill_hat",
            "p_decode",
            "flag",
        ]
        h("<p>Коэффициенты и k — от Qwen3-0.6B, без подгонки.</p>" + _table(tr[cols]))
        tpf = ev.prefill_steps[ev.prefill_steps["source"] == "transfer"]
        tdc = ev.decode_steps[ev.decode_steps["source"] == "transfer"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        _pred_vs_meas(
            axes[0],
            tpf.assign(predicted=tpf["pred"]),
            "T",
            "predicted",
            "part",
            "transfer: prefill steps",
        )
        if "pred" in tdc:
            _pred_vs_meas(
                axes[1],
                tdc.assign(predicted=tdc["pred"]),
                "T",
                "predicted",
                "part",
                "transfer: decode steps (additive)",
            )
        h(_img(fig))
    else:
        h("<p>Данных для переноса нет.</p>")

    # 8. Coverage.
    h("<h2>8. Диапазоны данных</h2>")
    cov = []
    sources = [
        (
            "train model",
            meta["prefill_requests"],
            ev.decode_steps[ev.decode_steps["source"] == "train_model"],
            meta["runs"],
        )
    ]
    if meta.get("transfer_requests") is not None:
        sources.append(
            (
                "transfer",
                meta["transfer_requests"],
                ev.decode_steps[ev.decode_steps["source"] == "transfer"],
                meta["transfer_runs"],
            )
        )
    for label, rq, dsteps, rr in sources:
        for name, values in (
            ("L", rq["L"]),
            ("R", rq["R"]),
            ("B", dsteps["B"]),
            ("Σℓ", dsteps["sum_l"]),
            ("p_decode", rr["p_decode"]),
        ):
            v = np.asarray(values, float)
            cov.append(
                {
                    "data": label,
                    "variable": name,
                    "min": v.min(),
                    "p05": np.percentile(v, 5),
                    "median": np.median(v),
                    "p95": np.percentile(v, 95),
                    "max": v.max(),
                    "n": len(v),
                }
            )
    h(_table(pd.DataFrame(cov)))
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.6))
    for label, rq, dsteps, rr in sources:
        color = COLORS["transfer"] if label == "transfer" else COLORS["train"]
        for ax, (name, values) in zip(
            axes,
            (
                ("L", rq["L"]),
                ("R", rq["R"]),
                ("B", dsteps["B"]),
                ("Σℓ", dsteps["sum_l"]),
                ("p_decode", rr["p_decode"]),
            ),
            strict=True,
        ):
            ax.hist(
                np.asarray(values, float),
                bins=30,
                alpha=0.5,
                color=color,
                density=True,
                label=label,
            )
            ax.set_title(name)
    axes[0].legend(fontsize=8)
    h(_img(fig))

    h(
        "<h2>Что посчитать нельзя или с оговорками</h2><ul>"
        + "".join(f"<li>{html.escape(n)}</li>" for n in notes)
        + "</ul>"
    )

    head = (
        "<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Модель времени nano-vLLM</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:1400px;margin:24px auto;"
        "padding:0 16px;color:#222;background:#fff}"
        "table.t{border-collapse:collapse;font-size:13px;margin:8px 0}"
        "table.t td,table.t th{border-bottom:1px solid #ddd;padding:4px "
        "8px;text-align:right}"
        "table.t th{background:#f4f4f4}img{max-width:100%}h2{margin-top:36px}"
        "code{background:#f4f4f4;padding:1px 4px}</style></head><body>"
        "<h1>Модель времени nano-vLLM</h1>"
        f"<p>{html.escape(meta['summary'])}</p>"
    )
    return head + "\n".join(parts) + "</body></html>"
