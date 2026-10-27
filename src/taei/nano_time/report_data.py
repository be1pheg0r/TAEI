"""Данные отчёта в фиксированном формате (``time_model.json``, версия 2).

Документ — ``{schema_version, meta, models: [...]}``: по разделу на каждую
подобранную модель (свои коэффициенты, разбиение и наборы переноса). Отчёт
строится только из этого словаря, поэтому новая модель или набор переноса
появляются в отчёте без правки шаблона. Таблицы шагов урезаны для графиков
(decode — случайная выборка), а статистики по корзинам считаются по полным
данным. Все числа — обычные JSON-числа или ``null``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import prices
from .decode import LATEX as DECODE_LATEX
from .evaluate import TRAIN_SOURCE, Evaluation
from .fit import LATEX as FIT_LATEX
from .prefill import LATEX as PREFILL_LATEX
from .prefill_forms import analyze as analyze_prefill_forms

# Notation shared by every model in the report (inline TeX).
NOTATION = [
    [r"L", "токены запроса после prefill-шага"],
    [r"C", "токены запроса в кэше до шага"],
    [r"R = C / L", "доля токенов из кэша"],
    [r"\ell", "текущая длина контекста запроса в decode-шаге"],
    [r"B", "число запросов в decode-шаге"],
    [r"I", "входные токены за время T (все токены промпта, включая кэш)"],
    [r"O", "выходные токены за время T"],
    [
        r"\mathrm{ITPS} = I / T,\ \mathrm{OTPS} = O / T",
        "входные и выходные токены в секунду",
    ],
    [r"\tilde a,\ \tilde b", "цена входного и выходного токена, с/токен"],
    [r"T,\ \hat T", "измеренное и предсказанное время; у прогона T = wall_seconds"],
]

FORMULAS = [
    {
        "title": "Prefill-шаг (модель A)",
        "text": r"\(\sum\) — по запросам в шаге. Линейная часть — новые токены; "
        r"квадратичная — внимание новых токенов к кэшу и к себе (\(\tfrac12\) входит "
        r"в \(b\)).",
        "latex": PREFILL_LATEX,
    },
    {"title": "Decode-шаг", "text": "", "latex": DECODE_LATEX},
    {
        "title": "Время прогона",
        "text": "Сумма предсказаний по всем шагам трассы.",
        "latex": [r"T_{\mathrm{total}} = \sum T_{\mathrm{pf}} + \sum T_{\mathrm{dc}}"],
    },
    {
        "title": "Цены токенов",
        "text": r"Для нагрузки (набора запросов) — взвешенные по токенам, а не среднее "
        r"прямых. \(\sum L\) здесь — все токены промптов, то есть \(I\): кусок chunked "
        r"prefill и пересчёт не считаются дважды.",
        "latex": [
            r"\tilde a = \frac{T_{\mathrm{pf}}}{L} = a(1 - R) + b L (1 - R^2) + "
            r"\frac{c}{L}",
            r"\tilde b = \frac{T_{\mathrm{dc}}}{B} = \frac{d_0}{B} + d_1 \bar\ell "
            r"+ d_2",
            r"\tilde a = \frac{\sum T_{\mathrm{pf}}}{\sum L}, \qquad "
            r"\tilde b = \frac{\sum T_{\mathrm{dc}}}{\sum O}",
        ],
    },
    {
        "title": "Бюджет времени GPU",
        "text": r"\(= 1\) — GPU занят полностью, \(< 1\) — простой.",
        "latex": [
            r"\tilde a I + \tilde b O = T_{\mathrm{total}} \iff "
            r"\tilde a \cdot \mathrm{ITPS} + \tilde b \cdot \mathrm{OTPS} \le 1"
        ],
    },
    {
        "title": "Пользовательские метрики",
        "text": "TTFT — время до первого токена, TPOT — время на выходной токен, "
        "TTPS — токены в секунду у одного запроса.",
        "latex": [
            r"\mathrm{TTFT} = \text{ожидание} + \text{prefill}, \qquad "
            r"\mathrm{TTPS} \approx 1 / \mathrm{TPOT}"
        ],
    },
]

SCHEMA_VERSION = 2
DECODE_SAMPLE = 4000


def clean(value):
    """Recursively convert numpy/pandas values and NaN into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    return value


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    have = [c for c in columns if c in frame]
    return clean(frame[have].to_dict(orient="records"))


def binned(frame: pd.DataFrame, x: str, y: str, bins: int = 12) -> list[dict]:
    """Median and quartiles of y in quantile bins of x, per (source, part)."""
    out = []
    for (source, part), sub in frame.groupby(["source", "part"], sort=False):
        sub = sub[[x, y]].dropna()
        if part == "unused" or sub.empty:
            continue
        values = sub[x].to_numpy(float)
        if np.unique(values).size <= bins:
            groups = sub.groupby(x)
        else:
            edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
            labels = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
            groups = sub.groupby(labels)
        for _, g in groups:
            out.append(
                {
                    "source": source,
                    "part": part,
                    "x": float(g[x].median()),
                    "median": float(g[y].median()),
                    "q25": float(g[y].quantile(0.25)),
                    "q75": float(g[y].quantile(0.75)),
                    "n": int(len(g)),
                }
            )
    return out


def grid(frame: pd.DataFrame, x: str, y: str, z: str, nx: int = 6, ny: int = 5) -> dict:
    """Median z on a quantile grid of x and y (empty cells are null)."""
    sub = frame[[x, y, z]].dropna()
    if sub.empty:
        return {}

    def edges(values, n):
        e = np.unique(np.quantile(values, np.linspace(0, 1, n + 1)))
        return e if len(e) > 1 else np.array([values.min(), values.max() + 1e-9])

    ex, ey = edges(sub[x].to_numpy(float), nx), edges(sub[y].to_numpy(float), ny)
    ix = np.clip(np.digitize(sub[x], ex[1:-1]), 0, len(ex) - 2)
    iy = np.clip(np.digitize(sub[y], ey[1:-1]), 0, len(ey) - 2)
    median = np.full((len(ey) - 1, len(ex) - 1), np.nan)
    count = np.zeros_like(median, dtype=int)
    for j in range(len(ey) - 1):
        for i in range(len(ex) - 1):
            cell = sub[z].to_numpy()[(ix == i) & (iy == j)]
            count[j, i] = len(cell)
            if len(cell):
                median[j, i] = np.median(cell)
    centers = lambda e: ((e[:-1] + e[1:]) / 2).tolist()  # noqa: E731
    return {
        "x": centers(ex),
        "y": centers(ey),
        "x_edges": ex.tolist(),
        "y_edges": ey.tolist(),
        "median": median.tolist(),
        "count": count.tolist(),
    }


def _span(values) -> str:
    v = np.asarray(values, float)
    return f"{v.min():,.0f}–{v.max():,.0f}".replace(",", " ")


def pipeline(ev: Evaluation, labels: dict[str, str]) -> list[dict]:
    """Project pipeline steps with what the data of this model has and lacks."""
    have = {k: [] for k in ("prefill", "decode", "total", "load")}
    missing = {k: [] for k in have}
    for source, data in ev.datasets.items():
        name = labels.get(source, source)
        reqs, runs = data.prefill_requests, data.runs
        hits = reqs[reqs["C"] > 0]
        decode_one = (data.requests.groupby("run_id")["O_requested"].max() == 1).sum()
        text = f"{name}: L {_span(data.requests['prompt'])}"
        if len(hits):
            text += (
                f", R {hits['R'].min():.2f}–{hits['R'].max():.2f} у {len(hits)} строк"
            )
        else:
            text += ", R = 0"
        if decode_one:
            text += f"; {decode_one} прогонов с decode = 1"
        have["prefill"].append(text + ".")
        if (runs["cache"] == "on").sum() == 0:
            missing["prefill"].append(f"{name}: прогонов с кэшем нет.")
        lo, hi = data.requests["prompt"].min(), data.requests["prompt"].max()
        gaps = []
        if lo > 1024:
            gaps.append(f"1k–{lo / 1024:.0f}k")
        if hi < 32768:
            gaps.append(f"{hi / 1024:.0f}k–32k")
        if gaps:
            missing["prefill"].append(
                f"{name}: контексты {', '.join(gaps)} не мерились."
            )
        dc = data.decode_steps
        outputs = sorted(data.requests["O_requested"].unique())
        if len(dc):
            have["decode"].append(
                f"{name}: Σℓ {_span(dc['sum_l'])}, B {int(dc['B'].min())}–"
                f"{int(dc['B'].max())}, длины генерации "
                f"{', '.join(str(int(o)) for o in outputs)}."
            )
        have["total"].append(
            f"{name}: {len(runs)} прогонов"
            + (
                f", test {len(ev.split['test'])}"
                if source == TRAIN_SOURCE
                else " (перенос)"
            )
            + (
                f"; у {int((runs['lost_tokens'] > 0).sum())} прогонов сборщик "
                "потерял вытесненный запрос (шаги трассы верны)"
                if (runs["lost_tokens"] > 0).any()
                else ""
            )
            + "."
        )
        enqueue = data.requests.groupby("run_id")["enqueue_ns"]
        spread = (enqueue.max() - enqueue.min()).max() * 1e-9
        have["load"].append(
            f"{name}: все запросы прогона приходят за {spread * 1e3:.0f} мс."
        )
    missing["decode"].append(
        "Эксперимента «контекст фиксирован, длина генерации 1…256» нет: длина "
        "генерации не менялась при фиксированном контексте."
    )
    missing["load"].append(
        "Приход запросов во времени не варьировался, поэтому модель очереди "
        "проверить не на чем."
    )
    return [
        {
            "title": "Вопросы",
            "text": "Сколько токенов в секунду даёт сервер и как на это влияет кэш "
            "(\\(R\\)).",
            "have": [],
            "missing": [],
        },
        {
            "title": "Prefill",
            "text": "\\(T_{\\mathrm{pf}}(L, R)\\) — эксперимент «контекст 1k…32k, "
            "decode = 1»; проверка зависимости от \\(R\\) (reuse cache).",
            "have": have["prefill"],
            "missing": missing["prefill"],
        },
        {
            "title": "Decode",
            "text": "\\(T_{\\mathrm{dc}}\\) — эксперимент «контекст фиксирован, длина "
            "генерации 1…256»; зависимость от \\(\\ell\\) и \\(B\\).",
            "have": have["decode"],
            "missing": missing["decode"],
        },
        {
            "title": "Сборка",
            "text": "\\(T_{\\mathrm{total}} = \\sum T_{\\mathrm{pf}} + \\sum "
            "T_{\\mathrm{dc}}\\); сравнение с реальным временем через ошибку "
            "\\(e\\).",
            "have": have["total"],
            "missing": [],
        },
        {
            "title": "Цены токенов",
            "text": "\\(\\tilde a,\\ \\tilde b\\) → граница ITPS–OTPS.",
            "have": ["Считаются из коэффициентов модели и трассы, раздел 03."],
            "missing": [],
        },
        {
            "title": "Нагрузка",
            "text": "TTFT = ожидание + prefill, TTPS ≈ 1/TPOT → график TTPS–TTFT. "
            "Модель очереди (ожидание) — следующий этап (пункт 4*).",
            "have": have["load"],
            "missing": missing["load"],
        },
    ]


def build(ev: Evaluation, label: str, meta: dict, notes: list[str]) -> dict:
    """Assemble one model's section of the report document."""
    labels = {**meta.get("labels", {}), TRAIN_SOURCE: label}
    sources = [
        {
            "id": source,
            "label": labels.get(source, source),
            "role": "train" if source == TRAIN_SOURCE else "transfer",
            "runs": len(data.runs),
            "prefill_steps": len(data.prefill_steps),
            "decode_steps": len(data.decode_steps),
        }
        for source, data in ev.datasets.items()
    ]

    # Prefill steps with per-step L and R summaries.
    reqs = pd.concat(
        [d.prefill_requests.assign(source=s) for s, d in ev.datasets.items()],
        ignore_index=True,
    )
    step_lr = (
        reqs.groupby(["source", "run_id", "step_id"])
        .agg(L_mean=("L", "mean"), L_sum=("L", "sum"), C_sum=("C", "sum"))
        .reset_index()
    )
    step_lr["R_step"] = step_lr["C_sum"] / step_lr["L_sum"]
    pf = ev.prefill_steps.merge(step_lr, on=["source", "run_id", "step_id"], how="left")
    # R views use only steps with cached tokens: cache-off runs put every step at
    # R = 0 and would hide the dependence.
    cached = pf[pf["R_step"] > 0]
    prefill_binned = {
        "L_mean": binned(pf, "L_mean", "rel_err", bins=8),
        "R_step": binned(cached, "R_step", "rel_err", bins=6),
    }
    prefill_grid = grid(
        cached[cached["source"] == TRAIN_SOURCE], "L_mean", "R_step", "rel_err"
    )

    # Decode: a plotting sample per source, binned statistics on all steps.
    dc = ev.decode_steps
    sample = pd.concat(
        [
            g.sample(min(DECODE_SAMPLE, len(g)), random_state=0)
            for _, g in dc.groupby("source", sort=False)
        ]
    )
    decode_binned = {
        "sum_l": binned(dc, "sum_l", "rel_err"),
        "B": binned(dc, "B", "rel_err"),
    }

    runs = ev.runs.copy()
    runs["abs_err"] = runs["rel_err"].abs()
    test = runs[runs["part"] == "test"]
    breakdown = (
        test.groupby(["scheduler", "cache"])["rel_err"]
        .agg(
            mean=lambda e: e.abs().mean(),
            median=lambda e: e.abs().median(),
            max=lambda e: e.abs().max(),
            mean_signed="mean",
            count="count",
        )
        .reset_index()
    )

    requests = prices.requests(ev)
    coverage_values, coverage = {}, []
    for source, data in ev.datasets.items():
        dsample = sample[sample["source"] == source]
        coverage_values[source] = {
            "L": data.prefill_requests["L"].tolist(),
            "R": data.prefill_requests["R"].tolist(),
            "B": dsample["B"].tolist(),
            "Σℓ": dsample["sum_l"].tolist(),
            "O": data.requests["O"].tolist(),
        }
        full = {
            "L": data.prefill_requests["L"],
            "R": data.prefill_requests["R"],
            "B": data.decode_steps["B"],
            "Σℓ": data.decode_steps["sum_l"],
            "O": data.requests["O"],
        }
        for name, values in full.items():
            v = np.asarray(values, float)
            if not len(v):
                continue
            coverage.append(
                {
                    "source": source,
                    "variable": name,
                    "min": v.min(),
                    "p05": np.percentile(v, 5),
                    "median": np.median(v),
                    "p95": np.percentile(v, 95),
                    "max": v.max(),
                    "n": len(v),
                }
            )

    run_columns = [
        "run_id",
        "source",
        "part",
        "set_id",
        "seed",
        "condition_id",
        "cache",
        "scheduler",
        "wall_seconds",
        "pred",
        "rel_err",
        "prefill_seconds",
        "prefill_hat",
        "I",
        "O",
        "lost_tokens",
        "L_mean",
        "R_run",
    ]
    return clean(
        {
            "label": label,
            "sources": sources,
            "split": ev.split,
            "cache_split": ev.cache_split,
            "prefill_model": {
                "latex": PREFILL_LATEX,
                "coef": ev.prefill.coef,
                "cache_split_coef": ev.prefill_cache_split.coef
                if ev.prefill_cache_split
                else None,
            },
            "decode_model": {"latex": DECODE_LATEX, "coef": ev.decode.coef},
            "math": {
                "notation": NOTATION,
                "formulas": FORMULAS,
                "error": r"e = \frac{\hat T - T}{T}",
                "fit": FIT_LATEX.get(meta.get("fit"), ""),
                "fit_method": meta.get("fit"),
            },
            "pipeline": pipeline(ev, labels),
            "metrics": ev.metrics,
            "request_metrics": prices.request_errors(requests),
            "runs": _records(runs, run_columns),
            "test_breakdown": breakdown.to_dict(orient="records"),
            "prices": {
                "lines": prices.lines(ev),
                "runs": prices.run_prices(ev).to_dict(orient="records"),
            },
            "requests": _records(
                requests,
                [
                    "source",
                    "part",
                    "run_id",
                    "request_id",
                    "scheduler",
                    "cache",
                    "prompt",
                    "O",
                    "ttft",
                    "ttft_hat",
                    "tpot",
                    "tpot_hat",
                    "ttps",
                    "ttps_hat",
                ],
            ),
            "prefill_steps": _records(
                pf,
                [
                    "source",
                    "part",
                    "run_id",
                    "step_id",
                    "T",
                    "pred",
                    "rel_err",
                    "L_mean",
                    "R_step",
                    "new",
                    "quad",
                ],
            ),
            "decode_sample": _records(
                sample,
                ["source", "part", "run_id", "T", "B", "sum_l", "pred", "rel_err"],
            ),
            "decode_binned": decode_binned,
            "prefill_binned": prefill_binned,
            "prefill_grid": prefill_grid,
            "prefill_forms": analyze_prefill_forms(ev.datasets[TRAIN_SOURCE]),
            "coverage": coverage,
            "coverage_values": coverage_values,
            "notes": notes,
        }
    )


def document(meta: dict, models: list[dict]) -> dict:
    """The whole report: shared metadata and one section per fitted model."""
    return clean({"schema_version": SCHEMA_VERSION, "meta": meta, "models": models})
