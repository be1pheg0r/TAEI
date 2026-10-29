"""Блок ``system`` отчёта: модель «от N» и её проверка на S5 и S8.

Модель шага — модель A ядра (prefill) и три варианта decode, подобранные на
S2 + S6, κ — на S7 (``decode_variants``); вариант выбирается по ошибке шага на
train и test S3. Прогноз (``system``) берёт только пул запросов из плана, N,
кэш, планировщик и параметры движка. Замер (``closed_loop``) — S5 фаз C и D и
S8 W1–W4 фазы D; по ним ничего не подбирается. Ошибка e = (прогноз − замер) /
замер — по точкам, отдельно ниже и выше колена KV по факту (для fixed_batch
колено — волна). Плюс ошибки колена, потолка RPS и очереди в TTFT, экономика
по прогнозу и замеру и подписи к графикам, посчитанные из чисел.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import closed_loop, economics
from . import data as nano_data
from . import decode_variants as dv
from . import system as sm
from .evaluate import Evaluation

CACHE_LABEL = {"on": "вкл", "off": "выкл"}
SCHED_LABEL = {"upstream_dynamic": "dynamic", "fixed_batch": "fixed_batch"}
LOAD_LABEL = {
    "S5": "S5 — смесь S4 (L 513–4096, O 16–256, 4 семьи)",
    "W1": "W1 — L 2048, O 128, префикс 1024",
    "W2": "W2 — L 256–1024, O 256–1024 (длинные ответы)",
    "W3": "W3 — L 4096–12288, O 16–64 (длинные промпты)",
    "W4": "W4 — смесь S5 с R ≈ 0,8",
}
ROLES = [
    {
        "series": "S1",
        "phase": "A",
        "role": "train шага",
        "what": "prefill: L, R, k запросов в шаге (decode нет)",
    },
    {
        "series": "S2",
        "phase": "A",
        "role": "train шага",
        "what": "decode: B одинаковых запросов разом, Σℓ до ~37k",
    },
    {
        "series": "S6",
        "phase": "D",
        "role": "train шага",
        "what": "decode: Σℓ до полного KV (до 45k), равные и смешанные длины",
    },
    {
        "series": "S7",
        "phase": "D",
        "role": "train шага (κ)",
        "what": "decode с общим префиксом: B × ℓ × R × семей",
    },
    {
        "series": "S3",
        "phase": "B",
        "role": "test шага",
        "what": "смеси по доле prefill, закрытые батчи",
    },
    {
        "series": "S5",
        "phase": "C + D",
        "role": "проверка системы",
        "what": "закрытый цикл на смеси S4, N = 1…64",
    },
    {
        "series": "S8",
        "phase": "D",
        "role": "проверка системы",
        "what": "закрытый цикл W1–W4, N до 96",
    },
    {
        "series": "S4",
        "phase": "B",
        "role": "приложение",
        "what": "открытый поток (вне постановки «от N»)",
    },
    {
        "series": "S0",
        "phase": "A",
        "role": "приложение",
        "what": "серверная матрица на этой GPU",
    },
    {
        "series": "REF",
        "phase": "A–D",
        "role": "контроль",
        "what": "реперный прогон: шум и дрейф машины",
    },
]
PHASE_D_ISSUES = [
    "Кончилось место на /home в середине S8-W1 (попытка S8 весит 70–220 МБ, место "
    "заранее не проверялось). "
    "Данных не потеряно: недописанная попытка переснята, остальное дописано на "
    "корневой раздел и после сбора перенесено в phase_D/runs.",
    "S7: R = 0,9 при ℓ = 1025 даёт фактически R = 0,999 — префикс берётся целыми "
    "блоками (4 из 4 полных). "
    "κ и U считаются по фактическому C, поэтому это учтено.",
    "W3 с кэшем: колено по сетке было 5,3, по факту 6,1. Сетка вычитала общие блоки "
    "по самому длинному префиксу семьи "
    "во всём пуле, поэтому выше колена вышла одна точка (N = 8) вместо двух.",
    "Разброс REF в W2 до +2,1% (в S6 и S7 ±0,4%): тренда нет, частота 2100 МГц, порог "
    "3% не превышен.",
    "Не влезли по KV и не снимались: смешанные длины на 44k (заменены наибольшей "
    "суммой 43,8k / 42,4k / 39,7k), "
    "23 точки S7 из 150, в том числе 5 из 15 пар контроля R = 0.",
]
LIMITS = [
    "Модель берёт извне: пул запросов (распределения L, O и префиксов из плана), "
    "параметры движка и коэффициенты "
    "шага, подобранные на сетках S1, S2, S6, S7. Время вне шагов (промежутки между "
    "шагами, ~0,4%) не моделируется.",
    "Основная область — N не выше колена KV. Выше колена движок вытесняет запросы и "
    "пересчитывает их prefill, "
    "модель этого не считает. Чаще всего — в W2 (порядка одного вытеснения на "
    "завершение); ошибка выше колена "
    "показана отдельно.",
    "Очередь в TTFT — среднее по закону Литтла, (N − B)/RPS, одинаковое для всех "
    "запросов. Разброс ожидания в очереди "
    "не моделируется, поэтому выше колена p90 TTFT занижен.",
    "Когорты (все L и O пула одинаковые, W1): отдельная ветка формул до колена. Выше "
    "колена когорты распадаются "
    "на пачки, там общая формула.",
    "Малый B_KV: когда на колене всего несколько запросов, каждый до четверти KV (W3 "
    "с кэшем), упор чаще "
    "случается на длинных запросах, и средние по пулу завышают колено (~+10%).",
    "Короткий замер: у W2 при N = 2 и 4 окно в 3 мин даёт 40–80 завершений, p90 там "
    "шумный. У W3 и W4 по 2 повтора.",
    "κ = const — гипотеза: при 2 запросах на семью κ ≈ 0, при 4 и больше 0,4–0,6, "
    "разброс по точкам шире шума.",
]
ERR_METRICS = ("rps", "itps_new", "otps", "ttft_p50", "ttft_p90", "tpot")
PRED_KEYS = (
    "N", "B", "B_kv", "above", "branch", "rps", "itps_new", "itps_full", "itps_cache",
    "otps", "all_tokens", "ttft_p50", "ttft_p90", "tpot", "ttps", "queue", "T_dc",
    "T_pf", "a_tilde", "a_tilde_full", "b_tilde",
)  # fmt: skip


SHIP_KEYS = (
    "N", "rps", "itps_new", "itps_full", "otps", "ttft_p50", "ttft_p90", "tpot",
    "ttps", "a_tilde", "a_tilde_full", "b_tilde",
)  # fmt: skip


def curve_grid(top: int) -> list[int]:
    """N of the prediction curve: every N to 8, then steps of 2, 4, 8 and 16."""
    out = list(range(1, min(top, 8) + 1))
    for lo, hi, step in ((8, 16, 2), (16, 32, 4), (32, 64, 8), (64, 10**6, 16)):
        out += list(range(lo + step, min(top, hi) + 1, step))
    return out


def plan_points(plan: dict[str, dict]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Plan point of every run (ℓ, R target, families) and its decode warm-up."""
    rows, warmup = [], {}
    for run_id, row in plan.items():
        point = (row.get("workload_spec") or {}).get("point") or {}
        warmup[run_id] = int(point.get("warmup_steps", 0))
        rows.append(
            {
                "run_id": run_id,
                "ell": point.get("L"),
                "R_target": point.get("R_target", point.get("R", 0.0)),
                "families": point.get("families", 0),
            }
        )
    return pd.DataFrame(rows), warmup


def _mean(rows: list[dict]) -> dict:
    out = {}
    for k in PRED_KEYS:
        vals = [r[k] for r in rows]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = float(np.mean(vals))
        else:
            out[k] = vals[0]
    out["above"] = any(r["above"] for r in rows)
    return out


def _err(pred: float | None, meas: float | None) -> float | None:
    if pred is None or meas is None or not meas:
        return None
    return pred / meas - 1


def _pct(x: float) -> str:
    return f"{100 * x:+.0f}%"


def _caption(w: dict, choice: str) -> dict[str, str]:
    """1–2 lines per chart from the numbers of this workload."""
    pts, curve = w["points"], w["curve"][choice]
    below = [p for p in pts if p["region"] == "below"]
    mae = lambda rs, k: (  # noqa: E731
        float(
            np.mean(
                [abs(p["e"][choice][k]) for p in rs if p["e"][choice][k] is not None]
            )
        )
        if rs
        else None
    )
    first, top = curve[0], max(curve, key=lambda r: r["rps"])
    meas_top = max(pts, key=lambda p: p["rps"])
    knee = w["knee_pred"]
    fact = w["knee_fact"]
    knee_text = (
        f"колено KV по модели B_KV = {knee:.1f}"
        + (f", по замеру {fact:.1f}" if fact else "")
        if w["scheduler"] != "fixed_batch"
        else f"волна {knee:.0f}"
    )
    rps_err = mae(below, "rps")
    ratio = w["pool"]["O_mean"] / w["pool"]["new_mean"]
    at_knee = [r for r in curve if r["N"] <= max(knee, 1.0)][-1]
    out = {
        "metrics": (
            f"RPS растёт до N ≈ {knee:.0f} ({knee_text}), потолок по модели "
            f"{top['rps']:.2f} зап/с, по замеру {meas_top['rps']:.2f}."
            + (
                f" Ниже колена средняя |e| RPS {100 * rps_err:.0f}%."
                if rps_err is not None
                else ""
            )
        ),
        "line": (
            "Чем дальше прямая от нуля, тем больше GPU может отдать при этом N. "
            "Прямая растёт до колена: 1/b̃ (только выход) от "
            f"{1 / first['b_tilde']:.0f} при N = 1 до {1 / top['b_tilde']:.0f} "
            f"ток/с; точка модели — на луче Ō/Ī = {ratio:.2f}."
        ),
        "throughput": (
            "Правее — быстрее одному пользователю, выше — больше токенов у GPU. "
            f"От N = 1 до потолка OTPS растёт в {top['otps'] / first['otps']:.1f} "
            f"раза, а TTPS падает в {first['ttps'] / top['ttps']:.1f} раза; выше "
            "колена OTPS не растёт, и N добавляет только очередь."
        ),
        "ttft": (
            "Вверх-вправо — лучше. До колена TTFT p50 по модели "
            f"{first['ttft_p50']:.2f}–{at_knee['ttft_p50']:.2f} с, выше колена "
            f"растёт на очереди: {curve[-1]['ttft_p50']:.1f} с при "
            f"N = {curve[-1]['N']:.0f}."
        ),
    }
    return out


def build(ev: Evaluation, plan: dict[str, dict], cfg: dict, econ: dict) -> dict:
    engine = sm.Engine(**cfg["engine"])
    points, warmup = plan_points(plan)
    steps_data = nano_data.load(
        cfg["step_roots"], "decode variants", None, list(cfg["step_series"])
    )
    fit = dv.fit_all(steps_data.decode_steps, points, warmup)
    steps = {k: sm.StepModel(ev.prefill.coef, v) for k, v in fit.variants.items()}

    frame = closed_loop.read(cfg["check_roots"])
    fact = closed_loop.knees(frame)
    measured = closed_loop.points(frame)
    workloads = []
    for (load, sched, cache), g in frame.groupby(["load", "scheduler", "cache"]):
        pools = {}
        for _, r in g.iterrows():  # D extends the S5 pools: keep the longest
            if r["seed"] not in pools or len(r["pool"]) > len(pools[r["seed"]]):
                pools[r["seed"]] = r["pool"]
        pools = {
            s: sm.Pool.from_spec(p, cache, engine.block_size) for s, p in pools.items()
        }
        runs = {
            s: sm.running(p, engine, int(cfg["draws"]), int(cfg["seed"]))
            for s, p in pools.items()
        }
        knee_pred = (
            float(np.mean([sm.kv_knee(r, engine) for r in runs.values()]))
            if sched != "fixed_batch"
            else float(engine.wave)
        )
        n_meas = sorted(g["N"].unique())
        top_n = min(int(cfg["curve_max_n"]), int(np.ceil(1.25 * max(n_meas))))
        grid = sorted({n for n in curve_grid(top_n)} | set(n_meas))

        def at(N, step, pools=pools, runs=runs, sched=sched):
            if sched == "fixed_batch":
                rows = [
                    sm.predict_fixed(
                        p, engine, step, int(N), int(cfg["warmup"]), int(cfg["measure"])
                    )
                    for p in pools.values()
                ]
            else:
                rows = [
                    sm.predict_dynamic(p, engine, step, N, runs[s])
                    for s, p in pools.items()
                ]
            return _mean(rows)

        curve = {name: [at(N, step) for N in grid] for name, step in steps.items()}
        by_n = {name: {int(r["N"]): r for r in rows} for name, rows in curve.items()}
        fact_knee = fact.get((load, sched, cache))
        bound = float(engine.wave) if sched == "fixed_batch" else fact_knee
        pts = []
        sub = measured[
            (measured["load"] == load)
            & (measured["scheduler"] == sched)
            & (measured["cache"] == cache)
        ]
        for _, m in sub.sort_values(["N", "phase"]).iterrows():
            N = int(m["N"])
            meas = {k: float(m[k]) for k in (*closed_loop.METRICS, "queue")}
            meas["ttps"] = 1 / meas["tpot"]
            meas["all_tokens"] = meas["itps_full"] + meas["otps"]
            meas["itps_cache"] = meas["itps_full"] - meas["itps_new"]
            pred = {name: by_n[name][N] for name in steps}
            pts.append(
                {
                    "N": N,
                    "phase": m["phase"],
                    "seeds": int(m["seeds"]),
                    "completed_min": int(m["completed_min"]),
                    "region": "below" if bound is None or bound >= N else "above",
                    **meas,
                    "pred": pred,
                    "e": {
                        name: {k: _err(p[k], meas[k]) for k in ERR_METRICS}
                        for name, p in pred.items()
                    },
                }
            )
        pool0 = next(iter(pools.values()))
        w = {
            "key": f"{load}|{sched}|{cache}",
            "load": load,
            "scheduler": sched,
            "cache": cache,
            "label": f"{load} · {SCHED_LABEL[sched]} · кэш {CACHE_LABEL[cache]}",
            "load_label": LOAD_LABEL.get(load, load),
            "pool": {
                "requests": len(pool0.L),
                "L_mean": float(np.mean([p.L.mean() for p in pools.values()])),
                "O_mean": float(np.mean([p.out.mean() for p in pools.values()])),
                "new_mean": float(np.mean([p.new.mean() for p in pools.values()])),
                "R": float(np.mean([p.C.sum() / p.L.sum() for p in pools.values()])),
                "uniform": bool(pool0.uniform),
                "seeds": len(pools),
            },
            "knee_pred": knee_pred,
            "knee_fact": fact_knee,
            "curve": curve,
            "points": pts,
        }
        w["caption"] = _caption(w, fit.choice)
        workloads.append(w)

    out = {
        "roles": ROLES,
        "phase_d_issues": PHASE_D_ISSUES,
        "limits": LIMITS,
        "engine": cfg["engine"],
        "variants": {
            k: {"label": dv.LABELS[k], "latex": dv.LATEX[k], "coef": v.coef}
            for k, v in fit.variants.items()
        },
        "choice": fit.choice,
        "core": {**ev.prefill.coef, **ev.decode.coef},
        "step_errors": fit.errors,
        "kappa": {
            "value": fit.variants["kappa"].coef["kappa"],
            "points": len(fit.kappa_rows),
            "attempts_shared": len(fit.kappa_all),
            "min_shared": dv.MIN_SHARED,
            "floor_margin": dv.FLOOR_MARGIN,
            "table": dv.kappa_table(fit.kappa_rows),
        },
        "workloads": workloads,
        "summary": _summary(workloads, list(steps)),
        "special": _special(workloads, list(steps)),
        "economics": _economics(workloads, fit.choice, econ),
    }
    from . import presentation  # imports curve_grid from this module

    out["presentation"] = presentation.build(out)
    _slim(workloads, fit.choice)  # after the tables above, which use every field
    return out


def _slim(workloads: list[dict], choice: str) -> None:
    """Keep only what the charts read: curves of every variant (dotted lines),
    point predictions and errors of the chosen one (tables)."""
    for w in workloads:
        w["curve"] = {
            v: [{k: _round(r[k]) for k in SHIP_KEYS} for r in rows]
            for v, rows in w["curve"].items()
        }
        for p in w["points"]:
            p["pred"] = {choice: {k: _round(p["pred"][choice][k]) for k in SHIP_KEYS}}
            p["e"] = {choice: p["e"][choice]}


def _round(x):
    return float(f"{x:.4g}") if isinstance(x, float) else x


def _summary(workloads: list[dict], variants: list[str]) -> list[dict]:
    """Mean |e| and mean signed e per (variant, load, region) and metric."""
    out = []
    for name in variants:
        for load in sorted({w["load"] for w in workloads}) + ["все"]:
            for region in ("below", "above"):
                pts = [
                    p
                    for w in workloads
                    if load in ("все", w["load"])
                    for p in w["points"]
                    if p["region"] == region
                ]
                if not pts:
                    continue
                row = {
                    "variant": name,
                    "load": load,
                    "region": region,
                    "points": len(pts),
                }
                for k in ERR_METRICS:
                    e = [p["e"][name][k] for p in pts if p["e"][name][k] is not None]
                    row[f"{k}_abs"] = float(np.mean(np.abs(e))) if e else None
                    row[f"{k}_signed"] = float(np.mean(e)) if e else None
                out.append(row)
    return out


def _special(workloads: list[dict], variants: list[str]) -> dict:
    knees, ceilings, queues = [], [], []
    for w in workloads:
        if w["scheduler"] != "fixed_batch" and w["knee_fact"]:
            knees.append(
                {
                    "label": w["label"],
                    "pred": w["knee_pred"],
                    "fact": w["knee_fact"],
                    "e": w["knee_pred"] / w["knee_fact"] - 1,
                }
            )
        top = max(w["points"], key=lambda p: p["N"])
        ceilings.append(
            {
                "label": w["label"],
                "N": top["N"],
                "meas": top["rps"],
                **{f"pred_{v}": top["pred"][v]["rps"] for v in variants},
                **{f"e_{v}": top["e"][v]["rps"] for v in variants},
            }
        )
        for p in w["points"]:
            if p["region"] == "above" and w["scheduler"] != "fixed_batch":
                queues.append(
                    {
                        "label": w["label"],
                        "N": p["N"],
                        "meas": p["queue"],
                        **{f"pred_{v}": p["pred"][v]["queue"] for v in variants},
                        **{
                            f"e_{v}": _err(p["pred"][v]["queue"], p["queue"])
                            for v in variants
                        },
                    }
                )
    return {"knee": knees, "ceiling": ceilings, "queue": queues}


def _economics(workloads: list[dict], choice: str, econ: dict) -> dict:
    """Steps of the economics per dynamic workload and SLO, by the prediction:
    N — the largest meeting the SLO, τ = 1/RPS there, then v, Q, Rev, profit."""
    prices = economics.scaled_prices(econ["prices"]["base"], econ["prices"]["scale"])
    gpu, u = econ["gpu"], float(econ.get("utilization", 1.0))
    rows = []
    for w in workloads:
        if w["scheduler"] != "upstream_dynamic":
            continue
        pool = w["pool"]
        tokens = {
            "new": pool["new_mean"],
            "cached": pool["L_mean"] - pool["new_mean"],
            "out": pool["O_mean"],
        }
        for slo in econ["slo"]:
            p = economics.max_n_under_slo(w["curve"][choice], slo)
            row = {"load": w["load"], "cache": w["cache"], "label": w["label"]}
            row |= {"slo": slo["label"], "N": p["N"] if p else None}
            if p:
                row |= economics.hour(prices, gpu, tokens, 1 / p["rps"], u)
            rows.append(row)
    return {
        "prices": prices,
        "base": econ["prices"]["base"],
        "scale": econ["prices"]["scale"],
        "gpu": gpu,
        "electricity_per_hour": economics.electricity_per_hour(gpu),
        "utilization": u,
        "slo": econ["slo"],
        "rows": rows,
    }
