"""1. RPS закрытого цикла: завершения в окне / окно против плановых 200 / окно.

Окно — от конца шага, где завершился последний запрос прогрева, до конца шага
последнего измеряемого. Проверка — закон Литтла: среднее число запросов в
системе за окно / средняя задержка запросов, завершившихся в окне. Точка —
(фаза, нагрузка, планировщик, кэш, N), среднее по сидам.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from common import attempts, jsonl, table, write

PER_SLOT = 8


def in_system_mean(enq, fin, lo, hi) -> float:
    t = np.concatenate([enq, fin])
    d = np.concatenate([np.ones(len(enq)), -np.ones(len(fin))])
    order = np.lexsort((d, t))
    t, count = t[order], np.cumsum(d[order])
    edges = np.clip(np.concatenate([t, [hi]]), lo, hi)
    return float((np.diff(edges) * count).sum() / (hi - lo))


def run_row(run, attempt) -> dict:
    loop = run["closed_loop"]
    lo, hi = loop["window_start_ns"], loop["window_end_ns"]
    enq, fin = [], []
    for r in jsonl(attempt / "requests.jsonl"):
        enq.append(r["enqueue_monotonic_ns"])
        fin.append(r["finish_monotonic_ns"])
    enq, fin = np.array(enq, float), np.array(fin, float)
    inside = (fin > lo) & (fin <= hi)
    T = (hi - lo) * 1e-9
    return {
        "old": loop["measure"] / T,
        "new": inside.sum() / T,
        "little": in_system_mean(enq, fin, lo, hi)
        / ((fin - enq)[inside].mean() * 1e-9),
        "measure": loop["measure"],
        "completed": int(inside.sum()),
    }


def main() -> None:
    groups = defaultdict(list)
    sources = [("C", "C-S5-*"), ("D", "D-S5-*"), ("D", "D-S8-*")]
    for phase, pattern in sources:
        for run, attempt in attempts(phase, pattern):
            load = run.get("block", run["series"]).replace("S8-", "")
            key = (
                phase,
                load,
                run["scheduler_mode"],
                run["prefix_cache_mode"],
                run["closed_loop"]["N"],
            )
            groups[key].append(run_row(run, attempt))
    rows = []
    for (phase, load, sched, cache, n), rs in sorted(groups.items()):
        mean = {k: float(np.mean([r[k] for r in rs])) for k in ("old", "new", "little")}
        rows.append(
            {
                "phase": phase,
                "load": load,
                "sched": "fixed" if sched == "fixed_batch" else "dyn",
                "cache": cache,
                "N": n,
                "seeds": len(rs),
                **mean,
                "completed": "{}–{}".format(
                    min(r["completed"] for r in rs), max(r["completed"] for r in rs)
                ),
                "measure": max(r["measure"] for r in rs),
                "shift": 100 * (mean["new"] / mean["old"] - 1),
                "per_slot": min(r["completed"] for r in rs) / n,
                "check": 100 * max(abs(r["new"] / r["little"] - 1) for r in rs),
                "check_old": 100 * max(abs(r["old"] / r["little"] - 1) for r in rs),
                "signed": [100 * (r["new"] / r["little"] - 1) for r in rs],
            }
        )
    moved = [r for r in rows if abs(r["shift"]) >= 1]
    cols = [
        ("phase", "фаза", "s"),
        ("load", "нагрузка", "s"),
        ("sched", "планировщик", "s"),
        ("cache", "кэш", "s"),
        ("N", "N", "d"),
        ("completed", "завершений в окне", "s"),
        ("old", "RPS было", ".3f"),
        ("new", "RPS стало", ".3f"),
        ("little", "Литтл", ".3f"),
        ("shift", "сдвиг, %", "+.1f"),
    ]
    by_load = defaultdict(list)
    for r in rows:
        by_load[r["load"]].append(r)

    def worst(rs, key):
        ok = [r[key] for r in rs if r["per_slot"] >= PER_SLOT]
        return max(ok) if ok else None

    summary = [
        {
            "load": load,
            "points": len(rs),
            "moved": sum(abs(r["shift"]) >= 1 for r in rs),
            "max_shift": max(abs(r["shift"]) for r in rs),
            "checked": sum(r["per_slot"] >= PER_SLOT for r in rs),
            "check_old": worst(rs, "check_old"),
            "check": worst(rs, "check"),
        }
        for load, rs in by_load.items()
    ]
    # Little over a short window: the error at its borders is about one
    # request per slot, so it is a check only with many completions per slot.
    bands = []
    for lo_, hi_ in [(0, 4), (4, 8), (8, 16), (16, 1e9)]:
        rs = [r for r in rows if lo_ <= r["per_slot"] < hi_]
        if rs:
            bands.append(
                {
                    "band": f"{lo_:g}–{hi_:g}" if hi_ < 1e9 else f"≥ {lo_:g}",
                    "points": len(rs),
                    "check": max(r["check"] for r in rs),
                    "median": float(np.median([r["check"] for r in rs])),
                    "signed": float(np.mean([x for r in rs for x in r["signed"]])),
                }
            )
    text = "\n\n".join(
        [
            "# 1. RPS: завершения в окне / окно",
            f"Точек: {len(rows)}, попыток: {sum(r['seeds'] for r in rows)}. "
            "«Литтл» — среднее число в системе / средняя задержка "
            "завершившихся в окне.",
            "## По нагрузкам (максимум по точкам, расхождение с Литтлом — по попыткам)",
            table(
                summary,
                [
                    ("load", "нагрузка", "s"),
                    ("points", "точек", "d"),
                    ("moved", "сдвиг ≥ 1%", "d"),
                    ("max_shift", "макс. сдвиг, %", ".1f"),
                    ("checked", f"точек с ≥ {PER_SLOT} заверш. на слот", "d"),
                    ("check_old", "было против Литтла, макс. %", ".1f"),
                    ("check", "стало против Литтла, макс. %", ".2f"),
                ],
            ),
            "## Литтл против числа завершений на слот (завершений в окне / N)",
            table(
                bands,
                [
                    ("band", "завершений на слот", "s"),
                    ("points", "точек", "d"),
                    ("median", "стало против Литтла, медиана %", ".2f"),
                    ("signed", "средний знаковый, %", "+.2f"),
                    ("check", "макс. %", ".2f"),
                ],
            ),
            "## Точки, сдвинувшиеся на 1% и больше",
            table(moved, cols),
            "## Все точки",
            table(rows, cols),
        ]
    )
    write("rps.md", text + "\n")


if __name__ == "__main__":
    main()
