"""Доразбор κ: κ вместе с порогом шага F (``nano_time.decode_variants``).

T_dc = max(F, d0 + d1·(U + κ·(Σℓ − U)) + d2·B): d0, d1, d2, F — по decode-шагам
S2 + S6, κ — по S7, только попытки с T ≥ 1,2·F и долей повторов ≥ 20%.
Показывает три варианта decode на шагах S2, S3, S6, S7 и κ по факторам,
включая число запросов на семью (B / семей).
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from common import DATA, ROOT, table, write

sys.path.insert(0, str(ROOT / "src"))
from taei.nano_time import data as nano_data  # noqa: E402
from taei.nano_time import decode_variants as dv  # noqa: E402

PHASES = ("A", "B", "D")


def plan_points() -> tuple[pd.DataFrame, dict[str, int]]:
    rows, warmup = [], {}
    for phase in PHASES:
        plan = json.loads((DATA / f"phase_{phase}" / "plan.json").read_text())
        for row in plan["matrix"]:
            point = (row.get("workload_spec") or {}).get("point") or {}
            warmup[row["run_id"]] = int(point.get("warmup_steps", 0))
            rows.append(
                {
                    "run_id": row["run_id"],
                    "ell": point.get("L"),
                    "R_target": point.get("R_target", point.get("R", 0.0)),
                    "families": point.get("families", 0),
                }
            )
    return pd.DataFrame(rows), warmup


def main() -> None:
    roots = [DATA / f"phase_{p}" / "runs" for p in PHASES]
    data = nano_data.load(roots, "local", None, ["S2", "S3", "S6", "S7"])
    points, warmup = plan_points()
    fit = dv.fit_all(data.decode_steps, points, warmup)
    coef = {k: v.coef for k, v in fit.variants.items()}
    errors = pd.DataFrame(fit.errors)
    pivot = errors.pivot_table(index="series", columns="variant", values="mean_abs")
    err_rows = [
        {"series": s, **{v: 100 * pivot.loc[s, v] for v in dv.VARIANTS}}
        for s in pivot.index
    ]
    sel = fit.kappa_rows
    k = coef["kappa"]["kappa"]
    text = "\n\n".join(
        [
            "# Доразбор κ: κ вместе с порогом F",
            "Коэффициенты (с, с/ток): "
            + "; ".join(
                f"{dv.LABELS[n]}: d0 = {c['d0']:.3e}, d1 = {c['d1']:.3e}, "
                f"d2 = {c['d2']:.3e}, F = {c['F']:.3e}, κ = {c['kappa']:.3f}"
                for n, c in coef.items()
            )
            + ".",
            f"S7 с повторами: {len(fit.kappa_all)} попыток; в подбор κ "
            f"(T ≥ {dv.FLOOR_MARGIN}·F, доля повторов ≥ {dv.MIN_SHARED:.0%}): "
            f"{len(sel)}. "
            f"**κ = {k:.3f}**. Выбран вариант по шагу: {dv.LABELS[fit.choice]}.",
            "## Средняя |ошибка| decode-шага, %",
            table(
                err_rows,
                [("series", "серия", "s")]
                + [(v, dv.LABELS[v], ".1f") for v in dv.VARIANTS],
            ),
            "## κ по факторам (точка — среднее по сидам)",
            "per_family — запросов на семью, B / семей.",
            table(
                dv.kappa_table(sel),
                [
                    ("factor", "фактор", "s"),
                    ("value", "значение", "g"),
                    ("points", "точек", "d"),
                    ("median", "κ, медиана", ".2f"),
                    ("q1", "Q1", ".2f"),
                    ("q3", "Q3", ".2f"),
                ],
            ),
        ]
    )
    write("kappa_floor.md", text + "\n")


if __name__ == "__main__":
    main()
