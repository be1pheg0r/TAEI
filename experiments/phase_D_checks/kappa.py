"""4. Цена общего префикса в decode (S7): ``Σℓ_eff = U + κ·(Σℓ − U)``.

U — уникальные токены шага: Σℓ минус повторы префикса, ``Σ_f (m_f − 1)·C_f``
(m_f — запросов семьи f в батче, C_f — их фактический C из трассы, поэтому
R = 0,999 при ℓ = 1025 учтено). κ = 1 — decode платит за Σℓ, κ = 0 — только за
уникальные токены. d0, d1, d2 формы ядра подбираются ``fit_nonneg`` на
попытках без общих блоков фазы D (S6 и контроль S7 R = 0, кэш вкл и выкл) —
та же фаза, сдвига между фазами нет. Строка — попытка: медианы по её
decode-шагам, первые 4 отброшены. κ попытки решается из её T; κ точки —
среднее по сидам.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize_scalar

from common import ROOT, attempts, decode_steps, jsonl, table, write

sys.path.insert(0, str(ROOT / "src"))
from taei.nano_time.fit import fit_nonneg  # noqa: E402

SKIP = 4
MIN_SHARED = 0.1  # κ is not identified when repeats are a small share of Σℓ


def unique_tokens(contexts: dict, reqs: dict) -> float:
    by_family = defaultdict(list)
    for rid in contexts:
        r = reqs[rid]
        if r["prefix_family_id"] and r["prefix_reused_tokens"]:
            by_family[r["prefix_family_id"]].append(r["prefix_reused_tokens"])
    repeats = sum((len(c) - 1) * min(c) for c in by_family.values())
    return sum(contexts.values()) - repeats


def rows_of(pattern: str) -> list[dict]:
    rows = []
    for run, attempt in attempts("D", pattern):
        reqs = {r["request_id"]: r for r in jsonl(attempt / "requests.jsonl")}
        steps = decode_steps(attempt, SKIP)
        B = max(s["B"] for s in steps)
        full = [s for s in steps if s["B"] == B]
        point = run["workload_spec"]["point"]
        C = [r["prefix_reused_tokens"] / r["input_token_count"] for r in reqs.values()]
        rows.append(
            {
                "series": run["series"],
                "cache": run["prefix_cache_mode"],
                "B": B,
                "ell": point.get("L"),
                "R_target": point.get("R_target", 0.0),
                "R": float(np.mean(C)),
                "F": point.get("families", 0),
                "seed": run["seed"],
                "sum_l": float(np.median([s["sum_l"] for s in full])),
                "U": float(
                    np.median([unique_tokens(s["contexts"], reqs) for s in full])
                ),
                "T": float(np.median([s["T"] for s in full])),
            }
        )
    return rows


def fit_d(rows) -> np.ndarray:
    X = np.array([[1.0, r["sum_l"], r["B"]] for r in rows])
    return fit_nonneg(X, np.array([r["T"] for r in rows]), "regression")


def joint_kappa(rows) -> float:
    """κ with d0, d1, d2 refit on all rows (the same relative error)."""

    def loss(k):
        X = np.array([[1.0, r["U"] + k * (r["sum_l"] - r["U"]), r["B"]] for r in rows])
        y = np.array([r["T"] for r in rows])
        c = fit_nonneg(X, y, "regression")
        return float(((((X @ c) - y) / y) ** 2).sum())

    return float(minimize_scalar(loss, bounds=(-1.0, 2.0), method="bounded").x)


def main() -> None:
    s6, s7 = rows_of("D-S6-*"), rows_of("D-S7-*")
    base = s6 + [r for r in s7 if r["R_target"] == 0]
    shared = [r for r in s7 if r["R_target"] > 0]
    d0, d1, d2 = fit_d(base)
    base_err = [100 * ((d0 + d1 * r["sum_l"] + d2 * r["B"]) / r["T"] - 1) for r in base]
    for r in shared:
        r["share"] = 1 - r["U"] / r["sum_l"]
    ok = [r for r in shared if r["share"] >= MIN_SHARED]
    for r in ok:
        r["kappa"] = ((r["T"] - d0 - d2 * r["B"]) / d1 - r["U"]) / (r["sum_l"] - r["U"])
        # κ moved by a 1% error of T: the noise floor of κ at this point.
        r["dk"] = 0.01 * r["T"] / (d1 * (r["sum_l"] - r["U"]))
    x = np.array([d1 * (r["sum_l"] - r["U"]) / r["T"] for r in ok])
    y = np.array([(r["T"] - d0 - d2 * r["B"] - d1 * r["U"]) / r["T"] for r in ok])
    kappa = float((x * y).sum() / (x * x).sum())
    kappa_joint = joint_kappa(base + ok)

    # Paired with the R = 0 control of the same B and ℓ (cache on, the same Σℓ):
    # ΔT = d1·(1 − κ)·(Σℓ − U), so an error of the form at this B cancels.
    control = defaultdict(list)
    for r in s7:
        if r["R_target"] == 0 and r["cache"] == "on":
            control[(r["B"], r["ell"])].append(r["T"])
    points = defaultdict(list)
    for r in ok:
        points[(r["B"], r["ell"], r["R_target"], r["F"])].append(r)
    prows = [
        {
            "B": k[0],
            "ell": k[1],
            "R_target": k[2],
            "F": k[3],
            "R": np.mean([r["R"] for r in rs]),
            "share": np.mean([r["share"] for r in rs]),
            "kappa": np.mean([r["kappa"] for r in rs]),
            "sd": np.std([r["kappa"] for r in rs], ddof=1) if len(rs) > 1 else None,
            "dk": np.mean([r["dk"] for r in rs]),
            "pair": (
                1
                - (np.mean(control[k[:2]]) - np.mean([r["T"] for r in rs]))
                / (d1 * np.mean([r["sum_l"] - r["U"] for r in rs]))
                if k[:2] in control
                else None
            ),
        }
        for k, rs in sorted(points.items())
    ]

    def margin(key, label):
        out = []
        for v in sorted({p[key] for p in prows}):
            ps = [p for p in prows if p[key] == v]
            ks = np.array([p["kappa"] for p in ps])
            pair = [p["pair"] for p in ps if p["pair"] is not None]
            base_b = (
                [
                    e
                    for e, r in zip(base_err, base, strict=True)
                    if r["series"] == "S7" and r[key] == v
                ]
                if key in ("B", "ell")
                else []
            )
            out.append(
                {
                    "factor": label,
                    "value": f"{v:g}",
                    "points": len(ps),
                    "median": float(np.median(ks)),
                    "q1": float(np.percentile(ks, 25)),
                    "q3": float(np.percentile(ks, 75)),
                    "dk": float(np.median([p["dk"] for p in ps])),
                    "pairs": len(pair),
                    "pair": float(np.median(pair)) if pair else None,
                    "base": float(np.median(base_b)) if base_b else None,
                }
            )
        return out

    margins = (
        margin("B", "B")
        + margin("ell", "ℓ")
        + margin("R_target", "R (цель)")
        + margin("F", "семей")
    )
    text = "\n\n".join(
        [
            "# 4. Кэш в decode (S7): κ в Σℓ_eff = U + κ·(Σℓ − U)",
            f"d по {len(base)} попыткам без общих блоков (S6 и S7 R = 0): "
            f"d0 = {d0:.3e} с, d1 = {d1:.3e} с/ток, d2 = {d2:.3e} с; "
            f"медиана |ошибки| {np.median(np.abs(base_err)):.1f}%.",
            f"Попыток S7 с R > 0: {len(shared)}; с долей повторов "
            f"≥ {MIN_SHARED:.0%} — {len(ok)} ({len(points)} точек).",
            f"**κ общий = {kappa:.3f}** (d из попыток без общих блоков); "
            f"при d, подобранных вместе с κ на всех попытках, κ = {kappa_joint:.3f}.",
            "## κ по факторам (точка — среднее по сидам; медиана и квартили по точкам)",
            "«±1% T» — на сколько сдвигается κ точки от ошибки T в 1%: порог шума. "
            "«κ к контролю» — из разницы с попыткой R = 0 того же B и ℓ "
            "(кэш вкл, Σℓ тот же): ΔT = d1·(1 − κ)·(Σℓ − U), ошибка формы при этом B "
            "сокращается; контроль есть не у всех B·ℓ. «Ошибка формы» — медиана "
            "(T̂ − T)/T на контроле R = 0.",
            table(
                margins,
                [
                    ("factor", "фактор", "s"),
                    ("value", "значение", "s"),
                    ("points", "точек", "d"),
                    ("median", "κ, медиана", ".2f"),
                    ("q1", "Q1", ".2f"),
                    ("q3", "Q3", ".2f"),
                    ("dk", "±1% T", ".2f"),
                    ("pairs", "точек с контролем", "d"),
                    ("pair", "κ к контролю R = 0", ".2f"),
                    ("base", "ошибка формы на R = 0, %", "+.1f"),
                ],
            ),
            "## Все точки",
            table(
                prows,
                [
                    ("B", "B", "d"),
                    ("ell", "ℓ", "d"),
                    ("R_target", "R цель", "g"),
                    ("R", "R факт", ".3f"),
                    ("F", "семей", "d"),
                    ("share", "доля повторов", ".2f"),
                    ("kappa", "κ", ".2f"),
                    ("sd", "sd по сидам", ".2f"),
                    ("dk", "±1% T", ".2f"),
                    ("pair", "κ к контролю", ".2f"),
                ],
            ),
        ]
    )
    write("kappa.md", text + "\n")


if __name__ == "__main__":
    main()
