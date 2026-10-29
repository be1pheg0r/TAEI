"""3. Decode на S2 (фаза A) и S6 (фаза D) вместе, с поправкой на фазу.

Форма ядра ``T = d0 + d1·Σℓ + d2·B`` (``nano_time.decode``), подбор ``fit_nonneg``
(NNLS на относительной ошибке). Строка — попытка: медианы T и Σℓ по её
decode-шагам при полном B, первые 4 decode-шага отброшены (прогрев S6).
Поправка на фазу — множитель (1 + δ) у всех шагов фазы D; δ подбирается по
той же ошибке. Доверительный интервал δ — бутстрэп по точкам (все повторы
точки вместе: у S2 повторы с одним сидом, то есть с одинаковыми токенами).
"""

from __future__ import annotations

import re
import sys

import numpy as np
from scipy.optimize import minimize_scalar

from common import ROOT, attempts, decode_steps, table, write

sys.path.insert(0, str(ROOT / "src"))
from taei.nano_time.fit import fit_nonneg  # noqa: E402

SKIP = 4
BOOT = 1000


def rows_of(phase: str, pattern: str) -> list[dict]:
    rows = []
    for run, attempt in attempts(phase, pattern):
        steps = decode_steps(attempt, SKIP)
        B = max(s["B"] for s in steps)
        full = [s for s in steps if s["B"] == B]
        point = run["workload_spec"].get("point") or {}
        rows.append(
            {
                "phase": phase,
                "point": re.sub(r"-seed-\d+$", "", run["workload_id"]),
                "mixed": bool(point.get("mixed")),
                "B": B,
                "sum_l": float(np.median([s["sum_l"] for s in full])),
                "T": float(np.median([s["T"] for s in full])),
            }
        )
    return rows


def design(rows) -> np.ndarray:
    return np.array([[1.0, r["sum_l"], r["B"]] for r in rows])


def loss_at(delta, X, y, d) -> tuple[float, np.ndarray]:
    y_adj = y / (1 + delta * d)
    coef = fit_nonneg(X, y_adj, "regression")
    rel = (X @ coef - y_adj) / y_adj
    return float((rel**2).sum()), coef


def fit_delta(rows) -> tuple[float, np.ndarray]:
    X, y = design(rows), np.array([r["T"] for r in rows])
    d = np.array([r["phase"] == "D" for r in rows], float)
    best = minimize_scalar(
        lambda x: loss_at(x, X, y, d)[0], bounds=(-0.2, 0.2), method="bounded"
    )
    return float(best.x), loss_at(best.x, X, y, d)[1]


def delta_ci(rows, rng) -> tuple[float, np.ndarray, float, float]:
    delta, coef = fit_delta(rows)
    points = {}
    for r in rows:
        points.setdefault((r["phase"], r["point"]), []).append(r)
    by_phase = {p: [k for k in points if k[0] == p] for p in ("A", "D")}
    boot = []
    for _ in range(BOOT):
        sample = []
        for ks in by_phase.values():
            for i in rng.integers(len(ks), size=len(ks)):
                sample += points[ks[i]]
        boot.append(fit_delta(sample)[0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return delta, coef, float(lo), float(hi)


def overlap(rows) -> list[dict]:
    """S2 plus the S6 attempts whose Σℓ lies within the S2 range at the same B."""
    span = {}
    for r in rows:
        if r["phase"] == "A":
            a, b = span.get(r["B"], (np.inf, -np.inf))
            span[r["B"]] = (min(a, r["sum_l"]), max(b, r["sum_l"]))
    return [
        r
        for r in rows
        if r["phase"] == "A"
        or (r["B"] in span and span[r["B"]][0] <= r["sum_l"] <= span[r["B"]][1])
    ]


def main() -> None:
    rows = rows_of("A", "A-S2-*") + rows_of("D", "D-S6-*")
    rng = np.random.default_rng(0)
    delta, coef, lo, hi = delta_ci(rows, rng)
    both = overlap(rows)
    delta_o, _, lo_o, hi_o = delta_ci(both, rng)

    X, y = design(rows), np.array([r["T"] for r in rows])
    common = fit_nonneg(X, y, "regression")
    for r, p0, p1 in zip(rows, X @ common, X @ coef, strict=True):
        r["e0"] = p0 / r["T"] - 1
        r["e1"] = p1 * (1 + delta * (r["phase"] == "D")) / r["T"] - 1
    s2_only = [r for r in rows if r["phase"] == "A"]
    coef_a = fit_nonneg(
        design(s2_only), np.array([r["T"] for r in s2_only]), "regression"
    )
    for r, p in zip(rows, design(rows) @ coef_a, strict=True):
        r["eA"] = p / r["T"] - 1

    by_b = []
    for B in sorted({r["B"] for r in rows}):
        a = [r for r in rows if r["B"] == B and r["phase"] == "A"]
        d = [r for r in rows if r["B"] == B and r["phase"] == "D"]

        def med(rs, k):
            return 100 * float(np.median([r[k] for r in rs])) if rs else None

        by_b.append(
            {
                "B": B,
                "nA": len(a),
                "nD": len(d),
                "A_e0": med(a, "e0"),
                "D_e0": med(d, "e0"),
                "D_eA": med(d, "eA"),
                "A_e1": med(a, "e1"),
                "D_e1": med(d, "e1"),
            }
        )
    mixed = [r for r in rows if r["mixed"]]
    text = "\n\n".join(
        [
            "# 3. S6 против S2: decode вместе с поправкой на фазу",
            f"Попыток: S2 {sum(r['phase'] == 'A' for r in rows)}, "
            f"S6 {sum(r['phase'] == 'D' for r in rows)} (из них mixed {len(mixed)}). "
            f"Бутстрэп по точкам, {BOOT} раз.",
            f"**δ (фаза D к фазе A) = {100 * delta:+.2f}%**, 95% интервал "
            f"[{100 * lo:+.2f}%, {100 * hi:+.2f}%] — все попытки.",
            f"Только пересечение (S6 с Σℓ в диапазоне S2 при том же B, "
            f"{sum(r['phase'] == 'D' for r in both)} попыток S6): "
            f"δ = {100 * delta_o:+.2f}%, [{100 * lo_o:+.2f}%, {100 * hi_o:+.2f}%].",
            "Коэффициенты: общий подбор без поправки d0 = {:.3e} с, "
            "d1 = {:.3e} с/ток, d2 = {:.3e} с; с поправкой d0 = {:.3e}, "
            "d1 = {:.3e}, d2 = {:.3e}.".format(*common, *coef),
            "## Медианная относительная ошибка по B, %",
            "e0 — общий подбор без поправки; «по S2» — подбор только на S2; "
            "e1 — с поправкой δ.",
            table(
                by_b,
                [
                    ("B", "B", "d"),
                    ("nA", "попыток S2", "d"),
                    ("nD", "попыток S6", "d"),
                    ("A_e0", "S2, e0", "+.1f"),
                    ("D_e0", "S6, e0", "+.1f"),
                    ("D_eA", "S6 по S2", "+.1f"),
                    ("A_e1", "S2, e1", "+.1f"),
                    ("D_e1", "S6, e1", "+.1f"),
                ],
            ),
        ]
    )
    write("s6_vs_s2.md", text + "\n")


if __name__ == "__main__":
    main()
