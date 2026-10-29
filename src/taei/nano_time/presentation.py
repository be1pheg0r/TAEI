"""Данные вида «Презентация»: деньги по форме запроса из модели «от N».

Для сетки длин промпта L и ответа O строится синтетический пул запросов
(длины с разбросом ±10%, чтобы не было когорт одинаковых запросов; общее начало
промпта — доля ``SHARE``, четыре группы, кэш включён) и по модели «от N» ищется
наибольшее N, при котором выполняется требование к качеству (TTFT p90 и TPOT не
выше порогов). В клетке — это N, запросов и токенов в секунду, скорость для
пользователя и выручка в час по ценам экономики. Нужен только блок ``system``
отчёта (коэффициенты модели шага, движок, цены), поэтому блок считается и в
полной сборке, и на готовом ``time_model.json``.
"""

from __future__ import annotations

import numpy as np

from . import economics
from . import system as sm
from .decode_variants import DecodeVariant
from .system_report import curve_grid

GRID_L = (256, 512, 1024, 2048, 4096, 8192, 16384)
GRID_O = (16, 32, 64, 128, 256, 512, 1024)
SHARE = 0.5
SPREAD = 0.1
POOL = 200
FAMILIES = 4


def step_model(system: dict) -> sm.StepModel:
    core = system["core"]
    variant = system["variants"][system["choice"]]
    return sm.StepModel(
        {k: core[k] for k in ("a", "b", "c")},
        DecodeVariant(system["choice"], variant["coef"]),
    )


def pool(prompt: int, answer: int, block: int, seed: int = 0) -> sm.Pool:
    """Requests around (L, O) with a cached common start of SHARE of the prompt."""
    rng = np.random.default_rng(seed)
    spec = []
    for i in range(POOL):
        length = max(2, int(round(prompt * rng.uniform(1 - SPREAD, 1 + SPREAD))))
        out = max(2, int(round(answer * rng.uniform(1 - SPREAD, 1 + SPREAD))))
        blocks = min(int(SHARE * length // block), (length - 1) // block)
        spec.append(
            {
                "L": length,
                "O": out,
                "family": f"f{i % FAMILIES}",
                "prefix_blocks": blocks,
            }
        )
    return sm.Pool.from_spec(spec, "on", block)


def cell(system: dict, prompt: int, answer: int, slo: dict, draws: int) -> dict:
    engine = sm.Engine(**system["engine"])
    step = step_model(system)
    p = pool(prompt, answer, engine.block_size)
    run = sm.running(p, engine, draws)
    rows = [sm.predict_dynamic(p, engine, step, n, run) for n in curve_grid(256)]
    best = economics.max_n_under_slo(rows, slo)
    out = {"L": prompt, "O": answer}
    if best is None:
        return out | {"N": None}
    prices = system["economics"]["prices"]
    v = economics.check(
        prices, float(p.new.mean()), float(p.C.mean()), float(p.out.mean())
    )
    return out | {
        "N": best["N"],
        "rps": best["rps"],
        "itps_new": best["itps_new"],
        "otps": best["otps"],
        "ttps": best["ttps"],
        "rev": v * 3600 * best["rps"],
    }


def build(system: dict, slo_label: str = "чат", draws: int = 300) -> dict:
    slo = next(s for s in system["economics"]["slo"] if s["label"] == slo_label)
    grid = [cell(system, p, a, slo, draws) for a in GRID_O for p in GRID_L]
    return {"L": GRID_L, "O": GRID_O, "share": SHARE, "slo": slo_label, "grid": grid}
