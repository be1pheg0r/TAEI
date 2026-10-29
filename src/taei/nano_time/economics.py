"""Экономика одной GPU по шагам: цены → чек запроса → время GPU → выручка → прибыль.

0. Цены для 0.6B — от цен 8B пропорционально числу параметров: p = p(8B)·0.6/8.
1. Чек запроса v = (p_in·L(1−R) + p_c·L·R + p_out·O)/10⁶ (цены — $ за 1М токенов).
2. Время GPU на запрос τ = 1/RPS модели «от N» при наибольшем N, у которого
   TTFT p90 и TPOT не выше порогов SLO.
3. Запросов в час Q = 3600·u/τ, u — доля времени под нагрузкой.
4. Выручка в час Rev = v·Q.
5. Прибыль в час: при аренде Rev − C_rent, со своей картой Rev − C_el,
   C_el = мощность (кВт) × тариф ($/кВт·ч).
6. Окупаемость своей карты T_pay = P_gpu / (24·(Rev − C_el)) дней, если прибыль
   положительна.
"""

from __future__ import annotations


def scaled_prices(base: dict, scale: float) -> dict:
    """$ per 1M tokens for the small model: the base model's prices × scale."""
    return {k: base[k] * scale for k in ("input", "cached_input", "output")}


def check(prices: dict, new: float, cached: float, out: float) -> float:
    """v: revenue of one request: L(1−R) new, L·R cached and O output tokens."""
    return (
        prices["input"] * new + prices["cached_input"] * cached + prices["output"] * out
    ) / 1e6


def electricity_per_hour(gpu: dict) -> float:
    """C_el: power (kW) × tariff ($ per kWh)."""
    return gpu["power_kw"] * gpu["tariff_per_kwh"]


def payback_days(price: float, profit_per_hour: float) -> float | None:
    """T_pay = P_gpu / (24·profit); None when the card never pays back."""
    return price / (24 * profit_per_hour) if profit_per_hour > 0 else None


def hour(prices: dict, gpu: dict, v_rows: dict, tau: float, u: float = 1.0) -> dict:
    """Steps 2–6 at one point: τ from the model, ``v_rows`` — mean new, cached
    and output tokens per request of the pool."""
    v = check(prices, v_rows["new"], v_rows["cached"], v_rows["out"])
    q = 3600 * u / tau
    rev = v * q
    c_el = electricity_per_hour(gpu)
    return {
        "v": v,
        "tau": tau,
        "Q": q,
        "rev": rev,
        "profit_rent": rev - gpu["rent_per_hour"],
        "profit_own": rev - c_el,
        "payback_days": payback_days(gpu["price"], rev - c_el),
    }


def max_n_under_slo(rows: list[dict], slo: dict) -> dict | None:
    """The row with the largest N whose TTFT p90 and TPOT meet the SLO."""
    ok = [
        r for r in rows if r["ttft_p90"] <= slo["ttft_p90"] and r["tpot"] <= slo["tpot"]
    ]
    return max(ok, key=lambda r: r["N"]) if ok else None
