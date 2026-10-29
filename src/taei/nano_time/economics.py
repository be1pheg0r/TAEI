"""Экономика точки нагрузки: выручка, прибыль и окупаемость GPU, максимум при SLO.

Выручка в час = 3600·(p_вход·ITPS новые + p_кэш·ITPS из кэша + p_выход·OTPS)/10⁶,
цены — за 1М токенов. Прибыль в час = выручка − стоимость GPU в час;
окупаемость — цена GPU / прибыль в час (в часах), если прибыль положительна.
Максимум при SLO — наибольшее N, при котором TTFT p90 и TPOT не выше порогов.
"""

from __future__ import annotations


def revenue_per_hour(
    prices: dict, itps_new: float, itps_cache: float, otps: float
) -> float:
    return (
        3600
        * (
            prices["input"] * itps_new
            + prices["cached_input"] * itps_cache
            + prices["output"] * otps
        )
        / 1e6
    )


def point(
    prices: dict, gpu: dict, itps_new: float, itps_cache: float, otps: float
) -> dict:
    revenue = revenue_per_hour(prices, itps_new, itps_cache, otps)
    profit = revenue - gpu["cost_per_hour"]
    return {
        "revenue_per_hour": revenue,
        "profit_per_hour": profit,
        "payback_hours": gpu["price"] / profit if profit > 0 else None,
    }


def max_n_under_slo(rows: list[dict], slo: dict) -> dict | None:
    """The row with the largest N whose TTFT p90 and TPOT meet the SLO."""
    ok = [
        r for r in rows if r["ttft_p90"] <= slo["ttft_p90"] and r["tpot"] <= slo["tpot"]
    ]
    return max(ok, key=lambda r: r["N"]) if ok else None
