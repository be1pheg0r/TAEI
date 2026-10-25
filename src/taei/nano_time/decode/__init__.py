"""Способы учёта decode с общим интерфейсом «прогон → T̂_total».

Каждый модуль пакета объявляет класс-способ в переменной ``METHOD``; реестр
собирается автоматически, поэтому новый способ добавляется новым файлом без
правки остального кода. Интерфейс способа:

* ``name`` — имя для флага ``decode``;
* ``fit(data, train_runs, prefill, fit_method, **fit_kwargs)`` — подбор только
  на train-прогонах; ``prefill`` — уже подобранная модель A;
* ``predict_total(data, run_ids) -> pd.Series`` — T̂_total по прогонам;
* ``flags(data, run_ids) -> pd.Series`` — пометки прогонов (пустая строка, если
  пометок нет);
* ``describe() -> dict`` — коэффициенты и прочее для отчёта и json.
"""

from __future__ import annotations

import importlib
import pkgutil


def registry() -> dict[str, type]:
    methods = {}
    for module in pkgutil.iter_modules(__path__):
        loaded = importlib.import_module(f"{__name__}.{module.name}")
        method = getattr(loaded, "METHOD", None)
        if method is not None:
            methods[method.name] = method
    return dict(sorted(methods.items()))


def select(flag: str) -> dict[str, type]:
    """Methods for a ``decode`` flag: a method name or ``both`` (all methods)."""
    methods = registry()
    if flag in ("both", "all"):
        return methods
    if flag not in methods:
        raise ValueError(f"unknown decode method {flag!r}; known: {sorted(methods)}")
    return {flag: methods[flag]}
