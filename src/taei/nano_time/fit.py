"""Подбор неотрицательных коэффициентов линейной формы ``y ≈ X @ c``.

``regression`` — МНК с ограничением c >= 0 на относительной ошибке: строки
делятся на y, и NNLS минимизирует сумму квадратов ``(X c - y) / y``. ``optuna`` —
TPE с фиксированным seed минимизирует среднюю относительную ошибку
``|X c - y| / y``; коэффициенты ищутся в логарифмическом масштабе в диапазоне
вокруг масштаба данных, решение ``regression`` ставится первой пробой.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

METHODS = ("regression", "optuna")


def relative_errors(pred, meas) -> np.ndarray:
    meas = np.asarray(meas, dtype=float)
    return (np.asarray(pred, dtype=float) - meas) / meas


def _regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, _ = nnls(X / y[:, None], np.ones_like(y))
    return coef


def _optuna(X: np.ndarray, y: np.ndarray, seed: int, trials: int) -> np.ndarray:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start = _regression(X, y)
    # Per-coefficient scale: the value that alone would explain the median y.
    scale = np.array(
        [
            np.median(y) / max(np.median(X[:, j][X[:, j] > 0]), 1e-30)
            if (X[:, j] > 0).any()
            else 1.0
            for j in range(X.shape[1])
        ]
    )
    low, high = scale * 1e-7, scale * 10.0

    def objective(trial):
        coef = np.array(
            [
                trial.suggest_float(f"c{j}", low[j], high[j], log=True)
                for j in range(X.shape[1])
            ]
        )
        return float(np.mean(np.abs(X @ coef - y) / y))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(
        {f"c{j}": float(np.clip(start[j], low[j], high[j])) for j in range(X.shape[1])}
    )
    study.optimize(objective, n_trials=trials)
    return np.array([study.best_params[f"c{j}"] for j in range(X.shape[1])])


def fit_nonneg(
    X, y, method: str = "regression", seed: int = 0, trials: int = 1500
) -> np.ndarray:
    """Non-negative coefficients of ``y ≈ X @ c`` by the chosen method."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if method == "regression":
        return _regression(X, y)
    if method == "optuna":
        return _optuna(X, y, seed, trials)
    raise ValueError(f"unknown fit method {method!r}; expected one of {METHODS}")


def summary(rel: np.ndarray) -> dict[str, float]:
    """Mean, median and max absolute relative error plus mean signed error."""
    rel = np.asarray(rel, dtype=float)
    a = np.abs(rel)
    return {
        "mean_abs": float(a.mean()),
        "median_abs": float(np.median(a)),
        "max_abs": float(a.max()),
        "mean_signed": float(rel.mean()),
        "n": int(len(rel)),
    }
