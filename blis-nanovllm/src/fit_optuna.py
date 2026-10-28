"""Калибровка коэффициентов BLIS методом Optuna (те же признаки и loss)."""

import argparse
from pathlib import Path
import numpy as np
import optuna
from train_common import setup, report


def main(csv_path, split, out, trials, seed):
    parts, features, excluded, scale, a, b = setup(csv_path, split)
    # Приводим взвешенные столбцы к единичной норме.
    # Это существенно упрощает поиск в пространстве Optuna.
    col_norm = np.linalg.norm(a, axis=0)
    y_norm = float(np.linalg.norm(b))
    a = a / col_norm
    b = b / y_norm
    gram = a.T @ a
    cross = a.T @ b
    const = b @ b
    names = [*features, "intercept"]
    # Нормированное пространство с консервативной верхней границей.
    limits = [3.0] * len(names)

    def objective(trial):
        theta = np.array(
            [
                trial.suggest_float(name, 0.0, limit)
                for name, limit in zip(names, limits)
            ]
        )
        return float(theta @ gram @ theta - 2 * theta @ cross + const)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials)
    theta = np.array([study.best_params[name] for name in names])
    result = report(
        parts, features, theta * y_norm / col_norm / scale, excluded, "optuna", out
    )
    result["число_испытаний"] = trials
    result["лучшее_значение_loss"] = study.best_value
    import json

    (out / "optuna_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Подбор коэффициентов BLIS через Optuna")
    p.add_argument("csv_path", type=Path)
    p.add_argument("--split", type=Path, default=Path("outputs_v2/data_split.json"))
    p.add_argument("--out", type=Path, default=Path("outputs_v2"))
    p.add_argument("--trials", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.csv_path, args.split, args.out, args.trials, args.seed)
