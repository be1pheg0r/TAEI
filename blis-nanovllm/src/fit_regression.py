"""Калибровка коэффициентов BLIS ограниченной линейной регрессией."""

import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from train_common import setup, report


def main(csv_path, split, out):
    parts, features, excluded, scale, a, b = setup(csv_path, split)
    fit = lsq_linear(a, b, bounds=(0, np.inf), tol=1e-12, max_iter=3000)
    if not fit.success:
        raise RuntimeError(f"Регрессия не сошлась: {fit.message}")
    beta = fit.x / scale
    report(parts, features, beta, excluded, "regression", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Подбор коэффициентов BLIS регрессией")
    p.add_argument("csv_path", type=Path)
    p.add_argument("--split", type=Path, default=Path("outputs_v2/data_split.json"))
    p.add_argument("--out", type=Path, default=Path("outputs_v2"))
    args = p.parse_args()
    main(args.csv_path, args.split, args.out)
