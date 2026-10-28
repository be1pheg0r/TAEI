"""Единое описание формулы BLIS и проверка идентифицируемости коэффициентов."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np

CANDIDATES = (
    "t_prefill_compute",
    "t_prefill_kv",
    "t_decode_compute",
    "t_decode_kv",
    "t_tp",
    "batch_size",
)


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"run_id", "step_id", "phase", "measured_seconds", *CANDIDATES}
        absent = required - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f"Нет колонок: {sorted(absent)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Таблица пуста")
    for row in rows:
        for key in (*CANDIDATES, "measured_seconds"):
            row[key] = float(row[key])
            if not np.isfinite(row[key]):
                raise ValueError("NaN или бесконечность в таблице")
        if row["measured_seconds"] <= 0 or row["batch_size"] < 1:
            raise ValueError("Недопустимая длительность или размер батча")
    return rows


def select_features(rows):
    """Удаляем константные и линейно зависимые признаки на train, не глядя в test."""
    selected = []
    dropped = {}
    vectors = []
    for key in CANDIDATES:
        col = np.array([r[key] for r in rows], dtype=float)
        if np.ptp(col) <= max(1e-15, 1e-12 * max(abs(col.max()), abs(col.min()))):
            dropped[key] = "константный признак"
            continue
        vectors.append((key, col))
    base = np.ones((len(rows), 1))
    rank = 1
    for key, col in vectors:
        candidate = np.column_stack((base, col / max(abs(col).max(), 1e-30)))
        new_rank = np.linalg.matrix_rank(candidate, tol=1e-9)
        if new_rank > rank:
            selected.append(key)
            base = candidate
            rank = new_rank
        else:
            dropped[key] = "линейно зависим от ранее выбранных признаков"
    return selected, dropped


def design(rows, features):
    x = np.array([[r[k] for k in features] + [1.0] for r in rows], dtype=float)
    y = np.array([r["measured_seconds"] for r in rows], dtype=float)
    return x, y


def split_rows(rows, split_file):
    split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    names = ("train", "val")
    sets = [set(split[f"{name}_runs"]) for name in names]
    if sets[0] & sets[1]:
        raise ValueError("Train и validation пересекаются")
    observed = {r["run_id"] for r in rows}
    if sets[0] | sets[1] != observed:
        raise ValueError("Разбиение и CSV содержат разные прогоны")
    if not all(sets):
        raise ValueError("Одна из выборок пуста")
    return {
        name: [r for r in rows if r["run_id"] in group]
        for name, group in zip(names, sets)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Проверка независимости признаков BLIS"
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--split", type=Path, default=Path("outputs_v2/data_split.json")
    )
    args = parser.parse_args()
    parts = split_rows(load_rows(args.csv_path), args.split)
    features, dropped = select_features(parts["train"])
    print("Обучаемые признаки:", ", ".join(features))
    print("Исключённые признаки:", dropped)
    print("Размеры выборок:", {name: len(part) for name, part in parts.items()})
