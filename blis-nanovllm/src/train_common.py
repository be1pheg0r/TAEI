"""Общая функция потерь и отчёты для регрессии и Optuna."""

import csv
import json
from collections import defaultdict
import numpy as np
from blis_model import load_rows, select_features, split_rows, design


def setup(csv_path, split_path):
    parts = split_rows(load_rows(csv_path), split_path)
    features, excluded = select_features(parts["train"])
    x, y = design(parts["train"], features)
    counts = {
        p: sum(r["phase"] == p for r in parts["train"]) for p in ("prefill", "decode")
    }
    if min(counts.values()) == 0:
        raise ValueError("В train отсутствует prefill или decode")
    weights = np.array(
        [
            1 / (np.sqrt(2 * counts[r["phase"]]) * r["measured_seconds"])
            for r in parts["train"]
        ]
    )
    # Нормируем каждый столбец ПОСЛЕ взвешивания. Это улучшает численную
    # устойчивость для признаков масштаба 1e-9 и 1e-1 без изменения формулы.
    weighted_x = x * weights[:, None]
    scale = np.linalg.norm(weighted_x, axis=0)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Обнаружен нулевой или некорректный обучаемый столбец")
    return parts, features, excluded, scale, weighted_x / scale, y * weights


def metrics(rows, prediction):
    actual = np.array([r["measured_seconds"] for r in rows])
    p = np.asarray(prediction)
    if not len(rows):
        return None
    return {
        "MAPE_%": round(float(np.mean(np.abs((p - actual) / actual)) * 100), 3),
        "смещение_%": round(float(np.mean((p - actual) / actual) * 100), 3),
        "MSE_относительной_ошибки": float(np.mean(((p - actual) / actual) ** 2)),
    }


def report(parts, features, beta, excluded, method, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "метод": method,
        "признаки": features,
        "исключены": excluded,
        "коэффициенты": dict(zip([*features, "intercept"], map(float, beta))),
        "выборки": {},
    }
    for name, rows in parts.items():
        x, y = design(rows, features)
        prediction = x @ beta
        part = {"число_шагов": len(rows), "все": metrics(rows, prediction)}
        for phase in ("prefill", "decode"):
            indices = [i for i, r in enumerate(rows) if r["phase"] == phase]
            part[phase] = metrics([rows[i] for i in indices], prediction[indices])
        by_run = defaultdict(lambda: [0.0, 0.0])
        for r, p in zip(rows, prediction):
            by_run[r["run_id"]][0] += r["measured_seconds"]
            by_run[r["run_id"]][1] += float(p)
        part["средняя_ошибка_прогона_%"] = round(
            float(np.mean([abs(a - b) / a * 100 for a, b in by_run.values()])), 3
        )
        result["выборки"][name] = part
        with (out_dir / f"{method}_predictions_{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as target:
            writer = csv.DictWriter(
                target,
                fieldnames=[
                    "run_id",
                    "step_id",
                    "phase",
                    "measured_seconds",
                    "predicted_seconds",
                ],
            )
            writer.writeheader()
            for r, p in zip(rows, prediction):
                writer.writerow(
                    {
                        k: r[k]
                        for k in ("run_id", "step_id", "phase", "measured_seconds")
                    }
                    | {"predicted_seconds": float(p)}
                )
    (out_dir / f"{method}_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Обучаемые признаки:", ", ".join(features))
    print("Исключено:", excluded)
    print("Коэффициенты:", result["коэффициенты"])
    for name in ("train", "val"):
        info = result["выборки"][name]
        print(
            f"{name}: prefill MAPE={info['prefill']['MAPE_%']:.2f}%, decode MAPE={info['decode']['MAPE_%']:.2f}%, ошибка прогона={info['средняя_ошибка_прогона_%']:.2f}%"
        )
    return result
