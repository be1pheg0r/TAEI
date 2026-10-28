"""Разделение 48 прогонов на 24 train и 24 val по seed нагрузки.

В каждом из шести семейств seed=42 идёт в train, seed=43 — в val.
Все четыре сочетания кэша и планировщика представлены в обеих выборках.

Запуск: python src/make_split.py data/ds_qwen_3_8b/runs --out outputs_8b
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from data_source import open_dataset


def make_split(data_path, out_dir):
    groups = defaultdict(list)
    conditions = defaultdict(Counter)
    seen = set()
    with open_dataset(data_path) as archive:
        for filename in sorted(
            n for n in archive.namelist() if n.endswith("/runs.json")
        ):
            run = json.loads(archive.read(filename))
            run_id = run["run_id"]
            if run_id in seen:
                raise ValueError(f"Повторяется run_id: {run_id}")
            seen.add(run_id)
            if run.get("status") != "success":
                raise ValueError(f"Прогон {run_id} неуспешен: {run.get('status')}")
            family = run.get("split_group_id")
            workload_id = run.get("workload_id", "")
            match = re.search(r"-seed-(\d+)$", workload_id)
            if not family or not match:
                raise ValueError(f"Не найдено семейство или seed нагрузки для {run_id}")
            seed = int(match.group(1))
            if seed not in (42, 43):
                raise ValueError(f"Неожиданный seed {seed} в {run_id}; нужны 42 и 43")
            mode = run.get("prefix_cache_mode") or run.get("effective_config", {}).get(
                "prefix_cache_mode"
            )
            scheduler = run.get("scheduler_mode") or run.get(
                "effective_config", {}
            ).get("scheduler_mode")
            if mode not in ("on", "off") or scheduler not in (
                "fixed_batch",
                "upstream_dynamic",
            ):
                raise ValueError(
                    f"Неизвестная конфигурация cache/scheduler в {run_id}: {mode}/{scheduler}"
                )
            groups[(family, seed)].append(run_id)
            conditions[(family, seed)][(mode, scheduler)] += 1

    families = sorted({family for family, seed in groups})
    expected = {
        ("on", "fixed_batch"),
        ("on", "upstream_dynamic"),
        ("off", "fixed_batch"),
        ("off", "upstream_dynamic"),
    }
    if len(families) != 6 or len(seen) != 48:
        raise ValueError(
            f"Ожидалось 6 семейств и 48 прогонов; найдено {len(families)} и {len(seen)}"
        )
    for family in families:
        for seed in (42, 43):
            if (
                len(groups[(family, seed)]) != 4
                or set(conditions[(family, seed)]) != expected
                or any(count != 1 for count in conditions[(family, seed)].values())
            ):
                raise ValueError(
                    f"Неполный набор для {family}, seed={seed}: {dict(conditions[(family, seed)])}"
                )

    result = {
        "правило": "24/24: для каждого семейства seed=42 -> train, seed=43 -> val",
        "примечание": "Совпадает с отчётом коллеги по размеру выборок; точные ID не сопоставимы с его long-context L01-L48",
        "train_seed": 42,
        "val_seed": 43,
        "train_groups": families,
        "val_groups": families,
        "train_runs": sorted(
            (r for (family, seed), ids in groups.items() if seed == 42 for r in ids)
        ),
        "val_runs": sorted(
            (r for (family, seed), ids in groups.items() if seed == 43 for r in ids)
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "data_split.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Train: {len(result['train_runs'])} прогонов (seed 42)")
    print(f"Validation: {len(result['val_runs'])} прогонов (seed 43)")
    print("В обеих выборках: 6 семейств, cache on/off, fixed_batch/upstream_dynamic")
    print("Разбиение сохранено:", path)
    print("После смены разбиения нужно заново обучить регрессию и Optuna.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Разделение train/validation 24/24 по seed"
    )
    parser.add_argument(
        "data_path", type=Path, help="Папка runs или ZIP с экспериментами"
    )
    parser.add_argument("--out", type=Path, default=Path("outputs_8b"))
    args = parser.parse_args()
    make_split(args.data_path, args.out)
