"""Сопоставление двух методов на одном train/validation разбиении."""

import argparse
import json
from pathlib import Path


def main(out_dir):
    data = {
        name: json.loads((out_dir / f"{name}_results.json").read_text(encoding="utf-8"))
        for name in ("regression", "optuna")
    }
    for method, result in data.items():
        print(f"\n{method}:")
        for section in ("train", "val"):
            part = result["выборки"][section]
            print(
                f"  {section}: prefill={part['prefill']['MAPE_%']:.2f}%; "
                f"decode={part['decode']['MAPE_%']:.2f}%; "
                f"время прогона={part['средняя_ошибка_прогона_%']:.2f}%"
            )
    (out_dir / "comparison_results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Сравнение калибровок")
    p.add_argument("--out", type=Path, default=Path("outputs_v2"))
    args = p.parse_args()
    main(args.out)
