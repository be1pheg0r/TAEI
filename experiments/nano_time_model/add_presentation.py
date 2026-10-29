"""Дописывает блок «Презентация» в готовый ``time_model.json`` и пересобирает отчёт.

Блок считается только по разделу ``system`` документа (модель шага, движок, цены),
поэтому подбор модели не повторяется::

    python experiments/nano_time_model/add_presentation.py runs/<время>/time_model.json
"""

import json
import sys
from pathlib import Path

from taei.nano_time import presentation, report, system_report
from taei.nano_time.report_data import clean


def main(path: Path) -> Path:
    doc = json.loads(path.read_text(encoding="utf-8"))
    for model in doc["models"]:
        system = (model.get("local") or {}).get("system")
        if system:
            system["presentation"] = clean(presentation.build(system))
            system_report._slim(system["workloads"], system["choice"])
    path.write_text(
        json.dumps(doc, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return report.write_report(path)


if __name__ == "__main__":
    print(main(Path(sys.argv[1])))
