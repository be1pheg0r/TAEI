"""Отчёт и сводная страница из ``time_model.json`` по одному шаблону Plotly.

Шаблон ``report_template.html`` получает документ отчёта целиком и строит
разделы в браузере циклом по моделям и наборам переноса, поэтому формат
отчёта не зависит от набора моделей. Сводная страница ``runs/index.html``
использует тот же шаблон со списком запусков. Запуск из командной строки
перестраивает отчёт из готового json без пересчёта::

    python -m taei.nano_time.report runs/<время>/time_model.json
    python -m taei.nano_time.report --index runs
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

from .report_data import SCHEMA_VERSION

TEMPLATE = Path(__file__).with_name("report_template.html")
INDEXED_VERSIONS = (1, SCHEMA_VERSION)  # history keeps runs of the old format


def _fill(title: str, document: dict) -> str:
    data = json.dumps(document, ensure_ascii=False, allow_nan=False)
    # A JSON payload inside <script> must not close the element early.
    data = data.replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8")
    return page.replace("__TITLE__", html.escape(title)).replace(
        "__REPORT_DATA__", data
    )


def render_report(document: dict) -> str:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"report schema v{document.get('schema_version')} != v{SCHEMA_VERSION}"
        )
    return _fill("Модель времени nano-vLLM", {"kind": "report", **document})


def _v1_entry(doc: dict) -> dict:
    labels = {s["id"]: s["label"] for s in doc.get("sources", [])}
    test, transfer = {}, {}
    for row in doc.get("metrics", []):
        name, _, kind = row["what"].partition(": ")
        if kind != "T_total per run":
            continue
        if row["part"] == "test":
            test[name] = row["mean_abs"]
        elif row["part"] == "transfer":
            transfer.setdefault(labels.get(row["source"], row["source"]), {})[name] = (
                row["mean_signed"]
            )
    train = next((s for s in doc.get("sources", []) if s.get("role") == "train"), {})
    return {
        "models": [{"label": train.get("label"), "test": test, "transfer": transfer}]
    }


def _v2_entry(doc: dict) -> dict:
    models = []
    for model in doc["models"]:
        if model.get("kind") == "vidur":
            continue  # an operation-level branch: no T_total to index
        labels = {s["id"]: s["label"] for s in model["sources"]}
        rows = [r for r in model["metrics"] if r["what"] == "T_total per run"]
        test = [r["mean_abs"] for r in rows if r["part"] == "test"]
        models.append(
            {
                "label": model["label"],
                # v2 runs have one decode method: the sum over steps.
                "test": {"additive": test[0]} if test else {},
                "transfer": {
                    labels.get(r["source"], r["source"]): {"additive": r["mean_signed"]}
                    for r in rows
                    if r["part"] == "transfer"
                },
            }
        )
    return {"models": models}


def index_entry(json_path: Path, root: Path) -> dict:
    """One row per run: test |e| and transfer e of T_total, for v1 and v2 runs."""
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    meta = doc.get("meta", {})
    body = _v2_entry(doc) if doc.get("schema_version") == 2 else _v1_entry(doc)
    return {
        "run": json_path.parent.name,
        "report": os.path.relpath(json_path.parent / "report.html", root),
        "schema_version": doc.get("schema_version"),
        "fit": meta.get("fit"),
        "decode": meta.get("decode", "additive"),
        "seed": meta.get("seed"),
        "git_commit": meta.get("git_commit"),
        **body,
    }


def render_index(root: Path) -> str:
    entries = [
        index_entry(path, root)
        for path in sorted(root.glob("*/time_model.json"))
        if json.loads(path.read_text(encoding="utf-8")).get("schema_version")
        in INDEXED_VERSIONS
    ]
    return _fill(
        "Запуски модели времени",
        {"kind": "index", "schema_version": SCHEMA_VERSION, "entries": entries},
    )


def write_report(json_path: Path) -> Path:
    document = json.loads(json_path.read_text(encoding="utf-8"))
    out = json_path.with_name("report.html")
    out.write_text(render_report(document), encoding="utf-8")
    return out


def write_index(root: Path) -> Path:
    out = root / "index.html"
    out.write_text(render_index(root), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "path", type=Path, help="time_model.json, or runs/ with --index"
    )
    parser.add_argument("--index", action="store_true", help="rebuild runs/index.html")
    args = parser.parse_args()
    print(write_index(args.path) if args.index else write_report(args.path))


if __name__ == "__main__":
    main()
