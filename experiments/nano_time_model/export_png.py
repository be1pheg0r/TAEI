"""Выгружает графики вида отчёта в PNG (по умолчанию — «Презентация»).

Копия ``report.html`` с коротким скриптом открывается в Chrome без окна: скрипт
переводит каждый график в PNG через ``Plotly.toImage`` и кладёт картинки в
DOM, откуда они сохраняются файлами с номером и заголовком раздела::

    python experiments/nano_time_model/export_png.py runs/<время>/report.html \
        tmp_data/presentation_png [--view deck]
"""

import argparse
import base64
import html
import json
import re
import subprocess
import tempfile
from pathlib import Path

HOOK = r"""
<script>
// One bold black local font for slides: titles 24 px, other text 19 px; a local
// font so Plotly measures legends and margins with the face it draws.
const BOLD = { family: "Noto Sans", color: "#000", weight: "bold" };
function boost(o, inTitle = false) {
  if (Array.isArray(o)) return o.forEach((x) => boost(x, inTitle));
  if (!o || typeof o !== "object") return;
  for (const [k, v] of Object.entries(o)) {
    if (/font$/i.test(k) && v && typeof v === "object" && !Array.isArray(v)) {
      Object.assign(v, BOLD, { size: inTitle ? 24 : 19 });
    } else boost(v, k === "title");
    if (/^[xy]axis\d*$/.test(k) && v) v.automargin = true;
  }
}
setTimeout(async () => {
  const out = [];
  for (const box of document.querySelectorAll(".chart")) {
    if (!box.data) continue;
    const sec = box.closest("section");
    const h = sec?.querySelector("h2")?.textContent || "chart";
    const fig = structuredClone({ data: box.data, layout: box.layout });
    fig.layout.font = { ...fig.layout.font, ...BOLD, size: 19 };
    boost(fig);
    const lg = fig.layout.legend, top = { x: 0, y: 1.02, yanchor: "bottom" };
    if (lg?.orientation === "h") Object.assign(lg, top);
    const height = Math.max(700, box.clientHeight * 1.6);
    const opts = { format: "png", width: 1600, height, scale: 3 };
    const url = await Plotly.toImage(fig, opts);
    out.push({ title: h, url });
  }
  const pre = document.createElement("pre");
  pre.id = "png-export";
  pre.textContent = JSON.stringify(out);
  document.body.append(pre);
}, 6000);
</script>
"""


def main(report: Path, out: Path, view: str) -> list[Path]:
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "export.html"
        page.write_text(report.read_text(encoding="utf-8") + HOOK, encoding="utf-8")
        dom = subprocess.run(
            [
                "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
                f"--user-data-dir={tmp}/profile", "--virtual-time-budget=60000",
                "--dump-dom", f"file://{page}#{view}",
            ],
            capture_output=True, text=True, timeout=300, check=True,
        ).stdout  # fmt: skip
    found = re.search(r'<pre id="png-export">(.*?)</pre>', dom, re.S)
    if not found:
        raise SystemExit("charts were not exported: the page did not finish drawing")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    files = []
    for i, item in enumerate(json.loads(html.unescape(found.group(1))), 1):
        title = re.sub(r"^\d+\s*", "", item["title"]).strip()
        name = re.sub(r"[^\w\-–—. ]+", "", title).strip().replace(" ", "_")[:90]
        path = out / f"{i:02d}_{name}.png"
        path.write_bytes(base64.b64decode(item["url"].split(",", 1)[1]))
        files.append(path)
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--view", default="deck", help="deck, brief or m1…")
    args = parser.parse_args()
    for f in main(args.report, args.out, args.view):
        print(f)
