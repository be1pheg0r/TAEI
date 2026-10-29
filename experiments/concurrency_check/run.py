"""Справляется ли GPU с потоком запросов серии S4 или очередь растёт без предела.

В S4 нагрузка задана интенсивностью (запросов в секунду, ρ от оценки ёмкости),
а не числом одновременных запросов. Если GPU обслуживает медленнее, чем
приходят запросы, число запросов в системе N(t) = пришло − завершилось растёт
всё время прихода, и TTFT растёт с номером запроса. Скрипт по готовым трассам
(без GPU и перезапусков) считает для каждого прогона:

- предложенную и обслуженную интенсивность, входные и выходные токены в секунду;
- N(t), его наклон за окно прихода (рост на один пришедший запрос) и очередь
  ожидания из шагов планировщика;
- реальную параллельность — число запросов в батче шага (running);
- TTFT первой и последней четверти запросов.

Рост ≈ 0 — GPU справляется (N(t) стационарно), рост > growth_limit — очередь
растёт без предела, а измеренное время ответа зависит от длины прогона.
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig

from taei.runtime import run_experiment

REPO = Path(__file__).resolve().parents[2]
NS = 1e9


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> dict:
    return run_experiment(cfg, run)


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _in_system(arrive: np.ndarray, finish: np.ndarray, at: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(arrive), at, "right") - np.searchsorted(
        np.sort(finish), at, "right"
    )


def attempt(path: Path) -> dict:
    point = json.loads((path / "workload.json").read_text(encoding="utf-8"))
    point = point["summary"]["point"]
    req = sorted(
        _jsonl(path / "requests.jsonl"), key=lambda r: r["enqueue_monotonic_ns"]
    )
    steps = _jsonl(path / "steps.jsonl")
    _, rho, cache, scheduler = req[0]["split_group_id"].split("-")

    arrive = np.array([r["enqueue_monotonic_ns"] for r in req]) / NS
    first = np.array([r["first_host_observed_token_ns"] for r in req]) / NS
    finish = np.array([r["finish_monotonic_ns"] for r in req]) / NS
    t0, t1 = arrive[0], arrive[-1]
    window = t1 - t0
    n = len(req)

    # Tokens processed inside the arrival window (steps split by overlap).
    new_in = out = 0.0
    running, waiting = [], []
    for s in steps:
        a, b = s["step_host_start_ns"] / NS, s["step_host_end_ns"] / NS
        share = max(0.0, min(b, t1) - max(a, t0)) / max(b - a, 1e-12)
        new_in += share * s["fresh_prefill_tokens"]
        out += share * s["produced_tokens"]
        if t0 <= a <= t1:
            running.append(s["batch_size"])
            waiting.append(s["queue_before"])

    grid = np.linspace(t0, t1, 200)
    in_sys = _in_system(arrive, finish, grid)
    slope = float(np.polyfit(grid - t0, in_sys, 1)[0])  # requests per second
    offered = (n - 1) / window
    ttft = first - arrive
    q = n // 4
    half = len(waiting) // 2
    return {
        "run": path.parent.name,
        "rho": float(rho.removeprefix("rho")) / 100,
        "cache": cache,
        "scheduler": scheduler,
        "capacity_rps": point["capacity_rps"],
        "offered_rps": offered,
        "served_rps": n / (finish[-1] - t0),
        "done_in_window": int(np.sum(finish <= t1)),
        "offered_itps": sum(r["input_token_count"] for r in req) / window,
        "offered_new_itps": sum(
            r["input_token_count"] - r["prefix_reused_tokens"] for r in req
        )
        / window,
        "offered_otps": sum(r["output_token_count"] for r in req) / window,
        "served_new_itps": new_in / window,
        "served_otps": out / window,
        "growth": slope / offered,  # in-system requests added per arrival
        "in_system_q": [int(in_sys[i]) for i in (50, 100, 150, 199)],
        "in_system_max": int(in_sys.max()),
        "running_mean": mean(running),
        "running_max": max(running),
        "waiting_first_half": mean(waiting[:half]),
        "waiting_second_half": mean(waiting[half:]),
        "ttft_first_q": median(ttft[:q]),
        "ttft_last_q": median(ttft[-q:]),
        "drain_s": float(finish.max() - t1),
        "window_s": float(window),
    }


def run(cfg: DictConfig) -> dict:
    out = Path(HydraConfig.get().runtime.output_dir)
    root = REPO / cfg.runs_root
    paths = sorted(p for p in root.glob(f"*-{cfg.series}-*/*") if p.is_dir())
    rows = [attempt(p) for p in paths]

    groups = defaultdict(list)
    for r in rows:
        groups[(r["scheduler"], r["cache"], r["rho"])].append(r)
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    table = []
    for (scheduler, cache, rho), rs in sorted(groups.items()):
        avg = {k: mean(r[k] for r in rs) for k in keys if k != "rho"}
        avg["stable"] = all(r["growth"] <= cfg.growth_limit for r in rs)
        table.append({"scheduler": scheduler, "cache": cache, "rho": rho, **avg})

    head = (
        f"{'sched':<12}{'cache':<5}{'rho':>5}{'offer':>7}{'serve':>7}"
        f"{'newITPS':>9}{'/serv':>8}{'OTPS':>7}{'/serv':>7}{'growth':>8}"
        f"{'N(t) q1..q4':>18}{'run':>6}{'wait1':>7}{'wait2':>7}"
        f"{'TTFT1':>7}{'TTFT4':>7}{'drain':>7}  verdict"
    )
    lines = [head]
    for t in table:
        lines.append(
            f"{t['scheduler']:<12}{t['cache']:<5}{t['rho']:>5.2f}"
            f"{t['offered_rps']:>7.2f}{t['served_rps']:>7.2f}"
            f"{t['offered_new_itps']:>9.0f}{t['served_new_itps']:>8.0f}"
            f"{t['offered_otps']:>7.0f}{t['served_otps']:>7.0f}"
            f"{t['growth']:>8.2f}"
            f"{' '.join(str(round(x)) for x in _quarters(groups, t)):>18}"
            f"{t['running_mean']:>6.1f}{t['waiting_first_half']:>7.1f}"
            f"{t['waiting_second_half']:>7.1f}{t['ttft_first_q']:>7.2f}"
            f"{t['ttft_last_q']:>7.2f}{t['drain_s']:>7.1f}  "
            + ("справляется" if t["stable"] else "очередь растёт")
        )
    logger.info("S4, среднее по сидам:\n{}", "\n".join(lines))

    path = out / "concurrency.json"
    path.write_text(
        json.dumps({"attempts": rows, "table": table}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    stable = [t for t in table if t["stable"]]
    return {
        "concurrency_json": str(path),
        "attempts": len(rows),
        "points": len(table),
        "stable_points": len(stable),
    }


def _quarters(groups: dict, t: dict) -> list[float]:
    rs = groups[(t["scheduler"], t["cache"], t["rho"])]
    return [mean(r["in_system_q"][i] for r in rs) for i in range(4)]


if __name__ == "__main__":
    main()
