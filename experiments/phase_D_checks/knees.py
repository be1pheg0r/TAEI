"""2. Колено KV закрытого цикла (dynamic): оценка по пулу против трассы.

Оценка — только по пулу запросов (``workload_spec``: L, O, блоки префикса,
семья), блоки целые по 256, ёмкость 186:
- «как в сетке» — ``local_grid.kv_concurrency``: собственные блоки на середине
  decode, среднее по пулу; общие — самый длинный префикс каждой семьи в пуле;
- «+ вес по жизни» — собственные блоки запроса усреднены по его шагам
  (⌈(L + k)/256⌉, k = 0…O−1) и взвешены по времени жизни O; общие — как выше;
- «+ общие у работающих» — Монте-Карло: B запросов с весом O, у каждого
  случайный момент жизни; общие блоки семьи — самый длинный префикс среди её
  работающих запросов. B_KV — B, при котором среднее объединение блоков = 186;
- «+ остаток при приёме» — то же, но ёмкость меньше на половину того, что
  нужно новому запросу (без веса: новые идут в порядке пула). Запрос из головы
  очереди принимается, только если есть свободные блоки на весь его prompt,
  кроме блоков префикса, которые сейчас держат работающие запросы его семьи
  (``can_allocate``); в момент отказа в KV в среднем остаётся около половины.
Факт — средний B decode-шагов окна, перед которыми в очереди кто-то ждёт:
prefill идёт первым, поэтому такой шаг значит, что приём упёрся в KV.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

from common import BLOCK, KV_BLOCKS, ROOT, attempts, jsonl, table, write

sys.path.insert(0, str(ROOT / "src"))
from taei.nano_dataset.local_grid import kv_concurrency  # noqa: E402

DRAWS = 4000


def own_profile(r: dict, on: bool) -> np.ndarray:
    k = np.arange(r["O"])
    return np.ceil((r["L"] + k) / BLOCK) - (r["prefix_blocks"] if on else 0)


def weighted_own(pool, on) -> float:
    return sum(own_profile(r, on).sum() for r in pool) / sum(r["O"] for r in pool)


def longest_shared(pool) -> int:
    longest = {}
    for r in pool:
        if r["family"] is not None:
            longest[r["family"]] = max(longest.get(r["family"], 0), r["prefix_blocks"])
    return sum(longest.values())


def union_blocks(pool, on, B, rng) -> tuple[float, float, float]:
    """Over random running sets: mean union of KV blocks, of shared blocks, and
    the free blocks a new request (uniform over the pool) needs to be admitted.

    nano-vLLM admits a prompt only with free blocks for all of it except the
    prefix blocks now held by running requests (``BlockManager.can_allocate``);
    a cached block nobody holds sits in the free list and needs room too.
    """
    w = np.array([r["O"] for r in pool], float)
    idx = rng.choice(len(pool), size=(DRAWS, B), p=w / w.sum())
    phase = rng.random((DRAWS, B))
    new = rng.choice(len(pool), size=DRAWS)
    profiles = [own_profile(r, on) for r in pool]
    own = np.array(
        [
            [
                profiles[i][int(f * len(profiles[i]))]
                for i, f in zip(row, fr, strict=True)
            ]
            for row, fr in zip(idx, phase, strict=True)
        ]
    ).sum(axis=1)
    shared = np.zeros(DRAWS)
    need = np.array([np.ceil(pool[j]["L"] / BLOCK) for j in new])
    if on:
        fam = [r["family"] for r in pool]
        pref = np.array([r["prefix_blocks"] for r in pool])
        for d, row in enumerate(idx):
            held = {}
            for i in row:
                if fam[i] is not None:
                    held[fam[i]] = max(held.get(fam[i], 0), pref[i])
            shared[d] = sum(held.values())
            j = new[d]
            need[d] -= min(pref[j], held.get(fam[j], 0))
    return float((own + shared).mean()), float(shared.mean()), float(need.mean())


def mc_knee(pool, on, rng, left=False) -> tuple[float, float, float]:
    """B where the mean union (plus half the admission need when ``left``)
    crosses the capacity, linear between integers; shared and need there."""
    prev = None
    for B in range(1, 200):
        u, s, n = union_blocks(pool, on, B, rng)
        level = u + (n / 2 if left else 0)
        if level > KV_BLOCKS:
            if prev is None:
                return float(B), s, n
            (pl, ps, pn) = prev
            x = (KV_BLOCKS - pl) / (level - pl)
            return B - 1 + x, ps + x * (s - ps), pn + x * (n - pn)
        prev = (level, s, n)
    raise ValueError("pool never fills the KV cache")


def fact(attempt, lo, hi) -> tuple[float, float]:
    """Σ T·B and Σ T over decode steps of the window with someone waiting."""
    tb = tt = 0.0
    for s in jsonl(attempt / "steps.jsonl"):
        if (
            s["phase"] == "decode"
            and s["queue_before"] > 0
            and lo <= s["step_host_end_ns"] <= hi
        ):
            t = s["step_host_end_ns"] - s["step_host_start_ns"]
            tb += t * s["actual_batch_size"]
            tt += t
    return tb, tt


def blocked_state(run, attempt) -> list[dict]:
    """Trace check (not part of the estimate): KV and the head of the queue at
    decode steps of the window where admission stopped."""
    on = run["prefix_cache_mode"] == "on"
    loop = run["closed_loop"]
    lo, hi = loop["window_start_ns"], loop["window_end_ns"]
    reqs = {r["request_id"]: r for r in jsonl(attempt / "requests.jsonl")}
    order = sorted(reqs.values(), key=lambda r: r["enqueue_monotonic_ns"])
    rows = []
    for s in jsonl(attempt / "steps.jsonl"):
        if not (
            s["phase"] == "decode"
            and s["queue_before"] > 0
            and lo <= s["step_host_end_ns"] <= hi
        ):
            continue
        ctx, t0 = s["contexts_before"], s["step_host_start_ns"]
        head = next(
            r
            for r in order
            if r["enqueue_monotonic_ns"] <= t0
            and r["request_id"] not in ctx
            and r["finish_monotonic_ns"] > t0
        )
        held = {}
        for rid in ctx:
            f = reqs[rid]["prefix_family_id"]
            if on and f:
                held[f] = max(
                    held.get(f, 0), reqs[rid]["prefix_reused_tokens"] // BLOCK
                )
        need = -(-head["input_token_count"] // BLOCK)
        if on and head["prefix_family_id"]:
            need -= min(
                head["prefix_token_count"] // BLOCK,
                held.get(head["prefix_family_id"], 0),
            )
        rows.append(
            {
                "B": s["actual_batch_size"],
                "used": s["kv_unique_blocks"],
                "per_req": sum(-(-c // BLOCK) for c in ctx.values())
                / s["actual_batch_size"],
                "need": need,
                "blocked": need > KV_BLOCKS - s["kv_unique_blocks"],
            }
        )
    return rows


def main() -> None:
    pools = defaultdict(dict)
    facts = defaultdict(lambda: [0.0, 0.0, 0])
    w3 = defaultdict(list)
    for phase, pattern in [("C", "C-S5-*"), ("D", "D-S5-*"), ("D", "D-S8-*")]:
        for run, attempt in attempts(phase, pattern):
            if run["scheduler_mode"] != "upstream_dynamic":
                continue
            load = run.get("block", run["series"]).replace("S8-", "")
            key = (load, run["prefix_cache_mode"])
            pool = pools[key].get(run["seed"])
            spec = run["workload_spec"]["requests"]
            if pool is None or len(spec) > len(pool):  # D extends the S5 pools
                pools[key][run["seed"]] = spec
            loop = run["closed_loop"]
            tb, tt = fact(attempt, loop["window_start_ns"], loop["window_end_ns"])
            if load == "W3":
                w3[run["prefix_cache_mode"]] += blocked_state(run, attempt)
            f = facts[key]
            f[0] += tb
            f[1] += tt
            f[2] += tt > 0
    rng = np.random.default_rng(0)
    rows = []
    for (load, cache), by_seed in sorted(pools.items()):
        on = cache == "on"
        old, weighted, mc, shared_mc, shared_old, own_mid, own_w = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        left, need = [], []
        for pool in by_seed.values():
            old.append(kv_concurrency(pool, cache, KV_BLOCKS))
            so = longest_shared(pool) if on else 0
            ow = weighted_own(pool, on)
            weighted.append((KV_BLOCKS - so) / ow)
            b, s, _ = mc_knee(pool, on, rng)
            mc.append(b)
            shared_mc.append(s)
            b, _, n = mc_knee(pool, on, rng, left=True)
            left.append(b)
            need.append(n)
            shared_old.append(so)
            own_mid.append((KV_BLOCKS - so) / old[-1])
            own_w.append(ow)
        tb, tt, n = facts[(load, cache)]
        f = tb / tt if tt else None
        rows.append(
            {
                "load": load,
                "cache": cache,
                "old": np.mean(old),
                "weighted": np.mean(weighted),
                "mc": np.mean(mc),
                "left": np.mean(left),
                "need": np.mean(need),
                "fact": f,
                "attempts": n,
                "e_old": 100 * (np.mean(old) / f - 1) if f else None,
                "e_mc": 100 * (np.mean(mc) / f - 1) if f else None,
                "e_left": 100 * (np.mean(left) / f - 1) if f else None,
                "own_mid": np.mean(own_mid),
                "own_w": np.mean(own_w),
                "shared_old": np.mean(shared_old),
                "shared_mc": np.mean(shared_mc),
            }
        )
    diag = [
        {
            "cache": cache,
            "steps": len(rs),
            "B": np.mean([r["B"] for r in rs]),
            "used": np.mean([r["used"] for r in rs]),
            "free": KV_BLOCKS - np.mean([r["used"] for r in rs]),
            "per_req": np.mean([r["per_req"] for r in rs]),
            "pool_per_req": next(
                x for x in rows if x["load"] == "W3" and x["cache"] == "off"
            )["own_w"],
            "need": np.mean([r["need"] for r in rs]),
            "pool_need": next(
                x for x in rows if x["load"] == "W3" and x["cache"] == cache
            )["need"],
            "blocked": 100 * np.mean([r["blocked"] for r in rs]),
        }
        for cache, rs in sorted(w3.items())
    ]
    text = "\n\n".join(
        [
            "# 2. Колено KV: оценка по пулу против трассы",
            f"Ёмкость {KV_BLOCKS} блоков по {BLOCK}; Монте-Карло — {DRAWS} наборов "
            "на B; по сидам — среднее. Факт — средний B (с весом по времени шага) "
            "decode-шагов окна с непустой очередью, по всем N.",
            table(
                rows,
                [
                    ("load", "нагрузка", "s"),
                    ("cache", "кэш", "s"),
                    ("old", "как в сетке", ".1f"),
                    ("weighted", "+ вес по жизни", ".1f"),
                    ("mc", "+ общие у работающих", ".1f"),
                    ("left", "+ остаток при приёме", ".1f"),
                    ("fact", "факт", ".1f"),
                    ("e_old", "ошибка сетки, %", "+.0f"),
                    ("e_mc", "ошибка с весом и общими, %", "+.0f"),
                    ("e_left", "ошибка с остатком, %", "+.0f"),
                    ("attempts", "попыток с упором в KV", "d"),
                ],
            ),
            "## Из чего складывается (блоков на запрос и общих блоков)",
            table(
                rows,
                [
                    ("load", "нагрузка", "s"),
                    ("cache", "кэш", "s"),
                    ("own_mid", "свои, середина decode", ".2f"),
                    ("own_w", "свои, с весом по жизни", ".2f"),
                    ("shared_old", "общие: самый длинный в пуле", ".1f"),
                    ("shared_mc", "общие: у работающих на колене", ".1f"),
                    ("need", "нужно новому на колене, блоков", ".1f"),
                ],
            ),
            "## W3 по трассе: шаги, где приём остановился (диагностика, не оценка)",
            table(
                diag,
                [
                    ("cache", "кэш", "s"),
                    ("steps", "шагов", "d"),
                    ("B", "B", ".2f"),
                    ("used", "занято блоков", ".1f"),
                    ("free", "свободно", ".1f"),
                    ("per_req", "блоков на работающий (без общих)", ".1f"),
                    ("pool_per_req", "то же по пулу с весом", ".1f"),
                    ("need", "нужно голове очереди", ".1f"),
                    ("pool_need", "то же по пулу", ".1f"),
                    ("blocked", "голове не хватает, % шагов", ".0f"),
                ],
            ),
        ]
    )
    write("knees.md", text + "\n")


if __name__ == "__main__":
    main()
