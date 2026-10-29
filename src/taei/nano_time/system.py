"""Модель системы «от N»: прогноз закрытого цикла по тому, что известно до запуска.

Вход — N (запросов в системе одновременно), пул запросов (L, O, семья и длина
префикса), кэш, планировщик, параметры движка (блоки KV, размер блока,
``max_num_seqs``, ``max_num_batched_tokens``, волна fixed_batch) и модель шага:
модель A для prefill и вариант decode (``decode_variants``). Выход — RPS, ITPS
(новые, полный вход, из кэша), OTPS, TTFT p50 и p90, TPOT и TTPS. Ни одна
величина трассы во вход не идёт.

Работающие запросы (upstream_dynamic). Запрос держит место в KV всё время
жизни, и время жизни растёт с O, поэтому состав работающих берётся с весом O:
набор из B запросов пула, у каждого случайный момент жизни k = 0…O − 1 (свои
блоки ⌈(L + k)/256⌉ без префикса из кэша). Общие блоки семьи — самый длинный
префикс среди её работающих. Новый запрос из головы очереди принимается, только
если есть свободные блоки на весь его prompt, кроме блоков префикса, которые
сейчас держат работающие (``BlockManager.can_allocate``), поэтому в момент
отказа в KV в среднем свободна половина того, что ему нужно. Колено B_KV —
B, при котором среднее «объединение блоков + половина нужного новому» равно
числу блоков. По тем же наборам — Σℓ и U (токены KV без повторов префикса).

Формулы (dynamic):
- B = min(N, B_KV, max_num_seqs); T_dc = decode(B, Σℓ(B), U(B));
- prefill на запрос T_pf = a·L(1 − R) + b·L²(1 − R²) + c/k, где k = max(1, B/Ō) —
  новых запросов на prefill-шаг (столько завершается за decode-шаг);
- τ = T̄_pf + (Ō − 1)·T_dc / B — время GPU на запрос (первый токен даёт prefill),
  RPS = 1/τ; ITPS новые = RPS·Ī_new, полный = RPS·L̄, OTPS = RPS·Ō;
- TPOT = B / (RPS·(Ō − 1)): по закону Литтла B работающих = RPS × время decode
  запроса, (Ō − 1)·TPOT; в TPOT входят и prefill-шаги новых запросов;
- TTFT запроса = его prefill-шаг (k запросов) + очередь max(0, N − B)/RPS.
  Новый запрос подаётся между шагами, поэтому ждать текущий шаг ему не нужно.

Когорты: если у всех запросов пула одинаковые L и O (W1), при N ≤ B_KV они
завершаются и подаются разом. Когорта из N проходит prefill-шаги по бюджету
токенов, затем O − 1 decode-шагов при B = N; TTFT — до своего prefill-шага.

fixed_batch: волна — до ``wave`` запросов из очереди, подряд по пулу. Волна
проходит prefill и decode до самого длинного O (B убывает); завершившийся
запрос сразу подаёт следующий, и тот ждёт конца волны. Волны считаются подряд
по пулу с предсказанными шагами, метрики — в окне после прогрева.

Прямая бюджета на точку: ã = T̄_pf / Ī_new, b̃ = (τ − T̄_pf) / Ō; тогда
ã·ITPS + b̃·OTPS = 1, а точка модели — пересечение прямой с лучом
OTPS/ITPS = Ō/Ī_new.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .decode_variants import DecodeVariant


@dataclass(frozen=True)
class Engine:
    kv_blocks: int = 186
    block_size: int = 256
    max_num_seqs: int = 256
    max_batched_tokens: int = 16384
    wave: int = 8


@dataclass
class StepModel:
    prefill: dict[str, float]  # a, b, c of model A
    decode: DecodeVariant

    def prefill_step(self, new: float, quad: float, steps: float = 1.0) -> float:
        p = self.prefill
        return p["a"] * new + p["b"] * quad + p["c"] * steps


@dataclass
class Pool:
    """Requests of a workload in submission order; C — prefix tokens from the cache."""

    L: np.ndarray
    out: np.ndarray  # output tokens O
    prefix: np.ndarray  # prefix blocks served by the cache (0 with the cache off)
    family: np.ndarray  # family code, −1 without a cached prefix
    block: int

    @classmethod
    def from_spec(cls, requests: list[dict], cache: str, block: int = 256) -> Pool:
        on = cache == "on"
        names: dict[str, int] = {}
        fam, pre = [], []
        for r in requests:
            shared = (
                on and r.get("family") is not None and r.get("prefix_blocks", 0) > 0
            )
            fam.append(names.setdefault(r["family"], len(names)) if shared else -1)
            pre.append(r["prefix_blocks"] if shared else 0)
        return cls(
            np.array([r["L"] for r in requests], float),
            np.array([r["O"] for r in requests], float),
            np.array(pre, float),
            np.array(fam, int),
            block,
        )

    @property
    def C(self) -> np.ndarray:
        return self.prefix * self.block

    @property
    def new(self) -> np.ndarray:
        return self.L - self.C

    @property
    def quad(self) -> np.ndarray:
        return self.L**2 - self.C**2

    @property
    def uniform(self) -> bool:
        return bool(np.ptp(self.L) == 0 and np.ptp(self.out) == 0)


@dataclass
class Running:
    """Means over random running sets of B = 1, 2, … requests."""

    union: np.ndarray  # KV blocks held
    need: np.ndarray  # free blocks a new request needs
    sum_l: np.ndarray
    sum_u: np.ndarray
    shared: np.ndarray  # blocks of shared prefixes held

    def at(self, B: float) -> dict[str, float]:
        """Linear between integer B (index 0 is B = 1)."""
        x = min(max(B, 1.0), float(len(self.union)))
        lo = int(math.floor(x)) - 1
        hi = min(lo + 1, len(self.union) - 1)
        f = x - (lo + 1)
        mix = lambda v: float(v[lo] + f * (v[hi] - v[lo]))  # noqa: E731
        return {
            k: mix(getattr(self, k))
            for k in ("union", "need", "sum_l", "sum_u", "shared")
        }


def _sets(pool: Pool, B: int, draws: int, rng) -> tuple[float, ...]:
    w = pool.out / pool.out.sum()
    idx = rng.choice(len(pool.L), size=(draws, B), p=w)
    k = np.floor(rng.random((draws, B)) * pool.out[idx])
    L, p, fam = pool.L[idx], pool.prefix[idx], pool.family[idx]
    own = np.ceil((L + k) / pool.block) - p
    new = rng.integers(len(pool.L), size=draws)
    need = np.ceil(pool.L[new] / pool.block)
    held = np.zeros(draws)
    repeats = np.zeros(draws)
    for f in range(pool.family.max() + 1):
        mine = fam == f
        pf = np.where(mine, p, 0.0)
        top = pf.max(axis=1)
        held += top
        repeats += (pf.sum(axis=1) - top) * pool.block
        need -= np.where(pool.family[new] == f, np.minimum(pool.prefix[new], top), 0.0)
    sum_l = (L + k).sum(axis=1)
    return (
        float((own.sum(axis=1) + held).mean()),
        float(need.mean()),
        float(sum_l.mean()),
        float((sum_l - repeats).mean()),
        float(held.mean()),
    )


def running(pool: Pool, engine: Engine, draws: int = 2000, seed: int = 0) -> Running:
    """Running-set means for B = 1 … up to the KV knee (+1) or ``max_num_seqs``."""
    rng = np.random.default_rng(seed)
    rows = []
    for B in range(1, engine.max_num_seqs + 1):
        rows.append(_sets(pool, B, draws, rng))
        if rows[-1][0] + rows[-1][1] / 2 > engine.kv_blocks:
            break
    cols = np.array(rows).T
    return Running(*cols)


def kv_knee(run: Running, engine: Engine) -> float:
    """B where the mean union plus half the admission need reaches the KV blocks."""
    level = run.union + run.need / 2
    above = np.nonzero(level > engine.kv_blocks)[0]
    if not len(above):
        return float(min(len(level), engine.max_num_seqs))
    i = int(above[0])
    if i == 0:
        return 1.0
    return i + (engine.kv_blocks - level[i - 1]) / (level[i] - level[i - 1])


def _quantiles(values, weights=None) -> tuple[float, float]:
    v = np.asarray(values, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    order = np.argsort(v)
    cum = np.cumsum(w[order]) / w.sum()
    pick = lambda q: float(v[order][np.searchsorted(cum, q)])  # noqa: E731
    return pick(0.5), pick(0.9)


def _finish(pool: Pool, N, B, knee, rps, P, ttft, tpot, branch, T_dc) -> dict:
    """Common outputs: rates from RPS and the pool means, the budget line."""
    p50, p90 = ttft
    new, L, o = float(pool.new.mean()), float(pool.L.mean()), float(pool.out.mean())
    tau = 1 / rps
    return {
        "N": float(N),
        "B": float(B),
        "B_kv": float(knee),
        "above": bool(knee < N),
        "branch": branch,
        "rps": rps,
        "itps_new": rps * new,
        "itps_full": rps * L,
        "itps_cache": rps * (L - new),
        "otps": rps * o,
        "all_tokens": rps * (L + o),
        "ttft_p50": p50,
        "ttft_p90": p90,
        "tpot": tpot,
        "ttps": 1 / tpot,
        "queue": max(0.0, N - B) / rps,
        "T_dc": T_dc,
        "T_pf": P,
        "a_tilde": P / new if new else None,
        "a_tilde_full": P / L,
        "b_tilde": (tau - P) / o,
    }


def predict_dynamic(
    pool: Pool, engine: Engine, step: StepModel, N: float, run: Running
) -> dict:
    knee = kv_knee(run, engine)
    B = min(float(N), knee, float(engine.max_num_seqs))
    budget = engine.max_batched_tokens
    if pool.uniform and knee >= N:
        return _cohort(pool, engine, step, int(round(N)), run, knee)
    s = run.at(B)
    T_dc = float(step.decode.predict(B, s["sum_l"], s["sum_u"]))
    o = float(pool.out.mean())
    new, quad = pool.new, pool.quad
    k = min(max(1.0, B / o), max(1.0, budget / float(new.mean())))
    P = (
        float((step.prefill["a"] * new + step.prefill["b"] * quad).mean())
        + step.prefill["c"] / k
    )
    rps = 1 / (P + (o - 1) * T_dc / B)
    tpot = B / (rps * (o - 1)) if o > 1 else T_dc
    own_step = step.prefill_step(
        new + (k - 1) * new.mean(), quad + (k - 1) * quad.mean()
    )
    queue = max(0.0, N - B) / rps
    return _finish(
        pool, N, B, knee, rps, P, _quantiles(own_step + queue), tpot, "dynamic", T_dc
    )


def _cohort(
    pool: Pool, engine: Engine, step: StepModel, N: int, run: Running, knee
) -> dict:
    new, quad, o = float(pool.new[0]), float(pool.quad[0]), float(pool.out[0])
    per_step = max(1, int(engine.max_batched_tokens // max(new, 1.0)))
    counts = [per_step] * (N // per_step) + ([N % per_step] if N % per_step else [])
    times = np.array([step.prefill_step(n * new, n * quad) for n in counts])
    cum = np.cumsum(times)
    s = run.at(N)
    T_dc = float(step.decode.predict(N, s["sum_l"], s["sum_u"]))
    cycle = float(cum[-1]) + (o - 1) * T_dc
    rps = N / cycle
    tpot_j = ((cum[-1] - cum) + (o - 1) * T_dc) / max(o - 1, 1)
    tpot = _quantiles(tpot_j, counts)[0]
    return _finish(
        pool,
        N,
        N,
        knee,
        rps,
        float(cum[-1]) / N,
        _quantiles(cum, counts),
        tpot,
        "cohort",
        T_dc,
    )


def predict_fixed(
    pool: Pool,
    engine: Engine,
    step: StepModel,
    N: int,
    warmup: int = 20,
    measure: int = 200,
) -> dict:
    """Waves in pool order with predicted steps; metrics in the window after warmup."""
    w = min(int(N), engine.wave)
    queue = deque((i, 0.0) for i in range(min(int(N), len(pool.L))))
    nxt, t, done = len(queue), 0.0, []
    pf_time = 0.0
    budget = engine.max_batched_tokens
    while queue and len(done) < warmup + measure:
        members = [queue.popleft() for _ in range(min(w, len(queue)))]
        ids = np.array([m[0] for m in members])
        first = {}
        packed, start = 0.0, 0
        for j, i in enumerate(ids):  # prefill steps by the token budget, in order
            if packed + pool.new[i] > budget and j > start:
                block = ids[start:j]
                dt = step.prefill_step(pool.new[block].sum(), pool.quad[block].sum())
                t += dt
                pf_time += dt
                first.update({b: t for b in block})
                packed, start = 0.0, j
            packed += pool.new[i]
        block = ids[start:]
        dt = step.prefill_step(pool.new[block].sum(), pool.quad[block].sum())
        t += dt
        pf_time += dt
        first.update({b: t for b in block})
        finish = {i: t for i in ids if pool.out[i] <= 1}
        for s in range(1, int(pool.out[ids].max())):
            live = ids[pool.out[ids] > s]
            ctx = pool.L[live] + s
            fam, pre = pool.family[live], pool.prefix[live] * pool.block
            rep = sum(
                pre[fam == f].sum() - pre[fam == f].max() for f in set(fam[fam >= 0])
            )
            t += float(step.decode.predict(len(live), ctx.sum(), ctx.sum() - rep))
            for i in live[pool.out[live] == s + 1]:
                finish[i] = t
        submit = dict(members)
        for i in sorted(finish, key=finish.get):
            done.append((i, submit[i], first[i], finish[i]))
            if nxt < len(pool.L):
                queue.append((nxt, finish[i]))
                nxt += 1
    done.sort(key=lambda r: r[3])
    lo = done[warmup - 1][3] if warmup else 0.0
    hi = done[min(warmup + measure, len(done)) - 1][3]
    inside = [r for r in done if lo < r[3] <= hi]
    rps = len(inside) / (hi - lo)
    ttft = [r[2] - r[1] for r in inside]
    tpots = [(r[3] - r[2]) / (pool.out[r[0]] - 1) for r in inside if pool.out[r[0]] > 1]
    P = pf_time / len(done)
    out = _finish(
        pool,
        N,
        w,
        float(engine.wave),
        rps,
        P,
        _quantiles(ttft),
        float(np.median(tpots)),
        "fixed",
        None,
    )
    out["queue"] = None
    return out


def predict(
    pool: Pool,
    engine: Engine,
    step: StepModel,
    N: float,
    scheduler: str,
    run: Running | None = None,
) -> dict:
    if scheduler == "fixed_batch":
        return predict_fixed(pool, engine, step, int(N))
    return predict_dynamic(pool, engine, step, N, run or running(pool, engine))


def budget_point(
    L: float,
    R: float,
    out: float,
    N: float,
    step: StepModel,
    engine: Engine | None = None,
) -> tuple[float, float]:
    """(ITPS new, OTPS) of a workload of identical requests (L, R, O = out) at N."""
    engine = engine or Engine()
    blocks = int(R * L // engine.block_size)
    spec = [
        {"L": L, "O": out, "family": "f0" if blocks else None, "prefix_blocks": blocks}
    ]
    pool = Pool.from_spec(spec, "on" if blocks else "off", engine.block_size)
    p = predict_dynamic(pool, engine, step, N, running(pool, engine, draws=200))
    return p["itps_new"], p["otps"]
