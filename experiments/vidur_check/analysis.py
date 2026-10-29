"""Законы модели времени на открытых профилях отдельных операций Vidur.

Профиль Vidur — время одной операции одного слоя на одном TP-шарде (мс,
колонки ``time_stats.<операция>.median``). Здесь он переводится в секунды и
проверяется теми же законами, что и наша модель: геометрия кэша в attention
prefill, линейная часть (MLP и проекции), отношение b/a к FLOPs и наклон
decode по KV против пропускной способности памяти. Подбор — ``fit_nonneg``
ядра (NNLS на относительной ошибке); ядро не меняется.

Обозначения: C′ = prefill_chunk_size (новые токены), C = kv_cache_size (уже в
KV), L = C + C′, R = C / L; наша модель A берёт b·L²(1 − R²) = 2b·(C·C′ + C′²/2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from taei.nano_time.fit import fit_nonneg, relative_errors, summary

MS = 1e-3
MLP_OPS = (
    "emb",
    "input_layernorm",
    "attn_pre_proj",
    "attn_rope",
    "attn_post_proj",
    "post_attention_layernorm",
    "mlp_up_proj",
    "mlp_act",
    "mlp_down_proj",
    "add",
)
# Prefill attention forms; each also has a free term t0.
FORMS = {
    "G1": ("β·(C·C′ + C′²/2) — наша", lambda c, n: [c * n + n**2 / 2]),
    "G2": ("β·C′·(C + C′) — без ½", lambda c, n: [n * (c + n)]),
    "G3": ("β·C′² — кэш не влияет", lambda c, n: [n**2]),
    "G4": ("β₁·C′² + β₂·C·C′ — свободные", lambda c, n: [n**2, c * n]),
}
CHART_MODEL = "Llama-2-7b-hf"  # on all three GPUs
CHART_CHUNKS = (256, 1024, 2048)
SMALL_N = 64
FLAT = 1.1  # a flat start: time within 10% of the smallest


def profile_files(root: Path) -> list[dict]:
    out = []
    for att in sorted(root.glob("data/profiling/compute/*/*/*/attention.csv")):
        gpu, org, model = att.parts[-4:-1]
        mlp = att.with_name("mlp.csv")
        out.append(
            {"gpu": gpu, "org": org, "model": model, "attention": att, "mlp": mlp}
        )
    return out


def _median_ops(frame: pd.DataFrame) -> list[str]:
    return [c.split(".")[1] for c in frame.columns if c.endswith(".median")]


def dims(att: pd.DataFrame, mlp: pd.DataFrame | None) -> dict:
    row = att.iloc[0]
    hidden, n_q = int(row["n_embd"]), int(row["n_q_head"])
    out = {
        "hidden": hidden,
        "n_q": n_q,
        "n_kv": int(row["n_kv_head"]),
        "d_head": hidden // n_q,
    }
    if mlp is not None and len(mlp):
        out["intermediate"] = int(mlp["n_expanded_embd"].iloc[0])
        out["gated"] = bool(mlp["use_gated_mlp"].iloc[0])
    return out


def layer_flops(d: dict, tp: int) -> float:
    """Linear FLOPs per token of one layer on one shard: QKV, O and MLP."""
    qkv = d["hidden"] * (d["n_q"] + 2 * d["n_kv"]) * d["d_head"]
    out = d["n_q"] * d["d_head"] * d["hidden"]
    mlp = (3 if d["gated"] else 2) * d["hidden"] * d["intermediate"]
    return 2 * (qkv + out + mlp) / tp


def pair_flops(d: dict, tp: int) -> float:
    """Attention FLOPs per (query, key) pair of one layer on one shard: QKᵀ, PV."""
    return 4 * d["n_q"] * d["d_head"] / tp


def kv_bytes(d: dict, tp: int) -> float:
    """KV bytes per cached token of one layer on one shard (K and V, 2 bytes)."""
    return 2 * d["n_kv"] * d["d_head"] * 2 / tp


def _fit(X: np.ndarray, y: np.ndarray):
    coef = fit_nonneg(X, y, "regression")
    return coef, relative_errors(X @ coef, y)


def _design(form: str, c: np.ndarray, n: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(c), *FORMS[form][1](c, n)])


def coverage(att: pd.DataFrame, mlp: pd.DataFrame | None, item: dict) -> dict:
    pf, dc = att[att["is_prefill"]], att[~att["is_prefill"]]
    tp1 = att["num_tensor_parallel_workers"].min()
    # A median of 0 is below the 1 µs resolution: no relative error there.
    zero = lambda f, op: (  # noqa: E731
        int((f[f"time_stats.{op}.median"].fillna(0) <= 0).sum())
        if f"time_stats.{op}.median" in f
        else None
    )
    one = att["num_tensor_parallel_workers"] == tp1
    ops = _median_ops(att)
    mlp_ops = _median_ops(mlp) if mlp is not None else []
    rng = lambda s: [int(s.min()), int(s.max())] if len(s) else None  # noqa: E731
    return {
        "gpu": item["gpu"],
        "model": item["model"],
        "org": item["org"],
        "attention": True,
        "mlp": mlp is not None,
        "tp": sorted(int(t) for t in att["num_tensor_parallel_workers"].unique()),
        "tp_used": int(tp1),
        "backend": sorted(
            str(b).replace("AttentionBackend.", "")
            for b in att["attention_backend"].unique()
        ),
        "attention_ops": ops,
        "kv_cache_save": "attn_kv_cache_save" in ops,
        "prefill_rows": int(len(pf)),
        "prefill_zero": zero(pf[one[pf.index]], "attn_prefill"),
        "decode_zero": zero(dc[one[dc.index]], "attn_decode"),
        "prefill_batch": rng(pf["batch_size"]),
        "chunk": rng(pf["prefill_chunk_size"]),
        "kv_prefill": rng(pf["kv_cache_size"]),
        "decode_rows": int(len(dc)),
        "decode_batch": rng(dc["batch_size"]),
        "kv_decode": rng(dc["kv_cache_size"]),
        "mlp_ops": mlp_ops,
        "mlp_missing": [o for o in MLP_OPS if o not in mlp_ops]
        if mlp is not None
        else None,
        "num_tokens": rng(mlp["num_tokens"]) if mlp is not None else None,
        **dims(att, mlp),
    }


def prefill_forms(att: pd.DataFrame) -> tuple[list[dict], dict]:
    """G1–G4 on attn_prefill: fitted on all points, and on C = 0 only for C > 0."""
    pf = att[att["is_prefill"] & (att["time_stats.attn_prefill.median"] > 0)]
    c = pf["kv_cache_size"].to_numpy(float)
    n = pf["prefill_chunk_size"].to_numpy(float)
    y = pf["time_stats.attn_prefill.median"].to_numpy(float) * MS
    rows, fits = [], {}
    nocache = c == 0
    for form, (text, _) in FORMS.items():
        X = _design(form, c, n)
        coef, rel = _fit(X, y)
        coef0, _ = _fit(X[nocache], y[nocache])
        rel0 = relative_errors(X[~nocache] @ coef0, y[~nocache])
        fits[form] = coef
        row = {
            "form": form,
            "text": text,
            "coef": coef.tolist(),
            **{f"all_{k}": v for k, v in summary(rel).items()},
            **{f"nocache_{k}": v for k, v in summary(rel0).items()},
            "n_c0": int(nocache.sum()),
        }
        if form == "G4":
            # Geometry: β₂ (C·C′) / β₁ (C′²) = 2, as in C·C′ + C′²/2.
            row["ratio"] = coef[2] / coef[1] if coef[1] > 0 else None
            row["ratio_vs_geometry"] = row["ratio"] / 2 if row["ratio"] else None
            row["note_nocache"] = "на C = 0 член C·C′ нулевой: β₂ не определяется"
        rows.append(row)
    return rows, {"c": c, "n": n, "y": y, "coef": fits}


def kv_cache_save(att: pd.DataFrame) -> dict | None:
    col = "time_stats.attn_kv_cache_save.median"
    if col not in att:
        return None
    pf = att[att["is_prefill"] & (att[col] > 0)]
    n = pf["prefill_chunk_size"].to_numpy(float)
    y = pf[col].to_numpy(float) * MS
    coef, rel = _fit(np.column_stack([np.ones_like(n), n]), y)
    return {"t0": coef[0], "k": coef[1], **summary(rel)}


def mlp_check(mlp: pd.DataFrame, d: dict, tp: int, peak: float) -> tuple[dict, dict]:
    ops = [o for o in MLP_OPS if f"time_stats.{o}.median" in mlp]
    frame = mlp.dropna(subset=[f"time_stats.{o}.median" for o in ops])
    y = frame[[f"time_stats.{o}.median" for o in ops]].sum(axis=1).to_numpy() * MS
    frame, y = frame[y > 0], y[y > 0]
    N = frame["num_tokens"].to_numpy(float)
    coef, rel = _fit(np.column_stack([np.ones_like(N), N]), y)
    c, a = coef
    a_flops = layer_flops(d, tp) / (peak * 1e12)
    small = N <= SMALL_N
    first = y[np.argmin(N)]
    flat_to = N[(y <= FLAT * first)].max() if len(N) else None
    return (
        {
            "ops": ops,
            "c": c,
            "a": a,
            **summary(rel),
            "knee_tokens": c / a if a > 0 else None,
            "flat_to": float(flat_to) if flat_to is not None else None,
            "small_mean_abs": float(np.abs(rel[small]).mean()) if small.any() else None,
            "small_mean_signed": float(rel[small].mean()) if small.any() else None,
            "flops_per_token": layer_flops(d, tp),
            "a_flops": a_flops,
            "peak_share": a_flops / a if a > 0 else None,
        },
        {"N": N, "y": y, "pred": np.column_stack([np.ones_like(N), N]) @ coef},
    )


def decode_check(att: pd.DataFrame, d: dict, tp: int, bandwidth: float) -> tuple:
    col = "time_stats.attn_decode.median"
    dc = att[~att["is_prefill"] & (att[col] > 0)]
    B = dc["batch_size"].to_numpy(float)
    kv = dc["kv_cache_size"].to_numpy(float)
    y = dc[col].to_numpy(float) * MS
    X = np.column_stack([np.ones_like(B), B * kv, B])
    coef, rel = _fit(X, y)
    t0, d1, d2 = coef
    d1_mem = kv_bytes(d, tp) / (bandwidth * 1e9)
    # Flat start: medians over log2 bands of B·KV until 10% above the first.
    total = B * kv
    band = np.floor(np.log2(total)).astype(int)
    med = pd.Series(y).groupby(band).median().sort_index()
    above = med[med > FLAT * med.iloc[0]]
    flat_to = float(2.0 ** above.index[0]) if len(above) else None
    return (
        {
            "t0": t0,
            "d1": d1,
            "d2": d2,
            **summary(rel),
            "d1_memory": d1_mem,
            "bandwidth_share": d1_mem / d1 if d1 > 0 else None,
            "knee_sum_l": t0 / d1 if d1 > 0 else None,
            "flat_to_sum_l": flat_to,
        },
        {"B": B, "kv": kv, "y": y, "pred": X @ coef},
    )


def _chart_prefill(fit: dict) -> dict:
    c, n, y = fit["c"], fit["n"], fit["y"]
    chunks = sorted(set(n))
    picked = [min(chunks, key=lambda v: abs(v - target)) for target in CHART_CHUNKS]
    out = []
    for chunk in picked:
        m = n == chunk
        order = np.argsort(c[m])
        cc, yy = c[m][order], y[m][order]
        lines = {
            form: (_design(form, cc, np.full_like(cc, chunk)) @ coef).tolist()
            for form, coef in fit["coef"].items()
        }
        out.append(
            {"chunk": int(chunk), "C": cc.tolist(), "T": yy.tolist(), "forms": lines}
        )
    return {"series": out}


def _chart_mlp(fit: dict) -> dict:
    order = np.argsort(fit["N"])
    return {k: fit[k][order].tolist() for k in ("N", "y", "pred")}


def _chart_decode(fit: dict, n: int = 1500) -> dict:
    idx = np.random.default_rng(0).permutation(len(fit["B"]))[:n]
    return {k: fit[k][idx].tolist() for k in ("B", "kv", "y", "pred")}


def analyze(root: Path, specs: dict, ours: list[dict]) -> dict:
    """Every check for every GPU × model with attention.csv and mlp.csv."""
    gpus = specs["gpus"]
    cov, forms, saves, mlps, decodes, ratios, charts = [], [], [], [], [], [], {}
    for item in profile_files(root):
        att = pd.read_csv(item["attention"])
        mlp = pd.read_csv(item["mlp"]) if item["mlp"].exists() else None
        cov.append(coverage(att, mlp, item))
        if mlp is None:
            continue
        tp = int(att["num_tensor_parallel_workers"].min())
        att1 = att[att["num_tensor_parallel_workers"] == tp]
        mlp1 = mlp[
            mlp["num_tensor_parallel_workers"]
            == mlp["num_tensor_parallel_workers"].min()
        ]
        d = dims(att1, mlp1)
        spec = gpus[item["gpu"]]
        key = {"gpu": item["gpu"], "model": item["model"], "tp": tp}
        rows, pf_fit = prefill_forms(att1)
        forms += [{**key, **r} for r in rows]
        save = kv_cache_save(att1)
        if save:
            saves.append({**key, **save})
        m, m_fit = mlp_check(mlp1, d, tp, spec["bf16_tflops"])
        mlps.append({**key, **m})
        dec, d_fit = decode_check(att1, d, tp, spec["mem_gbs"])
        decodes.append({**key, **dec})
        beta = pf_fit["coef"]["G1"][1]
        b = beta / 2  # our b multiplies L²(1 − R²) = 2(C·C′ + C′²/2)
        ratios.append(
            {
                **key,
                "a": m["a"],
                "b": b,
                "ratio": b / m["a"],
                "ratio_flops": (pair_flops(d, tp) / 2) / layer_flops(d, tp),
            }
        )
        if item["model"] == CHART_MODEL:
            charts[item["gpu"]] = {
                "prefill": _chart_prefill(pf_fit),
                "mlp": _chart_mlp(m_fit),
                "decode": _chart_decode(d_fit),
            }
    for r in ratios:
        r["ratio_vs_flops"] = r["ratio"] / r["ratio_flops"]
    return {
        "coverage": cov,
        "attention_forms": forms,
        "kv_cache_save": saves,
        "mlp": mlps,
        "decode": decodes,
        "ratio": ratios,
        "gpus": _gpu_table(ratios, mlps, decodes),
        "ours": _ours(ours, gpus),
        "a4000": _a4000(ratios, mlps, decodes, gpus, ours),
        "charts": charts,
    }


def _gpu_table(ratios, mlps, decodes) -> list[dict]:
    key = lambda r: (r["gpu"], r["model"])  # noqa: E731
    m = {key(r): r for r in mlps}
    d = {key(r): r for r in decodes}
    return [
        {
            "gpu": r["gpu"],
            "model": r["model"],
            "a": r["a"],
            "b": r["b"],
            "ratio": r["ratio"],
            "ratio_vs_flops": r["ratio_vs_flops"],
            "d1": d[key(r)]["d1"],
            "bandwidth_share": d[key(r)]["bandwidth_share"],
            "mlp_flat_to": m[key(r)]["flat_to"],
            "mlp_knee": m[key(r)]["knee_tokens"],
            "decode_flat_to": d[key(r)]["flat_to_sum_l"],
            "peak_share": m[key(r)]["peak_share"],
        }
        for r in ratios
    ]


def _ours(ours: list[dict], gpus: dict) -> list[dict]:
    """Our fitted b/a and d1 against the same FLOPs and memory estimates."""
    out = []
    for o in ours:
        d, tp, spec = o["dims"], o["tp"], gpus[o["gpu"]]
        dd = {**d, "hidden": d["hidden"]}
        layers = d["layers"]
        a_flops = layers * layer_flops(dd, tp) / (spec["bf16_tflops"] * 1e12)
        d1_mem = layers * kv_bytes(dd, tp) / (spec["mem_gbs"] * 1e9)
        coef = o["coef"]
        out.append(
            {
                "label": o["label"],
                "gpu": o["gpu"],
                "tp": tp,
                "a": coef["a"],
                "b": coef["b"],
                "d1": coef["d1"],
                "ratio": coef["b"] / coef["a"] if coef["a"] else None,
                "ratio_flops": (pair_flops(dd, tp) / 2) / layer_flops(dd, tp),
                "a_flops": a_flops,
                "peak_share": a_flops / coef["a"] if coef["a"] else None,
                "d1_memory": d1_mem,
                "bandwidth_share": d1_mem / coef["d1"] if coef["d1"] else None,
            }
        )
    for r in out:
        r["ratio_vs_flops"] = r["ratio"] / r["ratio_flops"] if r["ratio"] else None
    return out


def _a4000(ratios, mlps, decodes, gpus, ours) -> dict:
    """Llama-2-7b on A40 per layer, scaled to A4000 by datasheet peaks, then to
    Qwen3-8B (36 layers, TP2) by FLOPs and KV bytes; against our fit."""
    pick = lambda rows: next(  # noqa: E731
        r for r in rows if r["gpu"] == "a40" and r["model"] == CHART_MODEL
    )
    r, d = pick(ratios), pick(decodes)
    a40, a4000 = gpus["a40"], gpus["a4000"]
    compute = a40["bf16_tflops"] / a4000["bf16_tflops"]
    memory = a40["mem_gbs"] / a4000["mem_gbs"]
    llama = {"hidden": 4096, "n_q": 32, "n_kv": 32, "d_head": 128,
             "intermediate": 11008, "gated": True}  # fmt: skip
    target = next(o for o in ours if o["label"] == "Qwen3-8B long context")
    q, tp, layers = target["dims"], target["tp"], target["dims"]["layers"]
    per_layer = {
        "a": r["a"] * compute,
        "b": r["b"] * compute,
        "d1": d["d1"] * memory,
    }
    expected = {
        "a": per_layer["a"] * layers * layer_flops(q, tp) / layer_flops(llama, 1),
        "b": per_layer["b"] * layers * pair_flops(q, tp) / pair_flops(llama, 1),
        "d1": per_layer["d1"] * layers * kv_bytes(q, tp) / kv_bytes(llama, 1),
    }
    return {
        "compute_scale": compute,
        "memory_scale": memory,
        "per_layer_a4000": per_layer,
        "expected_qwen3_8b_tp2": expected,
        "ours": {k: target["coef"][k] for k in ("a", "b", "d1")},
        "ours_vs_expected": {k: target["coef"][k] / expected[k] for k in expected},
    }
