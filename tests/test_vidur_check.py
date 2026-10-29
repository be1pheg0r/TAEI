"""Проверяет законы на профилях Vidur на синтетике и встраивание ветки в отчёт.

Подбор форм attention восстанавливает геометрию G1 (β₂/β₁ = 2) и переносится с
C = 0 на C > 0, FLOPs слоя Llama-2-7b совпадают с ручным расчётом, а ветка с
``kind: "vidur"`` не ломает индекс запусков.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.vidur_check import analysis  # noqa: E402

LLAMA = {"hidden": 4096, "n_q": 32, "n_kv": 32, "d_head": 128, "intermediate": 11008}


def _attention(beta=2e-6, t0=0.01) -> pd.DataFrame:
    rows = []
    for n in (64, 128, 256, 512, 1024, 2048):
        for c in (0, 128, 512, 1024, 2048):
            rows.append(
                {
                    "is_prefill": True,
                    "prefill_chunk_size": n,
                    "kv_cache_size": c,
                    "time_stats.attn_prefill.median": t0 + beta * (c * n + n * n / 2),
                }
            )
    return pd.DataFrame(rows)


def test_forms_recover_the_cache_geometry():
    rows, fit = analysis.prefill_forms(_attention())
    by = {r["form"]: r for r in rows}
    assert by["G1"]["all_mean_abs"] < 1e-6
    assert by["G1"]["nocache_mean_abs"] < 1e-6  # C = 0 alone fixes β
    assert by["G4"]["ratio"] == pytest.approx(2.0, rel=1e-4)
    assert by["G2"]["nocache_mean_abs"] > 0.05 and by["G3"]["all_mean_abs"] > 0.05
    assert fit["coef"]["G1"][1] == pytest.approx(2e-6 * analysis.MS, rel=1e-4)


def test_layer_flops_and_kv_bytes_of_llama_2_7b():
    d = {**LLAMA, "gated": True}
    per_token = 2 * (4096 * 96 * 128 + 4096 * 4096 + 3 * 4096 * 11008)
    assert analysis.layer_flops(d, 1) == per_token
    assert analysis.layer_flops(d, 2) == per_token / 2
    assert analysis.pair_flops(d, 1) == 4 * 4096
    assert analysis.kv_bytes(d, 1) == 2 * 32 * 128 * 2


def test_decode_fit_recovers_the_kv_slope():
    B, kv = np.meshgrid([1, 2, 4, 8, 16], [32, 256, 1024, 4096])
    B, kv = B.ravel(), kv.ravel()
    att = pd.DataFrame(
        {
            "is_prefill": False,
            "batch_size": B,
            "kv_cache_size": kv,
            "time_stats.attn_decode.median": 0.01 + 1e-5 * B * kv + 1e-3 * B,
        }
    )
    out, _ = analysis.decode_check(att, {**LLAMA, "gated": True}, 1, 696)
    assert out["d1"] == pytest.approx(1e-5 * analysis.MS, rel=1e-4)
    assert out["mean_abs"] < 1e-6


def test_vidur_branch_is_skipped_by_the_run_index(tmp_path):
    from taei.nano_time import report

    run = tmp_path / "20260101_000000"
    run.mkdir()
    doc = {
        "schema_version": 2,
        "meta": {},
        "models": [{"kind": "vidur", "label": "Vidur"}],
    }
    (run / "time_model.json").write_text(json.dumps(doc), encoding="utf-8")
    entry = report.index_entry(run / "time_model.json", tmp_path)
    assert entry["models"] == []
