"""Coefficients for phase B: formula 1 and 3 fitted on S1 + S2 of phase A.

Usage: uv run --extra analysis python experiments/nano_vllm_qwen3_0_6b_local/
fit_phase_a.py <phase A runs dir>. Prints ``dataset.coef`` for the phase B
launch and the reference-run noise (spread of identical runs).
"""

from __future__ import annotations

import json
import sys

import numpy as np

from taei.nano_time import data
from taei.nano_time.decode import DecodeModel
from taei.nano_time.prefill import PrefillModel


def main(root: str) -> None:
    ds = data.load(root, "Qwen3-0.6B local")
    series = ds.runs.set_index("run_id")["set_id"]
    pf = ds.prefill_steps[ds.prefill_steps["run_id"].map(series) == "S1"]
    dc = ds.decode_steps[ds.decode_steps["run_id"].map(series) == "S2"]
    prefill = PrefillModel().fit(pf, "regression")
    decode = DecodeModel().fit(dc, "regression")
    for name, model, steps in (("prefill", prefill, pf), ("decode", decode, dc)):
        rel = model.predict_steps(steps) / steps["T"].to_numpy() - 1
        print(f"{name}: {len(steps)} steps, mean |e| {np.abs(rel).mean():.3f}")
    ref = ds.runs[ds.runs["run_id"].str.contains("-REF-")]
    wall = ref["wall_seconds"]
    print(
        f"REF: {len(ref)} runs, wall {wall.mean():.3f} s, "
        f"std/mean {wall.std() / wall.mean():.4f}, "
        f"SM {ref['gpu_sm_mhz'].min():.0f}-{ref['gpu_sm_mhz'].max():.0f} MHz"
    )
    print(json.dumps({**prefill.coef, **decode.coef}))


if __name__ == "__main__":
    main(sys.argv[1])
