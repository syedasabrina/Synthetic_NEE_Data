#!/usr/bin/env python
"""
Sweeps the text-quality gate thresholds and reports, for each setting:

  - the share of REAL BIPs it would reject (overall, worst element, and by
    which criterion), and
  - the share of the OLD degraded pools (rounds 1 to 4) it would let through.

Pick the setting that rejects few real BIPs and passes little of the old
pool. A gate that rejects real writing pushes the synthetic set toward
"cleaner than real", which test (b) can detect. The old pools are only a
proxy for garbage: a rebuilt generator will fail in subtler ways, so the
gate should catch gross outliers and leave fine ranking to the judge and
the authenticity model.

CPU only, about a minute:

    python scripts/sweep_gate.py --out results/gate_sweep.json
"""

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.corpus import load
from src.utils.text_quality import text_metrics

RT_CAPS = [2, 3, 5, 7, 10]
TC_CAPS = [0.40, 0.50, 0.60, 0.70]
MARK_MINS = [3.0, 2.5, 2.0, 1.5, 1.0]


def metric_frame(texts) -> pd.DataFrame:
    M = pd.DataFrame([text_metrics(t) for t in texts])
    return pd.DataFrame({
        "rt": np.maximum(M["rt"], M["last_rt"]),
        "tc": np.maximum(M["tc"], M["last_tc"]),
        "marks": np.minimum(M["marks"], M["last_marks"]),
    })


def fails(M: pd.DataFrame, rt: float, tc: float, mk: float) -> dict:
    return {
        "rt": (M["rt"] > rt).to_numpy(),
        "tc": (M["tc"] > tc).to_numpy(),
        "marks": (M["marks"] < mk).to_numpy(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/bips.csv")
    ap.add_argument("--pools", default="models/BoN_round*/accepted.jsonl")
    ap.add_argument("--max_real_reject", type=float, default=0.05)
    ap.add_argument("--max_element_reject", type=float, default=0.10)
    ap.add_argument("--out", default="results/gate_sweep.json")
    args = ap.parse_args()

    real = load(args.data)
    real_M = metric_frame(real["Text"].fillna(""))
    elements = real["Element_numberX"].to_numpy()

    pool_files = sorted(glob.glob(args.pools))
    pool_texts = []
    for p in pool_files:
        pool_texts += pd.read_json(p, lines=True)["completion"].tolist()
    pool_M = metric_frame(pool_texts) if pool_texts else None
    print(f"Real BIPs: {len(real_M):,}   old pool rows: {len(pool_texts):,} "
          f"from {len(pool_files)} files")

    # ── marginals: what each criterion alone does to real text ───────
    print("\nReal BIPs rejected by one criterion alone")
    print(f"{'criterion':<18}{'value':>7}{'overall':>9}   worst element")
    for name, vals, key in (("run-together cap", RT_CAPS, "rt"),
                            ("Title Case cap", TC_CAPS, "tc"),
                            ("min marks/100", MARK_MINS, "marks")):
        for v in vals:
            f = fails(real_M, v if key == "rt" else 1e9,
                      v if key == "tc" else 1e9,
                      v if key == "marks" else -1e9)[key]
            by_el = pd.Series(f).groupby(elements).mean()
            print(f"{name:<18}{v:>7}{f.mean():>9.3f}   "
                  f"{by_el.idxmax()} {by_el.max():.3f}")

    # ── full grid ───────────────────────────────────────────────────
    rows = []
    for rt, tc, mk in itertools.product(RT_CAPS, TC_CAPS, MARK_MINS):
        f = fails(real_M, rt, tc, mk)
        any_fail = f["rt"] | f["tc"] | f["marks"]
        by_el = pd.Series(any_fail).groupby(elements).mean()
        row = {
            "max_rt": rt, "max_tc": tc, "min_marks": mk,
            "real_reject": float(any_fail.mean()),
            "worst_element": by_el.idxmax(),
            "worst_element_reject": float(by_el.max()),
            "real_rejected_by": {k: float(v.mean()) for k, v in f.items()},
        }
        if pool_M is not None:
            pf = fails(pool_M, rt, tc, mk)
            row["old_pool_pass"] = float(1 - (pf["rt"] | pf["tc"] | pf["marks"]).mean())
        rows.append(row)
    grid = pd.DataFrame(rows)

    ok = grid[(grid["real_reject"] <= args.max_real_reject)
              & (grid["worst_element_reject"] <= args.max_element_reject)]
    print(f"\nSettings with real reject <= {args.max_real_reject:.0%} overall and "
          f"<= {args.max_element_reject:.0%} in every element "
          f"({len(ok)} of {len(grid)}), best old-pool rejection first")
    cols = ["max_rt", "max_tc", "min_marks", "real_reject",
            "worst_element", "worst_element_reject", "old_pool_pass"]
    if len(ok):
        print(ok.sort_values("old_pool_pass").head(10)[cols].to_string(index=False,
              float_format=lambda x: f"{x:.3f}"))
    else:
        print("none; relax --max_real_reject / --max_element_reject, or the gate "
              "is not separable on these three metrics")

    print("\nCurrent defaults for reference (rt 2, tc 0.40, marks 3.0)")
    cur = grid[(grid.max_rt == 2) & (grid.max_tc == 0.40) & (grid.min_marks == 3.0)]
    print(cur[cols + ["real_rejected_by"]].to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_json(out, orient="records", indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()