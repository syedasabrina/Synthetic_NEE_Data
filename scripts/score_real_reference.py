#!/usr/bin/env python
"""
Calibrates the two absolute thresholds that selection needs.

1. Text-quality gate (CPU). Computes the gate metrics on every real BIP in
   the training corpus and prints their quantiles and the share of real
   BIPs the current QualityGate would reject. Thresholds that reject a
   large share of real writing are wrong, because the synthetic pool would
   then be pushed away from how principals write. Pick caps near the real
   p95 to p99.

2. Authenticity floor (GPU, --auth). Scores held-out real text with the
   same function BoN uses, with and without the "ElementN:" prefix.
   Held-out here means the gold principals, who are excluded from
   BIPDomainSFT training (corpus.load drops them). The old calibration
   scored prefixed probes against unprefixed candidates and the domain
   model had trained on every real BIP, so no like-for-like reference
   existed. The printed delta quantiles are the source for
   --min_auth_delta.

    sbatch --partition=gpuq --qos=gpu --time=01:00:00 \
      --export=ALL,SCRIPT=scripts/score_real_reference.py,ARGS="--auth --out results/real_reference.json" \
      scripts/train.slurm
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.corpus import load, load_gold
from src.utils.text_quality import QualityGate, text_metrics

QS = [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def quantiles(x) -> dict:
    x = np.asarray([v for v in x if v == v], dtype=float)
    return {f"p{int(q * 100)}": float(np.quantile(x, q)) for q in QS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/bips.csv")
    ap.add_argument("--auth", action="store_true", help="also score authenticity on gold (GPU)")
    ap.add_argument("--out", default="results/real_reference.json")
    args = ap.parse_args()

    result = {}

    # ── 1. gate metrics on real training BIPs ───────────────────────
    df = load(args.data)
    metrics = pd.DataFrame([text_metrics(t) for t in df["Text"].fillna("")])
    metrics["element"] = df["Element_numberX"].to_numpy()
    print(f"Real BIPs: {len(metrics):,}")

    gate = QualityGate()
    print(f"\nCurrent gate: {gate}")
    gate_fail = [bool(gate.check(t, stopped=None)) for t in df["Text"].fillna("")]
    metrics["gate_fail"] = gate_fail
    print(f"Share of real BIPs the gate would reject: {np.mean(gate_fail):.3f}")
    print(metrics.groupby("element")["gate_fail"].mean().round(3).to_string())

    cols = ["marks", "rt", "tc", "last_marks", "last_rt", "last_tc"]
    print("\nQuantiles on real BIPs (whole text and last 100 words)")
    print(f"{'metric':<12}" + "".join(f"{('p' + str(int(q * 100))):>8}" for q in QS))
    result["real_quality"] = {}
    for c in cols:
        qd = quantiles(metrics[c])
        result["real_quality"][c] = qd
        print(f"{c:<12}" + "".join(f"{qd[k]:>8.2f}" for k in qd))
    result["real_ends_punct"] = float(metrics["ends_punct"].mean())
    result["real_gate_reject_share"] = float(np.mean(gate_fail))
    print(f"\nends on punctuation: {result['real_ends_punct']:.3f}")
    print("Reading: max_rt and max_title_case should sit near the real p95 to p99 of "
          "rt / last_rt and tc / last_tc; min_marks near the real p5 to p25 of marks.")

    # ── 2. authenticity on held-out real text ───────────────────────
    if args.auth:
        from src.rewards.authenticity_reward import AuthenticityReward

        gold = load_gold().drop_duplicates(subset=["Text"]).reset_index(drop=True)
        texts = gold["Text"].tolist()
        els = gold["Element_numberX"].tolist()
        print(f"\nGold texts (held out from BIPDomainSFT): {len(texts)}")

        auth = AuthenticityReward(device="cuda")
        with_prefix = auth.score_candidates(texts, elements=els, batch_size=4)
        no_prefix = auth.score_candidates(texts, elements=None, batch_size=4)

        result["auth_gold_with_prefix"] = {
            "delta": quantiles(with_prefix["delta"]),
            "reward_abs": quantiles(with_prefix["reward_abs"]),
            "nll_finetuned": quantiles(with_prefix["nll_finetuned"]),
        }
        result["auth_gold_no_prefix"] = {
            "delta": quantiles(no_prefix["delta"]),
            "reward_abs": quantiles(no_prefix["reward_abs"]),
            "nll_finetuned": quantiles(no_prefix["nll_finetuned"]),
        }
        print(f"\n{'':<22}" + "".join(f"{('p' + str(int(q * 100))):>8}" for q in QS))
        for name, key in (("delta, prefixed", "auth_gold_with_prefix"),
                          ("delta, bare", "auth_gold_no_prefix")):
            qd = result[key]["delta"]
            print(f"{name:<22}" + "".join(f"{qd[k]:>8.3f}" for k in qd))
        print("\nThe gap between the two rows is the size of the prefix confound in the "
              "old scores. Use the prefixed row (BoN now scores with the prefix). "
              "A --min_auth_delta near its p5 to p25 keeps the floor below most real text.")
        print("Old accepted pools: auth_reward_abs median 0.196, p90 0.267 (bare, in-loop).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
