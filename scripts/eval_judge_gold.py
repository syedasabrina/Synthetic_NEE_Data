#!/usr/bin/env python
"""
Evaluates the blind rubric judge on the gold set.

Reports, for the judge exactly as BoN uses it:

  - agreement on held-out rows only (gold rows used as few-shot
    demonstrations in the prompt are excluded) and on all rows, so the
    number is comparable with the old 0.661 and also honest
  - per-element exact agreement and per-element x level confusion
  - per-level recall (does the judge ever say 4 when the experts say 4?)
  - a principal-level bootstrap interval for exact agreement, because gold
    rows cluster by principal (24 principals, 6 to 9 rows each) and a row
    bootstrap is too narrow

Per-row predictions and raw judge output are saved so nothing has to be
re-run to look at a failure.

Also the entry point for the judge bake-off: same prompt, different
--judge_model.

    sbatch --partition=gpuq --qos=gpu --time=01:00:00 \
      --export=ALL,SCRIPT=scripts/eval_judge_gold.py,ARGS="--out results/judge_gold_e4b" \
      scripts/train.slurm

Gold is used here only as frozen few-shot demonstrations and as an
evaluation set. No gradients, no training.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

LEVELS = [0, 2, 4]


def qwk(y_true, y_pred) -> float:
    """Quadratic weighted kappa over the 0/2/4 labels."""
    idx = {v: i for i, v in enumerate(LEVELS)}
    k = len(LEVELS)
    obs = np.zeros((k, k))
    for t, p in zip(y_true, y_pred):
        obs[idx[t], idx[p]] += 1
    n = obs.sum()
    if n == 0:
        return float("nan")
    hist_t, hist_p = obs.sum(axis=1), obs.sum(axis=0)
    exp = np.outer(hist_t, hist_p) / n
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float)
    w /= (k - 1) ** 2
    denom = (w * exp).sum()
    if denom == 0:
        return float("nan")
    return 1.0 - (w * obs).sum() / denom


def summarize(df: pd.DataFrame, label: str) -> dict:
    ok = df[df["pred"].notna()]
    y, p = ok["score"].astype(int).to_numpy(), ok["pred"].astype(int).to_numpy()
    out = {
        "subset": label,
        "n": int(len(df)),
        "parsed": int(len(ok)),
        "exact": float((y == p).mean()) if len(ok) else float("nan"),
        "adjacent": float((np.abs(y - p) <= 2).mean()) if len(ok) else float("nan"),
        "signed_dev": float((p - y).mean()) if len(ok) else float("nan"),
        "qwk": float(qwk(y, p)) if len(ok) else float("nan"),
    }
    return out


def principal_bootstrap(df: pd.DataFrame, n_boot: int, seed: int = 0):
    """Percentile CI for exact agreement, resampling principals."""
    ok = df[df["pred"].notna()].copy()
    ok["hit"] = (ok["score"].astype(int) == ok["pred"].astype(int)).astype(float)
    groups = [g["hit"].to_numpy() for _, g in ok.groupby("PersonId")]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(np.concatenate([groups[i] for i in pick]).mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--few_shot_per_cell", type=int, default=1,
                    help="0 evaluates zero-shot (no gold rows in the prompt)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", required=True, help="output prefix, e.g. results/judge_gold_e4b")
    args = ap.parse_args()

    from src.data.corpus import load_gold
    from src.rewards.rubric_reward import RubricReward

    gold = load_gold()
    print(f"Gold rows: {len(gold)}  principals: {gold['PersonId'].nunique()}")

    if args.few_shot_per_cell > 0:
        few_shot, used = RubricReward.build_few_shot_examples(
            gold, max_examples=args.few_shot_per_cell, return_index=True
        )
    else:
        few_shot, used = {}, set()
    print(f"Few-shot cells: {len(few_shot)}  gold rows used as demonstrations: {len(used)}")

    judge = RubricReward(
        model_name=args.judge_model,
        device="cuda",
        few_shot_examples=few_shot,
        batch_size=args.batch_size,
    )

    preds, raws = judge.predict(
        gold["Text"].tolist(), gold["Element_numberX"].tolist(), return_raw=True
    )
    gold = gold.copy()
    gold["pred"] = preds
    gold["raw"] = raws
    gold["is_demo"] = gold.index.isin(used)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gold[["PersonId", "Element_numberX", "score", "pred", "raw", "is_demo"]].to_csv(
        f"{out}_rows.csv", index=False
    )

    held = gold[~gold["is_demo"]]
    results = {
        "judge_model": args.judge_model,
        "few_shot_per_cell": args.few_shot_per_cell,
        "all_rows": summarize(gold, "all"),
        "held_out": summarize(held, "held_out"),
    }
    lo, hi = principal_bootstrap(held, args.n_boot)
    results["held_out"]["exact_ci95_principal_bootstrap"] = [lo, hi]

    print("\n== Agreement ==")
    for k in ("all_rows", "held_out"):
        r = results[k]
        print(f"{r['subset']:<9} n={r['n']:>3} parsed={r['parsed']:>3} "
              f"exact={r['exact']:.3f} adjacent={r['adjacent']:.3f} "
              f"signed={r['signed_dev']:+.3f} qwk={r['qwk']:.3f}")
    print(f"held-out exact 95% CI (principal bootstrap): [{lo:.3f}, {hi:.3f}]")
    print("Old reference: exact 0.661, adjacent 0.988, signed 0.000 (all 171 rows, "
          "demonstrations included, double BOS).")

    print("\n== Per element (held-out) ==")
    per_el = {}
    for el, g in held.groupby("Element_numberX"):
        s = summarize(g, el)
        per_el[el] = s
        print(f"{el:<9} n={s['n']:>3} exact={s['exact']:.3f} signed={s['signed_dev']:+.2f}")
    results["per_element"] = per_el

    print("\n== Confusion, held-out (rows = expert label, cols = judge) ==")
    ok = held[held["pred"].notna()]
    conf = pd.crosstab(ok["score"].astype(int), ok["pred"].astype(int)).reindex(
        index=LEVELS, columns=LEVELS, fill_value=0)
    print(conf.to_string())
    results["confusion_held_out"] = conf.to_dict()

    print("\n== Confusion per element, held-out (label->pred counts) ==")
    per_el_conf = {}
    for el, g in ok.groupby("Element_numberX"):
        c = pd.crosstab(g["score"].astype(int), g["pred"].astype(int)).reindex(
            index=LEVELS, columns=LEVELS, fill_value=0)
        per_el_conf[el] = c.to_dict()
        print(f"\n{el}")
        print(c.to_string())
    results["confusion_per_element"] = per_el_conf

    print("\n== Recall by expert level (held-out) ==")
    for lv in LEVELS:
        g = ok[ok["score"].astype(int) == lv]
        if len(g):
            print(f"expert {lv}: n={len(g):>3} judge agrees {np.mean(g['pred'].astype(int) == lv):.3f}")

    unparsed = held[held["pred"].isna()]
    if len(unparsed):
        print(f"\n{len(unparsed)} unparsed held-out rows; first raw outputs:")
        for r in unparsed["raw"].head(5):
            print("  ", repr(r))

    with open(f"{out}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out}_rows.csv and {out}.json")


if __name__ == "__main__":
    main()
