#!/usr/bin/env python
"""
Scores a trained assessor on the held-out gold rows and compares it with
other scorers on exactly the same rows.

Held-out means the gold rows that are NOT few-shot demonstrations in the
judge's prompt. Those demonstrations influenced which synthetic BIPs were
kept, so testing on them would leak. The demonstration rows are rebuilt
with the same call BoN selection uses (one example per element and level),
so the split matches the data the assessor was trained on.

Reports, for held-out rows:
  - exact agreement, adjacent agreement, signed deviation, QWK, recall by level
  - principal-level 95% intervals for exact agreement and QWK
  - the always-answer-2 baseline
  - the same numbers without half-point gold rows (side check)
  - paired differences against other scorers' row files, with intervals

Per-row predictions are saved, so any scorer's file can be compared later
without rerunning it.

Run a trained assessor (GPU):
    python scripts/eval_assessor_gold.py \
        --adapter models/Assessor_synthetic_lr1e-4 --label synthetic_lr1e-4 \
        --out results/assessor_synthetic_lr1e-4 \
        --compare results/judge_gold_e4b_clean_rows.csv

Re-report an existing row file with principal-level intervals (CPU):
    python scripts/eval_assessor_gold.py \
        --from_rows results/judge_gold_e4b_clean_rows.csv --label judge_e4b \
        --out results/judge_e4b_principal
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluation.gold_metrics import (
    LEVELS, always_two, paired_principal_diff, principal_ci, summarize,
)

HALF_POINTS = (1, 3)


def heldout_frame(gold: pd.DataFrame) -> pd.DataFrame:
    """Adds is_demo and half_point columns to the gold frame."""
    from src.rewards.rubric_reward import RubricReward

    _, used = RubricReward.build_few_shot_examples(
        gold, max_examples=1, return_index=True
    )
    g = gold.copy()
    g["is_demo"] = g.index.isin(used)
    g["half_point"] = g["Scaled_Annotator_Rating"].isin(HALF_POINTS)
    return g


def predict_assessor(adapter, fallback_base, texts, elements, batch_size, max_length):
    """Class probabilities and predicted rubric points for each text."""
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # the adapter records the base it was trained on (the merged domain LM or
    # stock Qwen), so the right base is loaded without guessing
    base = PeftConfig.from_pretrained(adapter).base_model_name_or_path or fallback_base
    print(f"Assessor adapter: {adapter}\nBase model:       {base}", flush=True)

    tok = AutoTokenizer.from_pretrained(adapter)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=3, dtype=torch.bfloat16, device_map="cuda",
    )
    model.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    prefixed = [f"{e}: {t}" for e, t in zip(elements, texts)]
    probs = []
    with torch.no_grad():
        for i in range(0, len(prefixed), batch_size):
            enc = tok(prefixed[i:i + batch_size], return_tensors="pt",
                      truncation=True, max_length=max_length, padding=True).to("cuda")
            logits = model(**enc).logits.float()
            probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
    probs = np.asarray(probs)
    return probs, np.asarray(LEVELS)[probs.argmax(axis=1)]


def load_rows(path: Path, ours: pd.DataFrame) -> pd.DataFrame:
    """Loads another scorer's row file and checks it is row-aligned with ours."""
    other = pd.read_csv(path)
    key = ["PersonId", "Element_numberX", "score"]
    if len(other) != len(ours) or not (
        other[key].astype(str).to_numpy() == ours[key].astype(str).to_numpy()
    ).all():
        raise ValueError(
            f"{path} is not row-aligned with this gold set. Both files must come "
            f"from load_gold() on the same gold CSV."
        )
    return other


def report(g: pd.DataFrame, label: str, compare: list[Path], n_boot: int) -> dict:
    held = g[~g["is_demo"] & g["pred"].notna()]
    allr = g[g["pred"].notna()]
    y, p = held["score"].astype(int).to_numpy(), held["pred"].astype(int).to_numpy()
    pid = held["PersonId"].to_numpy()

    res = {"label": label, "held_out": summarize(y, p), "all_rows": summarize(
        allr["score"].astype(int), allr["pred"].astype(int)),
        "always_two_held_out": always_two(y)}
    res["held_out"]["exact_ci95"] = principal_ci(y, p, pid, "exact", n_boot)
    res["held_out"]["qwk_ci95"] = principal_ci(y, p, pid, "qwk", n_boot)
    res["n_principals_held_out"] = int(len(np.unique(pid)))

    nh = held[~held["half_point"]]
    res["no_half_point_rows"] = summarize(nh["score"].astype(int), nh["pred"].astype(int))

    res["per_element"] = {
        e: {"n": int(len(d)), "exact": float((d["score"] == d["pred"]).mean())}
        for e, d in held.groupby("Element_numberX")
    }
    conf = pd.crosstab(held["score"].astype(int), held["pred"].astype(int)).reindex(
        index=LEVELS, columns=LEVELS, fill_value=0)
    res["confusion"] = conf.to_dict()

    res["paired_vs"] = {}
    for path in compare:
        other = load_rows(path, g).loc[held.index]
        ok = other["pred"].notna().to_numpy()
        res["paired_vs"][Path(path).stem] = {
            "n_rows": int(ok.sum()),
            "exact": paired_principal_diff(
                y[ok], p[ok], other["pred"].to_numpy()[ok].astype(int), pid[ok],
                "exact", n_boot),
            "qwk": paired_principal_diff(
                y[ok], p[ok], other["pred"].to_numpy()[ok].astype(int), pid[ok],
                "qwk", n_boot),
        }
    return res


def print_report(r: dict) -> None:
    h = r["held_out"]
    print(f"\n== {r['label']}: held-out gold rows (n={h['n']}, "
          f"{r['n_principals_held_out']} principals) ==")
    print(f"exact     {h['exact']:.3f}  principal 95% CI "
          f"[{h['exact_ci95'][0]:.3f}, {h['exact_ci95'][1]:.3f}]   "
          f"always-2 baseline {r['always_two_held_out']['exact']:.3f}")
    print(f"QWK       {h['qwk']:.3f}  principal 95% CI "
          f"[{h['qwk_ci95'][0]:.3f}, {h['qwk_ci95'][1]:.3f}]   always-2 baseline 0.000")
    print(f"adjacent  {h['adjacent']:.3f}   signed deviation {h['signed_dev']:+.3f}")
    print("recall    " + "   ".join(
        f"level {lv}: {h[f'recall_{lv}']:.3f} (n={h[f'n_{lv}']})"
        if h[f'recall_{lv}'] is not None else f"level {lv}: n/a"
        for lv in LEVELS))
    a, nh = r["all_rows"], r["no_half_point_rows"]
    print(f"all {a['n']} rows (reference only): exact {a['exact']:.3f}, QWK {a['qwk']:.3f}")
    print(f"without half-point rows (n={nh['n']}): exact {nh['exact']:.3f}, QWK {nh['qwk']:.3f}")
    print("per element exact: " + "  ".join(
        f"{e[-1]}:{m['exact']:.2f}(n={m['n']})" for e, m in sorted(r["per_element"].items())))
    print("confusion (rows = expert, cols = scorer):")
    print(pd.DataFrame(r["confusion"]).rename_axis("expert").to_string())
    for name, c in r["paired_vs"].items():
        ex, qw = c["exact"], c["qwk"]
        print(f"\nvs {name} (same {c['n_rows']} rows, paired principal bootstrap)")
        print(f"  exact diff {ex['diff']:+.3f}  95% CI [{ex['lo']:+.3f}, {ex['hi']:+.3f}]  "
              f"share of resamples not better: {ex['share_a_not_better']:.3f}")
        print(f"  QWK diff   {qw['diff']:+.3f}  95% CI [{qw['lo']:+.3f}, {qw['hi']:+.3f}]  "
              f"share of resamples not better: {qw['share_a_not_better']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter", help="trained assessor adapter directory")
    src.add_argument("--from_rows", help="existing row file to re-report (CPU only)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="row files of other scorers for paired comparison")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B",
                    help="used only if the adapter does not record its base")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    from src.data.corpus import load_gold

    gold = heldout_frame(load_gold())
    print(f"Gold rows {len(gold)}, principals {gold['PersonId'].nunique()}, "
          f"few-shot demonstration rows excluded from the test: {int(gold['is_demo'].sum())}")

    if args.adapter:
        probs, pred = predict_assessor(
            args.adapter, args.base_model, gold["Text"].tolist(),
            gold["Element_numberX"].tolist(), args.batch_size, args.max_length)
        gold["pred"] = pred
        for i, lv in enumerate(LEVELS):
            gold[f"p{lv}"] = probs[:, i]
    else:
        prev = load_rows(Path(args.from_rows), gold)
        gold["pred"] = prev["pred"].to_numpy()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["PersonId", "Element_numberX", "score", "pred", "is_demo", "half_point"]
    cols += [c for c in ("p0", "p2", "p4") if c in gold]
    gold[cols].to_csv(f"{out}_rows.csv", index=False)

    r = report(gold, args.label, [Path(c) for c in args.compare], args.n_boot)
    print_report(r)
    with open(f"{out}.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"\nSaved {out}_rows.csv and {out}.json")


if __name__ == "__main__":
    main()
