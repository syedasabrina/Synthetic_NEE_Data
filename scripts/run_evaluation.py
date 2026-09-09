#!/usr/bin/env python
"""
Evaluates a trained assessor on the held-out gold standard set.

The 171 gold rows come from 24 principals excluded from every training
stage, so this is the only measurement in the pipeline that is not
downstream of the reward models the generator was optimised against.

Usage:
    python scripts/run_evaluation.py \
        --model models/Assessor_synthetic \
        --condition synthetic_only

    python scripts/run_evaluation.py --compare \
        results/assessor/synthetic_only.json \
        results/assessor/real_noisy.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.corpus import load_gold
from src.training.assessor import LABEL_MAP, INV_LABEL_MAP
from src.evaluation.metrics import (
    evaluate, print_report, save_results, compare_conditions,
)


def predict(model, tokenizer, texts, elements, device="cuda",
            batch_size=8, max_length=1024):
    """
    Predicted class indices and confidence for each gold BIP.

    The element prefix matches training, where it was added because the
    same response warrants different scores under different elements.
    """
    prefixed = [f"{e}: {t}" for e, t in zip(elements, texts)]
    preds, confs = [], []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i: i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=max_length, padding=True,
            ).to(device)
            logits = model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            confs.extend(probs.max(dim=-1).values.cpu().tolist())

    return np.array(preds), np.array(confs)


parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None)
parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B")
parser.add_argument("--condition", default="unknown")
parser.add_argument("--results_dir", default="results/assessor")
parser.add_argument("--bootstrap_n", type=int, default=1000)
parser.add_argument("--compare", nargs="*", default=None,
                    help="Result JSON files to tabulate side by side")
args = parser.parse_args()

# comparison mode reads saved results, no GPU needed
if args.compare:
    by_cond = {}
    for p in args.compare:
        r = json.load(open(p))
        by_cond[r.get("condition", Path(p).stem)] = r
    print(compare_conditions(by_cond).to_string(index=False))
    sys.exit(0)

if not args.model:
    print("--model is required unless --compare is given")
    sys.exit(1)

print("=" * 70)
print(f"EVALUATION  model={args.model}  condition={args.condition}")
print("=" * 70)

gold = load_gold()
print(f"Gold rows: {len(gold)}  principals: {gold['PersonId'].nunique()}")
print("Gold score distribution:")
print(gold["score"].value_counts().sort_index().to_string())
print()

# the merged domain LM is the base if it exists, since the assessor
# adapter was trained on top of it
merged = Path("models/_merged_domain_lm")
base_path = str(merged) if merged.exists() else args.base_model
print(f"Base: {base_path}")

tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    base_path, num_labels=3, dtype=torch.bfloat16, device_map="cuda",
)
model.config.pad_token_id = tokenizer.pad_token_id
model = PeftModel.from_pretrained(model, args.model)

preds, confs = predict(
    model, tokenizer,
    gold["Text"].tolist(), gold["Element_numberX"].tolist(),
)
y_true = np.array([LABEL_MAP[int(s)] for s in gold["score"]])

results = evaluate(
    y_true, preds,
    elements=gold["Element_numberX"].tolist(),
    bootstrap_n=args.bootstrap_n,
)
results["condition"] = args.condition
results["model_path"] = args.model
results["mean_confidence"] = float(confs.mean())

print_report(results, title=f"GOLD STANDARD -- {args.condition}")

# high-confidence subset, which is what the corpus audit will use
hi = confs >= np.percentile(confs, 50)
if hi.sum() >= 10:
    hi_res = evaluate(
        y_true[hi], preds[hi],
        elements=np.array(gold["Element_numberX"])[hi].tolist(),
        bootstrap_n=args.bootstrap_n, run_loo=False,
    )
    results["high_confidence"] = hi_res
    print(f"High-confidence half (n={int(hi.sum())}): "
          f"QWK={hi_res['qwk']:.4f}  exact={hi_res['exact_agreement']:.4f}")
    print("A large gap against the full set means confidence is a usable")
    print("filter for the corpus audit; no gap means it carries no signal.")
    print()

save_results(results, Path(args.results_dir) / f"{args.condition}.json")