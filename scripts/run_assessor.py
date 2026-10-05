#!/usr/bin/env python
"""
Trains the assessor under one experimental condition.

Conditions:
  synthetic_only  generated text with rubric-derived target scores
  real_noisy      real BIPs with supervisor scores, natural skew
  balanced_real   real BIPs downsampled to the synthetic score distribution
  judge_real      real BIPs labelled by the judge, matched to the synthetic
                  (element, score) counts. Needs --judge_labelled
  hybrid          synthetic and real pools combined

A slice of the training data (--val_fraction, default 10%) is held back and
scored after training, giving val_metrics.json. Use that, never the gold
set, to choose between hyperparameter settings. For synthetic data the
slice is split by prompt, so the two BIPs kept from one prompt never land
on both sides.

Usage:
    python scripts/run_assessor.py --condition synthetic_only \
        --synthetic models/BoN_v2_full/accepted.jsonl \
        --learning_rate 1e-4 --output models/Assessor_synthetic_lr1e-4
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.data.corpus import load, for_anchor_pool
from src.training.assessor import (
    LABEL_MAP, build_assessor_dataset, load_condition_data,
    split_by_group, train_assessor,
)
from src.training.ordinal_loss import compute_class_weights
from src.utils.config import AssessorConfig


parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/raw/bips.csv")
parser.add_argument("--condition", required=True,
                    choices=["synthetic_only", "real_noisy", "balanced_real",
                             "judge_real", "hybrid"])
parser.add_argument("--synthetic", default="models/BoN_v2_full/accepted.jsonl")
parser.add_argument("--judge_labelled", default="data/derived/judge_labelled_real.jsonl")
parser.add_argument("--output", required=True)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=None,
                    help="default: AssessorConfig.learning_rate (1e-4)")
parser.add_argument("--val_fraction", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no_domain_init", action="store_true",
                    help="Start from stock Qwen instead of BIPDomainSFT")
parser.add_argument("--no_ordinal_loss", action="store_true")
args = parser.parse_args()

print("=" * 70)
print(f"ASSESSOR  condition={args.condition}")
print(f"  output:    {args.output}")
print(f"  synthetic: {args.synthetic}")
print(f"  GPUs:      {torch.cuda.device_count()}")
print("=" * 70)

start = time.time()

anchor_df = None
if args.condition in ("real_noisy", "balanced_real", "hybrid"):
    anchor_df = for_anchor_pool(load(args.data))
    print(f"Anchor pool: {len(anchor_df):,}")

texts, elements, scores = load_condition_data(
    condition=args.condition,
    synthetic_path=args.synthetic,
    anchor_df=anchor_df,
    seed=args.seed,
    judge_labelled_path=args.judge_labelled,
)
print(f"Examples before the validation split: {len(texts):,}")

# group synthetic rows by prompt so siblings stay together
groups = None
if args.condition == "synthetic_only":
    syn = pd.read_json(args.synthetic, lines=True)
    if "prompt_idx" in syn.columns:
        groups = syn["prompt_idx"].tolist()
    else:
        print("WARNING: no prompt_idx column; validation split is by row")

train_idx, val_idx = split_by_group(len(texts), groups, args.val_fraction, args.seed)
tr = lambda xs: [xs[i] for i in train_idx]
va = lambda xs: [xs[i] for i in val_idx]
train_texts, train_elements, train_scores = tr(texts), tr(elements), tr(scores)
val = None
if len(val_idx):
    val = (va(texts), va(elements), [LABEL_MAP[int(s)] for s in va(scores)])
print(f"Train {len(train_idx):,}   validation {len(val_idx):,}")

print("Training score distribution:")
print(pd.Series(train_scores).value_counts().sort_index().to_string())

config = AssessorConfig(
    training_condition=args.condition,
    output_dir=args.output,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=16,
    ordinal_loss=not args.no_ordinal_loss,
    seed=args.seed,
)
if args.learning_rate is not None:
    config.learning_rate = args.learning_rate

Path(args.output).mkdir(parents=True, exist_ok=True)
with open(Path(args.output) / "run_args.json", "w") as f:
    json.dump({**vars(args), "learning_rate_used": config.learning_rate,
               "n_train": len(train_idx), "n_val": len(val_idx)}, f, indent=2)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = build_assessor_dataset(
    train_texts, train_elements, train_scores, tokenizer,
    max_length=config.max_seq_length,
)

class_weights = compute_class_weights(
    [LABEL_MAP[int(s)] for s in train_scores],
    num_classes=config.num_labels,
)

train_assessor(
    config, dataset, tokenizer,
    class_weights=class_weights,
    domain_checkpoint=None if args.no_domain_init else "models/BIPDomainSFT",
    val=val,
)

print(f"\nDone in {(time.time() - start)/60:.1f} min")
