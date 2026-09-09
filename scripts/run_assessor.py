#!/usr/bin/env python
"""
Trains the assessor under one experimental condition.

Conditions:
  synthetic_only  generated text with rubric-derived target scores
  real_noisy      real BIPs with supervisor scores, natural skew
  balanced_real   real BIPs downsampled to the synthetic distribution
  hybrid          both pools combined

Usage:
    python scripts/run_assessor.py --condition synthetic_only \
        --synthetic models/BoN_round1/accepted.jsonl \
        --output models/Assessor_synthetic
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from src.data.corpus import load, for_anchor_pool
from src.training.assessor import (
    build_assessor_dataset, load_condition_data, train_assessor, LABEL_MAP,
)
from src.training.ordinal_loss import compute_class_weights
from src.utils.config import AssessorConfig


parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/raw/bips.csv")
parser.add_argument("--condition", required=True,
                    choices=["synthetic_only", "real_noisy",
                             "balanced_real", "hybrid"])
parser.add_argument("--synthetic", default="models/BoN_round1/accepted.jsonl")
parser.add_argument("--output", required=True)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=2)
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
if args.condition != "synthetic_only":
    anchor_df = for_anchor_pool(load(args.data))
    print(f"Anchor pool: {len(anchor_df):,}")

texts, elements, scores = load_condition_data(
    condition=args.condition,
    synthetic_path=args.synthetic,
    anchor_df=anchor_df,
)
print(f"Training examples: {len(texts):,}")

import pandas as pd
print("Score distribution:")
print(pd.Series(scores).value_counts().sort_index().to_string())

config = AssessorConfig(
    training_condition=args.condition,
    output_dir=args.output,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=16,
    ordinal_loss=not args.no_ordinal_loss,
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = build_assessor_dataset(
    texts, elements, scores, tokenizer, max_length=config.max_seq_length,
)

class_weights = compute_class_weights(
    [LABEL_MAP[int(s)] for s in scores],
    num_classes=config.num_labels,
)

train_assessor(
    config, dataset, tokenizer,
    class_weights=class_weights,
    domain_checkpoint=None if args.no_domain_init else "models/BIPDomainSFT",
)

print(f"\nDone in {(time.time() - start)/60:.1f} min")