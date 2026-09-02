#!/usr/bin/env python
"""
Runs one round of best-of-n rejection-sampling fine-tuning.

Each round: sample N candidates per prompt from the current generator,
score with both reward models, keep the top k, retrain the generator on
the accepted set.

Round 1 starts from the SFT warmup checkpoint. Each subsequent round
starts from the previous round's output.

Usage:
    python scripts/run_bon_round.py --round 1 \
        --generator models/GeneratorSFT \
        --output models/BoN_round1 \
        --n_prompts 500
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from transformers import AutoTokenizer

from src.data.corpus import load, for_anchor_pool, load_gold
from src.training.bon_round import BestOfNRound, build_retrain_dataset
from src.training.generator_sft import train as train_generator
from src.utils.config import GeneratorSFTConfig


parser = argparse.ArgumentParser(description="One best-of-n round")
parser.add_argument("--data", default="data/raw/bips.csv")
parser.add_argument("--round", type=int, required=True)
parser.add_argument("--generator", required=True,
                    help="Adapter to sample from this round")
parser.add_argument("--output", required=True,
                    help="Where to write accepted set and new adapter")
parser.add_argument("--n_prompts", type=int, default=500)
parser.add_argument("--n_candidates", type=int, default=8)
parser.add_argument("--keep_top_k", type=int, default=2)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--min_reward", type=float, default=0.0)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--skip_retrain", action="store_true",
                    help="Generate and score only, no fine-tune")
args = parser.parse_args()

print("=" * 70)
print(f"BEST-OF-N ROUND {args.round}")
print(f"  generator:    {args.generator}")
print(f"  output:       {args.output}")
print(f"  n_prompts:    {args.n_prompts}")
print(f"  n_candidates: {args.n_candidates}")
print(f"  keep_top_k:   {args.keep_top_k}")
print(f"  alpha:        {args.alpha}")
print("=" * 70)

df = load(args.data)
anchor_df = for_anchor_pool(df)
gold_df = load_gold()
print(f"Anchor pool: {len(anchor_df):,}  Gold: {len(gold_df)}")

start = time.time()

round_runner = BestOfNRound(
    anchor_df=anchor_df,
    gold_df=gold_df,
    generator_adapter=args.generator,
    output_dir=args.output,
    n_candidates=args.n_candidates,
    keep_top_k=args.keep_top_k,
    alpha=args.alpha,
    min_reward=args.min_reward,
    seed=42 + args.round,
)

accepted = round_runner.run(n_prompts=args.n_prompts)

gen_elapsed = time.time() - start
print(f"\nGeneration and scoring: {gen_elapsed/60:.1f} min "
      f"({gen_elapsed/args.n_prompts:.2f} s/prompt)")

if len(accepted) == 0:
    print("No candidates accepted. Nothing to retrain on.")
    sys.exit(1)

if args.skip_retrain:
    print("skip_retrain set; stopping after generation.")
    sys.exit(0)

# ── retrain the generator on accepted candidates ────────────────────

print("\n" + "=" * 70)
print("RETRAINING GENERATOR ON ACCEPTED SET")
print("=" * 70)

config = GeneratorSFTConfig(
    output_dir=args.output,
    num_train_epochs=args.epochs,
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = build_retrain_dataset(
    accepted, tokenizer, max_length=config.max_seq_length
)
print(f"Retrain dataset: {len(dataset)} examples")

train_generator(config, dataset, tokenizer=tokenizer)

total = time.time() - start
print(f"\nRound {args.round} complete in {total/60:.1f} min")
print(f"New generator adapter: {args.output}")