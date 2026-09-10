#!/usr/bin/env python
"""
Runs one round of best-of-n rejection-sampling fine-tuning.

Each round: sample N candidates per prompt from the current generator,
score with both reward models, keep the top k, retrain the generator on
the accepted set, continuing from the adapter passed via --generator
rather than reinitializing LoRA from the base model.

Round 1 starts from the SFT warmup checkpoint; each later round starts
from the previous round's output.

The generation and retrain phases are separated deliberately. Round 1
generated 1000 accepted candidates over ~10 hours and then OOM'd at the
retrain step, losing nothing but wasting a queue slot. --retrain_only
picks up from accepted.jsonl so that work is never repeated.

Usage:
    python scripts/run_bon_round.py --round 2 \
        --generator models/BoN_round1 \
        --output models/BoN_round2 --n_prompts 500

    python scripts/run_bon_round.py --round 2 \
        --generator models/BoN_round1 \
        --output models/BoN_round2 --retrain_only
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.data.corpus import load, for_anchor_pool, load_gold
from src.training.bon_round import BestOfNRound, retrain_generator
from src.utils.config import GeneratorSFTConfig


parser = argparse.ArgumentParser(description="One best-of-n round")
parser.add_argument("--data", default="data/raw/bips.csv")
parser.add_argument("--round", type=int, required=True)
parser.add_argument("--generator", required=True,
                    help="Adapter to sample from, and to continue training from at retrain time")
parser.add_argument("--output", required=True)
parser.add_argument("--n_prompts", type=int, default=500)
parser.add_argument("--n_candidates", type=int, default=8)
parser.add_argument("--keep_top_k", type=int, default=2)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--min_reward", type=float, default=0.0)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--skip_retrain", action="store_true",
                    help="Generate and score only")
parser.add_argument("--retrain_only", action="store_true",
                    help="Skip generation; retrain from existing accepted.jsonl")
args = parser.parse_args()

print("=" * 70)
print(f"BEST-OF-N ROUND {args.round}")
print(f"  output:       {args.output}")
print(f"  generator:    {args.generator}")
if not args.retrain_only:
    print(f"  n_prompts:    {args.n_prompts}")
    print(f"  n_candidates: {args.n_candidates}")
    print(f"  keep_top_k:   {args.keep_top_k}")
    print(f"  alpha:        {args.alpha}")
print(f"  visible GPUs: {torch.cuda.device_count()}")
print("=" * 70)

start = time.time()

# ── generation and scoring ──────────────────────────────────────────

if args.retrain_only:
    path = Path(args.output) / "accepted.jsonl"
    if not path.exists():
        print(f"No accepted set at {path}")
        sys.exit(1)
    accepted = pd.read_json(path, lines=True)
    print(f"Loaded {len(accepted)} accepted candidates from {path}")
else:
    df = load(args.data)
    anchor_df = for_anchor_pool(df)
    gold_df = load_gold()
    print(f"Anchor pool: {len(anchor_df):,}  Gold: {len(gold_df)}")

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

    # free the generator and both reward models before training. they
    # hold roughly 30 GB and are not needed past this point.
    round_runner.release()
    del round_runner
    gc.collect()
    torch.cuda.empty_cache()

# ── retrain ─────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("RETRAINING GENERATOR ON ACCEPTED SET")
print(f"  continuing from: {args.generator}")
print("=" * 70)

config = GeneratorSFTConfig(
    output_dir=args.output,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=16,
)

retrain_generator(
    accepted_df=accepted,
    base_model_name=config.model_name,
    prev_adapter_path=args.generator,
    output_dir=config.output_dir,
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    warmup_ratio=config.warmup_ratio,
    max_seq_length=config.max_seq_length,
    seed=42 + args.round,
)

total = time.time() - start
print(f"\nRound {args.round} complete in {total/60:.1f} min")
print(f"New generator adapter: {args.output}")