#!/usr/bin/env python
"""
Runs a single PPO pilot configuration for a limited number of steps.
Submit multiple copies in parallel across your 4 GPUs to sweep
alpha/beta simultaneously.

Usage:
    python scripts/run_ppo_pilot.py --alpha 0.5 --beta 0.1 --steps 300
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import PPOConfig
from src.data.corpus import load, for_anchor_pool
from src.training.ppo_trainer import PPOPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/raw/bips.csv")
parser.add_argument("--sft_checkpoint", default="models/GeneratorSFT")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

config = PPOConfig(
    sft_checkpoint=args.sft_checkpoint,
    output_dir=f"models/PPOPilot_a{args.alpha}_b{args.beta}",
    log_dir=f"logs/PPOPilot_a{args.alpha}_b{args.beta}",
    alpha=args.alpha,
    beta=args.beta,
    batch_size=args.batch_size,
    max_steps=args.steps,
    save_every=100,
)

print(f"Pilot config: alpha={args.alpha} beta={args.beta} steps={args.steps}")

df = load(args.data)
anchor_df = for_anchor_pool(df)

pipeline = PPOPipeline(config, anchor_df)

start = time.time()
pipeline.run(max_steps=args.steps)
elapsed = time.time() - start

print(f"\nPilot complete in {elapsed/60:.1f} minutes "
      f"({elapsed/args.steps:.2f} sec/step)")
print(f"Extrapolated time for 5000 steps: "
      f"{(elapsed/args.steps * 5000)/3600:.1f} hours")