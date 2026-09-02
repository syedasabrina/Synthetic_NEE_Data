#!/usr/bin/env python
"""
Runs one PPO pilot configuration.

Prints measured seconds per step and extrapolates to a full run, and
checks for silent divergence every step so a bad configuration is
visible early rather than after hours of compute.

Usage:
    python scripts/run_ppo_pilot.py --alpha 0.5 --kl_coef 0.1 --steps 300
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.corpus import load, for_anchor_pool, load_gold
from src.training.ppo_trainer import CustomPPO, check_divergence
from src.generation.sampler import sample_prompt_spec


parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/raw/bips.csv")
parser.add_argument("--sft_checkpoint", default="models/GeneratorSFT")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--kl_coef", type=float, default=0.1)
parser.add_argument("--learning_rate", type=float, default=1.41e-5)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--save_every", type=int, default=100)
parser.add_argument("--output", default=None)
args = parser.parse_args()

out = args.output or f"models/PPOPilot_a{args.alpha}_kl{args.kl_coef}"

print("=" * 70)
print(f"PPO PILOT  alpha={args.alpha}  kl_coef={args.kl_coef}  "
      f"steps={args.steps}")
print(f"output: {out}")
print("=" * 70)

df = load(args.data)
anchor_df = for_anchor_pool(df)
gold_df = load_gold()
print(f"Anchor pool: {len(anchor_df):,}  Gold: {len(gold_df)}")

ppo = CustomPPO(
    sft_checkpoint=args.sft_checkpoint,
    output_dir=out,
    anchor_df=anchor_df,
    gold_df=gold_df,
    alpha=args.alpha,
    kl_coef=args.kl_coef,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
)

history = []
start = time.time()

for step in range(args.steps):
    specs = [
        sample_prompt_spec(
            anchor_df, ppo.rubric, ppo.element_cycler, ppo.rng
        )
        for _ in range(args.batch_size)
    ]

    stats = ppo.step(specs)
    stats["step"] = step
    history.append(stats)

    if step % 10 == 0:
        el = time.time() - start
        print(
            f"step {step:4d}  "
            f"reward={stats['reward_combined']:.4f}  "
            f"auth={stats['reward_auth']:.4f}  "
            f"rubric={stats['reward_rubric']:.4f}  "
            f"kl={stats['kl']:.2f}  "
            f"pg={stats['pg_loss']:.4f}  "
            f"v={stats['value_loss']:.4f}  "
            f"clip={stats['clip_frac']:.3f}  "
            f"words={stats['mean_gen_words']:.0f}  "
            f"[{el/(step+1):.1f}s/step]"
        )

    if step % 50 == 0 and step > 0:
        for w in check_divergence(history):
            print(f"  WARNING: {w}")

    if step % args.save_every == 0 and step > 0:
        ppo.policy.save_pretrained(f"{out}/checkpoint-{step}")
        with open(f"{out}/history.json", "w") as f:
            json.dump(history, f, indent=2)

elapsed = time.time() - start

ppo.policy.save_pretrained(out)
ppo.tokenizer.save_pretrained(out)
with open(f"{out}/history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\n" + "=" * 70)
print("PILOT SUMMARY")
print("=" * 70)
print(f"Elapsed: {elapsed/60:.1f} min  ({elapsed/args.steps:.2f} s/step)")
print(f"Extrapolated 5000 steps: {(elapsed/args.steps*5000)/3600:.1f} hours")

first = history[:25]
last = history[-25:]
print(f"\nreward   {np.mean([h['reward_combined'] for h in first]):.4f} "
      f"-> {np.mean([h['reward_combined'] for h in last]):.4f}")
print(f"auth     {np.mean([h['reward_auth'] for h in first]):.4f} "
      f"-> {np.mean([h['reward_auth'] for h in last]):.4f}")
print(f"rubric   {np.mean([h['reward_rubric'] for h in first]):.4f} "
      f"-> {np.mean([h['reward_rubric'] for h in last]):.4f}")
print(f"kl       {np.mean([h['kl'] for h in first]):.2f} "
      f"-> {np.mean([h['kl'] for h in last]):.2f}")
print(f"v_loss   {np.mean([h['value_loss'] for h in first]):.4f} "
      f"-> {np.mean([h['value_loss'] for h in last]):.4f}")
print(f"words    {np.mean([h['mean_gen_words'] for h in first]):.0f} "
      f"-> {np.mean([h['mean_gen_words'] for h in last]):.0f}")

warns = check_divergence(history)
if warns:
    print("\nDIVERGENCE WARNINGS:")
    for w in warns:
        print(f"  {w}")
    print("\nThis configuration is not learning cleanly.")
else:
    print("\nNo divergence warnings. Configuration looks trainable.")