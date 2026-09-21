#!/usr/bin/env python
"""
Runs one PPO configuration, pilot or full.

Checkpoints every --save_every steps (adapter, value head, optimizer,
history) and resumes from the newest complete checkpoint in --output on
restart. contrib-gpuq preempts and requeues guest jobs; without resume a
requeued run starts over from step 0.

Usage:
    python scripts/run_ppo_pilot.py --alpha 0.3 --kl_coef 0.2 --steps 5000
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from peft import load_peft_weights, set_peft_model_state_dict

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
# 2, matching CustomPPO and PPOConfig. 4 OOM'd on the policy device.
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--save_every", type=int, default=100)
parser.add_argument("--keep_checkpoints", type=int, default=2)
parser.add_argument("--output", default=None)
args = parser.parse_args()

out = Path(args.output or f"models/PPOPilot_a{args.alpha}_kl{args.kl_coef}")
out.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"PPO RUN  alpha={args.alpha}  kl_coef={args.kl_coef}  "
      f"steps={args.steps}")
print(f"output: {out}")
print("=" * 70)

df = load(args.data)
anchor_df = for_anchor_pool(df)
gold_df = load_gold()
print(f"Anchor pool: {len(anchor_df):,}  Gold: {len(gold_df)}")

ppo = CustomPPO(
    sft_checkpoint=args.sft_checkpoint,
    output_dir=str(out),
    anchor_df=anchor_df,
    gold_df=gold_df,
    alpha=args.alpha,
    kl_coef=args.kl_coef,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
)


def checkpoint_step(p):
    return int(p.name.split("-")[1])


def latest_checkpoint():
    done = [c for c in out.glob("checkpoint-*") if (c / "state.json").exists()]
    return max(done, key=checkpoint_step) if done else None


def save_checkpoint(next_step):
    # write to a temp dir, then rename. A preemption mid-save must not
    # leave a half-written checkpoint that the resume path would load.
    final = out / f"checkpoint-{next_step}"
    tmp = out / f"tmp-checkpoint-{next_step}"
    shutil.rmtree(tmp, ignore_errors=True)
    ppo.policy.save_pretrained(str(tmp))
    torch.save(ppo.value_head.state_dict(), tmp / "value_head.pt")
    torch.save(ppo.optimizer.state_dict(), tmp / "optimizer.pt")
    with open(tmp / "history.json", "w") as f:
        json.dump(history, f)
    with open(tmp / "state.json", "w") as f:
        json.dump({"next_step": next_step}, f)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    done = sorted(out.glob("checkpoint-*"), key=checkpoint_step)
    for old in done[:-args.keep_checkpoints]:
        shutil.rmtree(old, ignore_errors=True)
    print(f"  checkpoint saved: {final}")


# ── resume ──────────────────────────────────────────────────────────

for stale in out.glob("tmp-checkpoint-*"):
    shutil.rmtree(stale, ignore_errors=True)

history = []
start_step = 0
ckpt = latest_checkpoint()
if ckpt is not None:
    print(f"Resuming from {ckpt}")
    set_peft_model_state_dict(
        ppo.policy, load_peft_weights(str(ckpt), device=ppo.device)
    )
    ppo.value_head.load_state_dict(
        torch.load(ckpt / "value_head.pt", map_location=ppo.device)
    )
    ppo.optimizer.load_state_dict(
        torch.load(ckpt / "optimizer.pt", map_location=ppo.device)
    )
    with open(ckpt / "history.json") as f:
        history = json.load(f)
    with open(ckpt / "state.json") as f:
        start_step = json.load(f)["next_step"]
    # fresh sampling streams, so the resumed run does not replay the
    # prompts and samples it already trained on
    torch.manual_seed(42 + start_step)
    ppo.rng = np.random.default_rng(42 + start_step)
    ppo.element_cycler.rng = ppo.rng
    print(f"Resumed at step {start_step} with {len(history)} history rows")

# ── train ───────────────────────────────────────────────────────────

start = time.time()

for step in range(start_step, args.steps):
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
            f"[{el/(step - start_step + 1):.1f}s/step]"
        )

    if step % 50 == 0 and step > 0:
        for w in check_divergence(history):
            print(f"  WARNING: {w}")

    if (step + 1) % args.save_every == 0:
        save_checkpoint(step + 1)

elapsed = time.time() - start
steps_run = max(args.steps - start_step, 1)

ppo.policy.save_pretrained(str(out))
ppo.tokenizer.save_pretrained(str(out))
torch.save(ppo.value_head.state_dict(), out / "value_head.pt")
with open(out / "history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\n" + "=" * 70)
print("RUN SUMMARY")
print("=" * 70)
print(f"Elapsed this segment: {elapsed/60:.1f} min  "
      f"({elapsed/steps_run:.2f} s/step)")
print(f"Extrapolated 5000 steps: {(elapsed/steps_run*5000)/3600:.1f} hours")

first = history[:25]
last = history[-25:]
for key, label, fmt in [
    ("reward_combined", "reward", ".4f"),
    ("reward_auth", "auth", ".4f"),
    ("reward_rubric", "rubric", ".4f"),
    ("kl", "kl", ".2f"),
    ("value_loss", "v_loss", ".4f"),
    ("mean_gen_words", "words", ".0f"),
]:
    a = np.mean([h[key] for h in first])
    b = np.mean([h[key] for h in last])
    print(f"{label:8s} {a:{fmt}} -> {b:{fmt}}")

warns = check_divergence(history)
if warns:
    print("\nDIVERGENCE WARNINGS:")
    for w in warns:
        print(f"  {w}")
    print("\nThis configuration is not learning cleanly.")
else:
    print("\nNo divergence warnings. Configuration looks trainable.")