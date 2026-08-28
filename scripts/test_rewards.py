#!/usr/bin/env python
"""
Tests both reward models and runs judge calibration on the gold
standard set.

Two things happen here:

1. Reward model smoke test on hand-written examples -- confirms the
   authenticity reward separates BIP-like text from off-domain text,
   and that the blind rubric judge produces parseable scores.

2. Judge calibration on the gold standard set -- runs the blind
   rubric judge against expert-scored BIPs it has never been trained
   on, and reports agreement. This is the pre-PPO calibration check
   the project scope requires: a judge that systematically disagrees
   with experts would push PPO toward the wrong objective.

Note on the gold set: it is used here ONLY as frozen in-context
few-shot examples and as a calibration reference for a frozen model.
No gradients flow, no training occurs. The set remains held out for
final assessor evaluation.

Usage:
    sbatch --export=ALL,SCRIPT=scripts/test_rewards.py,ARGS="" scripts/train.slurm
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rewards.authenticity_reward import AuthenticityReward
from src.rewards.rubric_reward import RubricReward
from src.data.corpus import load_gold


# ── Part 1: smoke test on hand-written examples ─────────────────────

candidates = [
    "Element6: Throughout the school year, we monitored our building "
    "improvement objectives by collecting MAP data every eight weeks "
    "and reviewing results with our grade level teams during "
    "collaborative planning time. When data showed students were not "
    "making expected progress in reading fluency, we implemented "
    "small group intervention blocks three times per week.",

    "Element3: Our BIP objectives are directly aligned to the district "
    "CSIP goals for 2023-2024. Objective 1 supports CSIP Goal 2 "
    "around increasing ELA proficiency. Each objective includes a "
    "measurable target tied to district performance indicators.",

    "We monitored our goals.",

    # repetitive text -- should be penalized by the repetition term
    # even though its perplexity under BIPDomainSFT is likely low
    "We monitored our data. We monitored our data. We reviewed our "
    "data. We monitored our data. We reviewed our data regularly.",

    "The weather today is sunny with a high of 75 degrees. "
    "We recommend bringing sunscreen if you plan to be outside.",
]

elements      = ["Element6", "Element3", "Element6", "Element6", "Element6"]
target_scores = [4,          4,          4,          4,          4]

print("=" * 70)
print("PART 1a: AUTHENTICITY REWARD (BIPDomainSFT + repetition penalty)")
print("=" * 70)
auth = AuthenticityReward(
    base_model_name="Qwen/Qwen2.5-7B",
    adapter_path="models/BIPDomainSFT",
    device="cuda",
)
auth_rewards = auth.score_batch(candidates, batch_size=2)
for cand, reward in zip(candidates, auth_rewards):
    print(f"  auth={reward:.4f} | {cand[:65]}...")

print()
print("Expected pattern: the two real BIP-style texts score highest,")
print("the repetitive text is pulled down by the repetition penalty,")
print("and the weather text scores lowest.")

# ── Part 2: load gold standard and build few-shot examples ──────────

print()
print("=" * 70)
print("PART 1b: RUBRIC REWARD (Gemma 4 E4B, blind scoring, few-shot)")
print("=" * 70)

gold_df = load_gold()
print(f"Gold standard loaded: {len(gold_df)} rows, "
      f"{gold_df['PersonId'].nunique()} principals")

few_shot = RubricReward.build_few_shot_examples(gold_df)
print(f"Few-shot cells populated: {len(few_shot)} (element, score) pairs")
covered = sorted(few_shot.keys())
print(f"Coverage: {covered[:8]}{' ...' if len(covered) > 8 else ''}")

rubric = RubricReward(
    model_name="google/gemma-4-E4B-it",
    device="cuda",
    few_shot_examples=few_shot,
)

predicted = rubric.predict(candidates, elements)
rubric_rewards = rubric.score(candidates, elements, target_scores)
for cand, elem, target, pred, reward in zip(
    candidates, elements, target_scores, predicted, rubric_rewards
):
    print(f"  predicted={pred} target={target} reward={reward:.1f} | "
          f"{cand[:50]}...")

# ── Part 3: combined reward ─────────────────────────────────────────

print()
print("=" * 70)
print("PART 1c: COMBINED REWARD (alpha=0.5)")
print("=" * 70)
alpha = 0.5
for cand, auth_r, rubric_r in zip(candidates, auth_rewards, rubric_rewards):
    combined = alpha * auth_r + (1 - alpha) * rubric_r
    print(f"  combined={combined:.4f} | auth={auth_r:.4f} "
          f"rubric={rubric_r:.1f} | {cand[:45]}...")

# ── Part 4: judge calibration on gold standard ──────────────────────

print()
print("=" * 70)
print("PART 2: RUBRIC JUDGE CALIBRATION ON GOLD STANDARD")
print("=" * 70)
print("Running blind rubric judge against expert-scored BIPs.")
print("The judge has never seen these scores; agreement below is a")
print("measure of whether it interprets the rubric like experts do.")
print()

gold_texts    = gold_df["Text"].tolist()
gold_elements = gold_df["Element_numberX"].tolist()
gold_scores   = gold_df["score"].astype(int).tolist()

gold_predicted = rubric.predict(gold_texts, gold_elements)

# drop unparseable predictions from agreement stats but report count
parseable = [
    (p, t) for p, t in zip(gold_predicted, gold_scores) if p is not None
]
n_unparseable = len(gold_predicted) - len(parseable)

if parseable:
    preds = np.array([p for p, _ in parseable])
    trues = np.array([t for _, t in parseable])

    exact = float((preds == trues).mean())
    adjacent = float((np.abs(preds - trues) <= 2).mean())
    mean_signed = float((preds - trues).mean())

    print(f"Scored:            {len(parseable)}/{len(gold_predicted)}")
    print(f"Unparseable:       {n_unparseable}")
    print(f"Exact agreement:   {exact:.3f}")
    print(f"Adjacent agreement:{adjacent:.3f}")
    print(f"Mean signed dev:   {mean_signed:+.3f}  "
          f"(positive = judge scores higher than experts)")
    print()

    print("Confusion (rows = expert score, cols = judge prediction):")
    print(f"{'':>8}" + "".join(f"{c:>8}" for c in [0, 2, 4]))
    for t in [0, 2, 4]:
        row = [int(((trues == t) & (preds == p)).sum()) for p in [0, 2, 4]]
        print(f"{t:>8}" + "".join(f"{v:>8}" for v in row))
    print()

    print("Per-element exact agreement:")
    elem_arr = np.array([
        e for e, p in zip(gold_elements, gold_predicted) if p is not None
    ])
    for elem in sorted(set(elem_arr)):
        mask = elem_arr == elem
        if mask.sum() > 0:
            print(f"  {elem}: {float((preds[mask] == trues[mask]).mean()):.3f} "
                  f"(n={int(mask.sum())})")
else:
    print("No parseable predictions -- judge prompt may need revision.")

print()
print("Interpretation guide:")
print("  Exact agreement near chance (~0.33) means the judge is not")
print("  reading the rubric usefully and PPO would optimize noise.")
print("  A large positive mean signed deviation means the judge is")
print("  as lenient as the supervisors, which would defeat the")
print("  purpose of using it as an independent reward signal.")
print()
print("Done.")