#!/usr/bin/env python
"""
Tests both reward models together on a small set of examples.
Submit as a batch job when GPU nodes are busy.

Usage:
    python scripts/test_rewards.py
    sbatch --export=ALL,SCRIPT=scripts/test_rewards.py,ARGS="" scripts/train.slurm
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rewards.authenticity_reward import AuthenticityReward
from src.rewards.rubric_reward import RubricReward

# test texts: two good BIPs, one bad BIP, one completely off-topic
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

    "The weather today is sunny with a high of 75 degrees. "
    "We recommend bringing sunscreen if you plan to be outside.",
]

elements     = ["Element6", "Element3", "Element6", "Element6"]
target_scores = [4,          4,          4,           4]

print("=" * 60)
print("AUTHENTICITY REWARD (BIPDomainSFT perplexity)")
print("=" * 60)
auth = AuthenticityReward(
    base_model_name="Qwen/Qwen2.5-7B",
    adapter_path="models/BIPDomainSFT",
    device="cuda",
)
auth_rewards = auth.score_batch(candidates, batch_size=2)
for cand, reward in zip(candidates, auth_rewards):
    print(f"  auth={reward:.4f} | {cand[:70]}...")

print()
print("=" * 60)
print("RUBRIC REWARD (Gemma-2-2B few-shot judge)")
print("=" * 60)
rubric = RubricReward(
    model_name="google/gemma-2-2b-it",
    device="cuda",
)
rubric_rewards = rubric.score(candidates, elements, target_scores)
for cand, elem, target, reward in zip(
    candidates, elements, target_scores, rubric_rewards
):
    print(f"  rubric={reward:.1f} | {elem} | target={target} | "
          f"{cand[:60]}...")

print()
print("=" * 60)
print("COMBINED REWARD (alpha=0.5)")
print("=" * 60)
alpha = 0.5
for cand, auth_r, rubric_r in zip(candidates, auth_rewards, rubric_rewards):
    combined = alpha * auth_r + (1 - alpha) * rubric_r
    print(f"  combined={combined:.4f} | auth={auth_r:.4f} | "
          f"rubric={rubric_r:.1f} | {cand[:50]}...")

print("\nDone.")