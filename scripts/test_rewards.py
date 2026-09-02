#!/usr/bin/env python
"""
Reward model calibration.

Part 1 checks both reward models on hand-written probes, including the
two failure modes calibration previously exposed: off-domain fluent
text scoring high on authenticity, and repetitive text scoring high on
perplexity.

Part 2 runs the blind rubric judge against all gold standard BIPs and
reports agreement, plus raw model output for unparseable cases.

The gold set is used here only as frozen few-shot demonstrations and as
a calibration reference for a frozen model. No gradients, no training.
It remains held out for final assessor evaluation.

Usage:
    sbatch --export=ALL,SCRIPT=scripts/test_rewards.py,ARGS="" scripts/train.slurm
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rewards.authenticity_reward import AuthenticityReward
from src.rewards.rubric_reward import RubricReward
from src.data.corpus import load_gold


# ── probes ───────────────────────────────────────────────────────────
# ordered so the expected ranking is obvious when reading output:
# two genuine BIPs, then three texts that should all score lower

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

    "We monitored our data. We monitored our data. We reviewed our "
    "data. We monitored our data. We reviewed our data regularly.",

    "The weather today is sunny with a high of 75 degrees. "
    "We recommend bringing sunscreen if you plan to be outside.",
]

labels = [
    "REAL BIP (Element6)",
    "REAL BIP (Element3)",
    "too short",
    "repetitive",
    "off-domain (weather)",
]

elements      = ["Element6", "Element3", "Element6", "Element6", "Element6"]
target_scores = [4,          4,          4,          4,          4]


print("=" * 74)
print("PART 1a: AUTHENTICITY REWARD (contrastive + repetition penalty)")
print("=" * 74)

auth = AuthenticityReward(
    base_model_name="Qwen/Qwen2.5-7B",
    adapter_path="models/BIPDomainSFT",
    device="cuda",
)

comp = auth.score_batch(candidates, batch_size=4, normalize=False,
                        return_components=True)

print(f"\n{'probe':<24} {'reward':>8} {'delta':>8} {'nll_ft':>8} "
      f"{'nll_base':>9} {'rep':>6}")
print("-" * 74)
for i, label in enumerate(labels):
    print(f"{label:<24} {comp['reward'][i]:>8.4f} {comp['delta'][i]:>8.4f} "
          f"{comp['nll_finetuned'][i]:>8.4f} {comp['nll_base'][i]:>9.4f} "
          f"{comp['repetition'][i]:>6.3f}")

print("\nWhat to check:")
print("  delta = nll_base - nll_finetuned. Real BIPs should have the")
print("  largest delta, because fine-tuning specifically lowered their")
print("  loss. Off-domain text should have delta near zero: both models")
print("  find generic fluent English equally easy. If the weather probe")
print("  still outranks the real BIPs, the contrastive signal is not")
print("  separating domain membership and PPO would reward drift.")

# ── rubric judge ─────────────────────────────────────────────────────

print()
print("=" * 74)
print("PART 1b: RUBRIC REWARD (Gemma 4 E4B, blind, few-shot)")
print("=" * 74)

gold_df = load_gold()
print(f"Gold standard loaded: {len(gold_df)} rows, "
      f"{gold_df['PersonId'].nunique()} principals")

few_shot = RubricReward.build_few_shot_examples(gold_df, max_examples=1)
print(f"Few-shot cells populated: {len(few_shot)} (element, score) pairs")

rubric = RubricReward(
    model_name="google/gemma-4-E4B-it",
    device="cuda",
    few_shot_examples=few_shot,
)

predicted, raw = rubric.predict(candidates, elements, return_raw=True)
print()
for label, p, r in zip(labels, predicted, raw):
    print(f"  {label:<24} predicted={p}  raw={r!r}")

# ── combined ─────────────────────────────────────────────────────────

print()
print("=" * 74)
print("PART 1c: COMBINED REWARD (alpha=0.5)")
print("=" * 74)
alpha = 0.5
rubric_rewards = [rubric._compute_reward(p, t)
                  for p, t in zip(predicted, target_scores)]
for label, a, rr in zip(labels, comp["reward"], rubric_rewards):
    print(f"  {label:<24} combined={alpha*a + (1-alpha)*rr:.4f}  "
          f"auth={a:.4f}  rubric={rr:.1f}")

# ── calibration on gold standard ─────────────────────────────────────

print()
print("=" * 74)
print("PART 2: RUBRIC JUDGE CALIBRATION ON GOLD STANDARD")
print("=" * 74)

gold_texts    = gold_df["Text"].tolist()
gold_elements = gold_df["Element_numberX"].tolist()
gold_scores   = gold_df["score"].astype(int).tolist()

gold_pred, gold_raw = rubric.predict(gold_texts, gold_elements,
                                     return_raw=True)

parseable = [(p, t) for p, t in zip(gold_pred, gold_scores) if p is not None]
n_unparseable = len(gold_pred) - len(parseable)

print(f"Scored:      {len(parseable)}/{len(gold_pred)}")
print(f"Unparseable: {n_unparseable}")

if n_unparseable:
    print("\nSample raw output from unparseable cases:")
    shown = 0
    for p, r in zip(gold_pred, gold_raw):
        if p is None and shown < 8:
            print(f"  {r!r}")
            shown += 1
    print("\nIf these are empty strings, the model is emitting only")
    print("special tokens and the prompt likely needs a clearer answer")
    print("cue. If they are truncated sentences, raise max_new_tokens.")

if parseable:
    preds = np.array([p for p, _ in parseable])
    trues = np.array([t for _, t in parseable])

    exact    = float((preds == trues).mean())
    adjacent = float((np.abs(preds - trues) <= 2).mean())
    signed   = float((preds - trues).mean())

    print(f"\nExact agreement:    {exact:.3f}")
    print(f"Adjacent agreement: {adjacent:.3f}")
    print(f"Mean signed dev:    {signed:+.3f}  "
          f"(negative = judge harsher than experts)")

    print("\nJudge prediction distribution:", dict(Counter(preds.tolist())))
    print("Expert score distribution:    ", dict(Counter(trues.tolist())))

    print("\nConfusion (rows = expert, cols = judge):")
    print(f"{'':>8}" + "".join(f"{c:>8}" for c in [0, 2, 4]))
    for t in [0, 2, 4]:
        row = [int(((trues == t) & (preds == p)).sum()) for p in [0, 2, 4]]
        print(f"{t:>8}" + "".join(f"{v:>8}" for v in row))

    elem_arr = np.array([e for e, p in zip(gold_elements, gold_pred)
                         if p is not None])
    print("\nPer-element exact agreement:")
    for elem in sorted(set(elem_arr)):
        mask = elem_arr == elem
        if mask.sum():
            print(f"  {elem}: {float((preds[mask] == trues[mask]).mean()):.3f} "
                  f"(n={int(mask.sum())})")

    print("\nHow to read this:")
    print("  Exact agreement near 0.33 is chance. A strongly negative")
    print("  signed deviation means the judge collapses toward low")
    print("  scores; PPO would then chase whatever it grudgingly")
    print("  accepts as a 4. A strongly positive one means it is as")
    print("  lenient as the supervisors, which removes the independence")
    print("  that makes it useful as a reward at all.")
else:
    print("No parseable predictions. The prompt needs revision before PPO.")

print("\nDone.")