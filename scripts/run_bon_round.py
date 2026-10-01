#!/usr/bin/env python
"""
Runs one round of best-of-n generation, scoring, selection and (optionally)
generator retraining.

Every candidate is written to <output>/candidates.jsonl as it is scored.
A killed job resumes by running the same command again: finished prompts
are skipped and the prompt specs are a function of (seed, prompt index), so
the resumed run matches an uninterrupted one. Settings that affect
generation are recorded in run_config.json and a mismatch aborts.

Selection can be redone without a GPU:

    python scripts/run_bon_round.py --round 5 --generator <adapter> \
        --output models/BoN_round5 --n_prompts 500 --reselect \
        --min_judge_reward 0.5 --max_title_case 0.35

Usage:
    python scripts/run_bon_round.py --round 1 \
        --generator models/GeneratorSFT --output models/BoN_round1 \
        --n_prompts 500 --skip_retrain

    python scripts/run_bon_round.py --round 1 \
        --generator models/GeneratorSFT --output models/BoN_round1 \
        --retrain_only

Rounds 1 to 4 used the old generator and cannot be resumed or compared
with runs from this version.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.data.corpus import load, for_anchor_pool, load_gold
from src.training.bon_round import (
    BestOfNRound, SelectionConfig, finalize_round, retrain_generator,
)
from src.utils.config import GeneratorSFTConfig
from src.utils.text_quality import QualityGate

p = argparse.ArgumentParser(description="One best-of-n round")
p.add_argument("--data", default="data/raw/bips.csv")
p.add_argument("--round", type=int, required=True)
p.add_argument("--generator", required=True,
               help="Adapter to sample from, and to continue training from at retrain time")
p.add_argument("--output", required=True)
p.add_argument("--n_prompts", type=int, default=500)
p.add_argument("--n_candidates", type=int, default=8)
p.add_argument("--elements", nargs="+", default=None,
               help="Restrict to these elements, e.g. Element1 Element2 Element6 Element7")

# decoding
p.add_argument("--max_new_tokens", type=int, default=512)
p.add_argument("--temperature", type=float, default=0.9)
p.add_argument("--top_p", type=float, default=0.95)
p.add_argument("--repetition_penalty", type=float, default=1.0)
p.add_argument("--no_repeat_ngram_size", type=int, default=0)
p.add_argument("--max_prompt_tokens", type=int, default=640)

# selection
p.add_argument("--keep_top_k", type=int, default=2)
p.add_argument("--min_judge_reward", type=float, default=1.0,
               help="1.0 = judge must name the target exactly, 0.5 = one level off allowed, "
                    "negative = no judge gate")
p.add_argument("--min_auth_delta", type=float, default=None,
               help="Absolute authenticity floor from scripts/score_real_reference.py")
p.add_argument("--rank_by", choices=["auth", "combined"], default="auth")
p.add_argument("--alpha", type=float, default=0.5)

# quality gate (defaults are provisional, see src/utils/text_quality.py)
p.add_argument("--max_rt", type=float, default=7.0, help="run-together words per 100")
p.add_argument("--max_title_case", type=float, default=0.70)
p.add_argument("--min_marks", type=float, default=1.0, help="sentence marks per 100 words")
p.add_argument("--no_require_stop", action="store_true",
               help="Do not require an EOS before the token cap")
p.add_argument("--require_end_punct", action="store_true")
p.add_argument("--skip_scoring_failed", action="store_true",
               help="Do not run the reward models on candidates that fail the gate (saves time)")

# retrain
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--batch_size", type=int, default=2)
p.add_argument("--skip_retrain", action="store_true", help="Generate and score only")
p.add_argument("--retrain_only", action="store_true",
               help="Skip generation; retrain from existing accepted.jsonl")
p.add_argument("--reselect", action="store_true",
               help="Rebuild accepted.jsonl from candidates.jsonl with the settings above; no GPU needed")
args = p.parse_args()

gate = QualityGate(
    max_rt_per_100=args.max_rt,
    max_title_case=args.max_title_case,
    min_marks_per_100=args.min_marks,
    require_stop=not args.no_require_stop,
    require_end_punct=args.require_end_punct,
)
selection = SelectionConfig(
    keep_top_k=args.keep_top_k,
    min_judge_reward=None if args.min_judge_reward < 0 else args.min_judge_reward,
    min_auth_delta=args.min_auth_delta,
    rank_by=args.rank_by,
    alpha=args.alpha,
)

print("=" * 70)
print(f"BEST-OF-N ROUND {args.round}")
print(f"  output:     {args.output}")
print(f"  generator:  {args.generator}")
print(f"  gate:       {gate}")
print(f"  selection:  {selection}")
print(f"  visible GPUs: {torch.cuda.device_count()}")
print("=" * 70)

start = time.time()
out_dir = Path(args.output)

if args.reselect:
    accepted = finalize_round(out_dir, args.n_prompts, gate, selection)
    print(f"Reselected {len(accepted)} rows.")
    if args.skip_retrain:
        sys.exit(0)

elif args.retrain_only:
    path = out_dir / "accepted.jsonl"
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

    runner = BestOfNRound(
        anchor_df=anchor_df,
        gold_df=gold_df,
        generator_adapter=args.generator,
        output_dir=args.output,
        n_candidates=args.n_candidates,
        selection=selection,
        gate=gate,
        score_gated_out=not args.skip_scoring_failed,
        elements=args.elements,
        max_prompt_tokens=args.max_prompt_tokens,
        sampler_kwargs=dict(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        ),
        seed=42 + args.round,
    )

    accepted = runner.run(n_prompts=args.n_prompts)

    gen_elapsed = time.time() - start
    print(f"\nGeneration and scoring: {gen_elapsed / 60:.1f} min "
          f"({gen_elapsed / max(args.n_prompts, 1):.2f} s/prompt)")

    if len(accepted) == 0:
        print("No candidates accepted. Nothing to retrain on. "
              "Inspect round_stats.json (gate_fail_counts) and candidates.jsonl.")
        sys.exit(1)

    if args.skip_retrain:
        print("skip_retrain set; stopping after generation.")
        sys.exit(0)

    # free the generator and both reward models before training
    runner.release()
    del runner
    gc.collect()
    torch.cuda.empty_cache()

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
print(f"\nRound {args.round} complete in {total / 60:.1f} min")
print(f"New generator adapter: {args.output}")
