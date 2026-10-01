#!/bin/bash
# Stage 1: diagnostics only. Nothing here trains or changes a model, and
# none of the jobs depend on each other, so all four can run at once.
#
# This replaces submit_all.sh, which chained PPO and the old rounds.
#
#   probe_round3    decoding probe on the last BoN adapter
#   probe_sft       decoding probe on the SFT adapter, same prompts and seed
#   judge_gold      current judge on held-out gold, per element and level
#   real_reference  gate metrics on real BIPs and authenticity on held-out gold
#
# Read these four outputs before choosing decoder settings, gate thresholds
# or the authenticity floor, and before submitting any generation job.
#
# Usage:  bash scripts/submit_stage1.sh

set -e
cd /scratch/sakter6/synthetic/Synthetic_NEE_Data
mkdir -p results logs/slurm

POOL=models/BoN_round3/accepted.jsonl

P3=$(sbatch --parsable --time=02:00:00 \
  --export=ALL,SCRIPT=scripts/probe_decoding.py,ARGS="--adapter models/BoN_round3 --pool $POOL --out results/probe_round3.jsonl --max_new_tokens 400" \
  scripts/train.slurm)
echo "probe_round3:   $P3"

PS=$(sbatch --parsable --time=02:00:00 \
  --export=ALL,SCRIPT=scripts/probe_decoding.py,ARGS="--adapter models/GeneratorSFT --pool $POOL --out results/probe_sft.jsonl --max_new_tokens 400" \
  scripts/train.slurm)
echo "probe_sft:      $PS"

JG=$(sbatch --parsable --time=01:00:00 \
  --export=ALL,SCRIPT=scripts/eval_judge_gold.py,ARGS="--out results/judge_gold_e4b" \
  scripts/train.slurm)
echo "judge_gold:     $JG"

RR=$(sbatch --parsable --time=01:00:00 \
  --export=ALL,SCRIPT=scripts/score_real_reference.py,ARGS="--auth --out results/real_reference.json" \
  scripts/train.slurm)
echo "real_reference: $RR"

echo
echo "Monitor: squeue -u sakter6 --start"
echo "Logs:    logs/slurm/bip-synthetic.<jobid>.out"
echo "Outputs: results/probe_round3.jsonl results/probe_sft.jsonl"
echo "         results/judge_gold_e4b_rows.csv results/judge_gold_e4b.json"
echo "         results/real_reference.json"
