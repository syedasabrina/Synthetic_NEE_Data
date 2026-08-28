#!/bin/bash
# Submits 4 PPO pilot runs in parallel, one per GPU, sweeping alpha.
# Run this after generator SFT warmup and BIPDomainSFT are both ready.
#
# Usage:
#   bash scripts/submit_ppo_pilots.sh

PROJECT_DIR=/scratch/sakter6/synthetic/Synthetic_NEE_Data
cd $PROJECT_DIR

# alpha=1.0 -> authenticity only (Ablation 1 equivalent)
# alpha=0.0 -> rubric only (Ablation 2 equivalent)
# alpha=0.5 and 0.7 -> balanced configurations to compare
CONFIGS=(
  "0.5 0.1"
  "0.7 0.1"
  "0.3 0.1"
  "0.5 0.05"
)

for cfg in "${CONFIGS[@]}"; do
  read -r alpha beta <<< "$cfg"
  echo "Submitting pilot: alpha=$alpha beta=$beta"
  sbatch --export=ALL,SCRIPT=scripts/run_ppo_pilot.py,ARGS="--alpha $alpha --beta $beta --steps 300" scripts/train.slurm
done

echo "All 4 pilots submitted. Check with: squeue -u sakter6"