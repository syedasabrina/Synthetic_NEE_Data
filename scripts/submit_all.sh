#!/bin/bash
# Submits both experimental chains as dependent SLURM jobs.
#
# Both consume the same reward models and the same GeneratorSFT
# checkpoint, and neither writes to the other's output directory, so
# they run in parallel without interference. Whichever produces the
# better generator becomes the primary result; the other becomes an
# ablation showing what the optimization method contributed.
#
# Everything waits on a reward calibration job, since both chains
# optimize against the same two reward models and running either
# against a miscalibrated judge wastes the compute.
#
# Usage:
#   bash scripts/submit_all.sh

set -e
cd /scratch/sakter6/synthetic/Synthetic_NEE_Data

echo "Submitting reward calibration..."
CAL=$(sbatch --parsable \
  --export=ALL,SCRIPT=scripts/test_rewards.py,ARGS="" \
  scripts/train.slurm)
echo "  calibration: $CAL"

# ── chain A: best-of-n, three sequential rounds ─────────────────────
# each round samples from the previous round's adapter, so these must
# run in sequence rather than in parallel

echo ""
echo "Submitting best-of-n chain..."

BON1=$(sbatch --parsable --dependency=afterok:$CAL \
  --export=ALL,SCRIPT=scripts/run_bon_round.py,ARGS="--round 1 --generator models/GeneratorSFT --output models/BoN_round1 --n_prompts 500" \
  scripts/train.slurm)
echo "  round 1: $BON1"

BON2=$(sbatch --parsable --dependency=afterok:$BON1 \
  --export=ALL,SCRIPT=scripts/run_bon_round.py,ARGS="--round 2 --generator models/BoN_round1 --output models/BoN_round2 --n_prompts 500" \
  scripts/train.slurm)
echo "  round 2: $BON2"

BON3=$(sbatch --parsable --dependency=afterok:$BON2 \
  --export=ALL,SCRIPT=scripts/run_bon_round.py,ARGS="--round 3 --generator models/BoN_round2 --output models/BoN_round3 --n_prompts 500" \
  scripts/train.slurm)
echo "  round 3: $BON3"

# ── chain B: PPO pilots, four configurations in parallel ────────────
# all start from the same SFT checkpoint, so these are independent

echo ""
echo "Submitting PPO pilot grid..."

for cfg in "0.5 0.1" "0.7 0.1" "0.3 0.1" "0.5 0.05"; do
  read -r alpha kl <<< "$cfg"
  JOB=$(sbatch --parsable --dependency=afterok:$CAL \
    --export=ALL,SCRIPT=scripts/run_ppo_pilot.py,ARGS="--alpha $alpha --kl_coef $kl --steps 300" \
    scripts/train.slurm)
  echo "  alpha=$alpha kl=$kl: $JOB"
done

echo ""
echo "Submitted. Monitor with: squeue -u sakter6"
echo ""
echo "What to read when they finish:"
echo "  calibration -- judge parse rate should be near 171/171 and the"
echo "    weather probe should score well below the real BIPs"
echo "  bon round stats -- models/BoN_round*/round_stats.json, watch"
echo "    mean_combined_accepted rise across rounds"
echo "  ppo pilots -- s/step and the reward trajectory; any divergence"
echo "    warning means that configuration is not learning"