#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=bip-synthetic
#
#SBATCH --output=/scratch/sakter6/synthetic/Synthetic_NEE_Data/logs/slurm/%x.%j.out
#SBATCH --error=/scratch/sakter6/synthetic/Synthetic_NEE_Data/logs/slurm/%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem-per-cpu=30G
#SBATCH --export=ALL
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sakter6@gmu.edu

set echo
umask 0027

nvidia-smi

module load gnu10
module load cuda

source /scratch/sakter6/bip-env/bin/activate

cd /scratch/sakter6/synthetic/Synthetic_NEE_Data

python $SCRIPT $ARGS


###sbatch --export=ALL,SCRIPT=scripts/run_style_learner.py,ARGS="" scripts/train.slurm
