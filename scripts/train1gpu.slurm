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
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --export=ALL
#SBATCH --time=0-02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sakter6@gmu.edu

set echo
umask 0027

nvidia-smi

module load gnu10
module load cuda

export PROJECT_DIR=/scratch/sakter6/synthetic/Synthetic_NEE_Data
export DATA_DIR=$PROJECT_DIR/data
export MODELS_DIR=$PROJECT_DIR/models
export LOGS_DIR=$PROJECT_DIR/logs
export HF_HOME=/scratch/sakter6/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export WANDB_API_KEY=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
export WANDB_PROJECT=synthetic-nee-bip
export WANDB_DIR=$LOGS_DIR/wandb
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH


# reduces fragmentation-driven OOM when several large models share a device
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /scratch/sakter6/bip-env/bin/activate

cd /scratch/sakter6/synthetic/Synthetic_NEE_Data

python $SCRIPT $ARGS


### sbatch --export=ALL,SCRIPT=scripts/run_BIPDomainSFT.py,ARGS="--data data/raw/bips.csv --model Qwen/Qwen2.5-7B" scripts/train.slurm
### sbatch --export=ALL,SCRIPT=scripts/test_rewards.py,ARGS="" scripts/train.slurm
### sbatch --export=ALL,SCRIPT=scripts/run_bon_round.py,ARGS="--round 1 --output models/BoN_round1 --retrain_only" scripts/train.slurm

