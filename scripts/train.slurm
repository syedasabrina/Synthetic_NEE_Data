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

export PROJECT_DIR=/scratch/sakter6/synthetic/Synthetic_NEE_Data
export DATA_DIR=$PROJECT_DIR/data
export MODELS_DIR=$PROJECT_DIR/models
export LOGS_DIR=$PROJECT_DIR/logs
export HF_HOME=/scratch/sakter6/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export WANDB_API_KEY=your_actWANDB_API_KEY=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
export WANDB_PROJECT=synthetic-nee-bip
export WANDB_DIR=$LOGS_DIR/wandb
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH


source /scratch/sakter6/bip-env/bin/activate

cd /scratch/sakter6/synthetic/Synthetic_NEE_Data

python $SCRIPT $ARGS


###sbatch --export=ALL,SCRIPT=scripts/run_style_learner.py,ARGS="" scripts/train.slurm
