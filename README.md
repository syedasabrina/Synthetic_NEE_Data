# BIP Synthetic Data Generation for Principal Assessment

Rubric-aligned synthetic BIP generation with dual-model verification,
trained assessor evaluation, and scoring deviation audit on the
NEE supervisor-scored corpus.

---

## Project Structure

```
├── configs/                   # per-module config defaults
│   ├── generation/
│   ├── training/
│   └── evaluation/
├── data/
│   ├── raw/                   # 18k real BIPs (not committed)
│   ├── anchors/               # sampled anchor pool
│   ├── synthetic/
│   │   ├── candidates/        # all generated BIPs before judging
│   │   ├── accepted/          # passed both judges
│   │   └── rejected/          # failed with reason codes
│   └── gold/                  # 23 gold standard BIPs (not committed)
├── docs/                      # project scope, rubric, notes
├── logs/                      # generation, judge, training, audit logs
│   ├── generation/
│   ├── judges/
│   ├── BIPDomainSFT/
│   ├── audit/
│   ├── slurm/
│   └── wandb/
├── models/                    # checkpoints (not committed)
├── notebooks/                 # exploration and analysis
├── results/
│   ├── assessor/              # evaluation results per condition
│   ├── audit/                 # deviation analysis outputs
│   └── diagnostics/           # pipeline diagnostics
├── scripts/                   # CLI entry points
│   ├── run_BIPDomainSFT.py
│   ├── run_generation.py
│   ├── run_assessor.py
│   ├── run_evaluation.py
│   ├── run_audit.py
│   ├── run_diagnostics.py
│   └── train.slurm
├── src/
│   ├── data/                  # corpus loading, anchor sampling, HF datasets
│   │   ├── corpus.py
│   │   ├── anchor_sampler.py
│   │   ├── dataset.py
│   │   └── filters.py
│   ├── generation/            # Module 2: rubric-conditioned generator
│   ├── judges/                # Module 3: authenticity + rubric judges
│   ├── training/              # Module 1: BIPDomainSFT, Module 4: assessor
│   │   ├── BIPDomainSFT.py
│   │   ├── assessor.py
│   │   ├── ordinal_loss.py
│   │   └── trainer.py
│   ├── evaluation/            # metrics, calibration, difficulty, error analysis
│   ├── audit/                 # deviation scoring, statistical tests, subgroup
│   └── utils/                 # config dataclasses, logging, io, embeddings
│       └── config.py
└── tests/
    ├── unit/
    └── integration/
```

---

## Pipeline Overview

```
18k real BIPs
    |
    |--► [Module 1] BIPDomainSFT -- Qwen2.5-7B fine-tuned on 12,508 BIP texts
    |         └--► authenticity judge (Module 3, Check 1)
    |
    └--► anchor pool (element + score stratified sampling)
              |
              v
         [Module 2] Rubric-conditioned generation -- Gemini 2.0 Flash
         input: rubric criteria + target score + element + anchor BIP
         output: candidate synthetic BIP
              |
              v
         [Module 3] Dual judge
         |-- Check 1: authenticity  (BIPDomainSFT -- different family from generator)
         └-- Check 2: rubric alignment  (Gemini 2.0 Flash)
              |
         ┌----┴-----------┐
       fail              pass
         |                 |
      regenerate      training pool
                           |
                           v
                   [Module 4] Assessor training
                   4 conditions: synthetic / real /
                   balanced-real / hybrid
                           |
                           v
                   Evaluation on 23 gold standard BIPs
                   QWK, calibration, difficulty gap
                           |
                           v
                   Scoring deviation audit
                   apply to 18k corpus, statistical tests,
                   subgroup analysis (district, rurality, year)
```

---

## Experimental Conditions

| Condition | Training Data | Purpose |
|---|---|---|
| A -- synthetic only | Synthetic (verified, balanced) | Primary condition |
| B -- real noisy | Real BIPs, supervisor scores | Lower-bound baseline |
| C -- balanced real | Real BIPs, reweighted to match synthetic distribution | Distribution control |
| D -- hybrid | Synthetic + real combined | Upper-bound test |

---

## Ablations

| Ablation | What is removed | What it isolates |
|---|---|---|
| 1 | Authenticity filter | Contribution of Check 1 |
| 2 | Rubric alignment filter | Contribution of Check 2 |
| 3 | Model diversity (self-judge) | Contribution of architectural separation |
| 4 | Anchor conditioning | Contribution of topical diversity |

---

## Corpus Statistics

| Metric | Value |
|---|---|
| Total BIPs (raw) | 19,719 |
| Usable BIPs for BIPDomainSFT | 12,508 |
| Scored BIPs in anchor pool | 9,398 |
| Score 0 anchors | 31 |
| Score 2 anchors | 406 |
| Score 4 anchors | 8,961 |
| Unique principals | 1,086 |
| Unique districts | 156 |
| School years | 2015-2016 to 2023-2024 |
| Rubric elements | 7 |

---

## Setup

```bash
# clone and enter
git clone <repo>
cd <repo>

# activate venv (Hopper)
source /scratch/sakter6/bip-env/bin/activate

# install dependencies
pip install -r requirements.txt

# set environment variables
export HF_HOME=/scratch/sakter6/.cache/huggingface
export WANDB_PROJECT=synthetic-nee-bip
export PYTHONPATH=/scratch/sakter6/synthetic/Synthetic_NEE_Data:$PYTHONPATH
```

---

## Running the Pipeline

### Interactive (CPU -- data and preprocessing)

```bash
salloc -p normal --nodes=1 --ntasks-per-node=12 --mem=5GB --time=0-0:30:00
```

### Interactive smoke test (MIG GPU -- verify pipeline logic)

```bash
salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=2 --gres=gpu:1g.10gb:1 --mem=15gb -t 0-01:00:00
python scripts/run_BIPDomainSFT.py --data data/raw/bips.csv --smoke_test
```

### Full training runs (A100 80GB via sbatch)

```bash
# 1. train BIPDomainSFT (Module 1)
sbatch --export=ALL,SCRIPT=scripts/run_BIPDomainSFT.py,ARGS="--data data/raw/bips.csv --model Qwen/Qwen2.5-7B" scripts/train.slurm

# 2. generate + judge synthetic BIPs (Modules 2 and 3)
sbatch --export=ALL,SCRIPT=scripts/run_generation.py,ARGS="" scripts/train.slurm

# 3. train assessor -- one condition at a time (Module 4)
sbatch --export=ALL,SCRIPT=scripts/run_assessor.py,ARGS="--condition synthetic_only" scripts/train.slurm
sbatch --export=ALL,SCRIPT=scripts/run_assessor.py,ARGS="--condition real_noisy" scripts/train.slurm
sbatch --export=ALL,SCRIPT=scripts/run_assessor.py,ARGS="--condition balanced_real" scripts/train.slurm
sbatch --export=ALL,SCRIPT=scripts/run_assessor.py,ARGS="--condition hybrid" scripts/train.slurm

# 4. evaluate on gold standard set
sbatch --export=ALL,SCRIPT=scripts/run_evaluation.py,ARGS="--condition synthetic_only" scripts/train.slurm

# 5. scoring deviation audit
sbatch --export=ALL,SCRIPT=scripts/run_audit.py,ARGS="" scripts/train.slurm

# 6. pipeline diagnostics
python scripts/run_diagnostics.py
```

---

## Monitoring

Training runs are logged to WandB under the `synthetic-nee-bip` project.

To check job status:
```bash
squeue -u sakter6
```

To monitor GPU utilization on a running node:
```bash
ssh <nodename>
watch -n 1 nvidia-smi
```

Slurm logs are written to `logs/slurm/`.

---

## Data

Raw BIPs and gold standard BIPs are not committed to this repository.

Accepted synthetic BIPs and all judge verdict logs will be released.

---

## Citation

TBD upon publication.