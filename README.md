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
├── models/                    # checkpoints (not committed)
├── notebooks/                 # exploration and analysis
├── results/
│   ├── assessor/              # evaluation results per condition
│   ├── audit/                 # deviation analysis outputs
│   └── diagnostics/           # pipeline diagnostics
├── scripts/                   # CLI entry points
│   ├── run_generation.py
│   ├── run_style_learner.py
│   ├── run_assessor.py
│   ├── run_evaluation.py
│   ├── run_audit.py
│   └── run_diagnostics.py
├── src/
│   ├── data/                  # corpus loading, anchor sampling, HF datasets
│   ├── generation/            # Module 2: rubric-conditioned generator
│   ├── judges/                # Module 3: authenticity + rubric judges
│   ├── training/              # Modules 1 & 4: style learner + assessor
│   ├── evaluation/            # metrics, calibration, difficulty, error analysis
│   ├── audit/                 # deviation scoring, statistical tests, subgroup
│   └── utils/                 # config dataclasses, logging, io, embeddings
└── tests/
    ├── unit/
    └── integration/
```

---

## Pipeline Overview

```
18k real BIPs
    │
    ├──► [Module 1] Fine-tune small LM (style learner)
    │         └──► authenticity judge (Module 3, Check 1)
    │
    └──► anchor pool (score-balanced sampling)
              │
              ▼
         [Module 2] Rubric-conditioned generation
         (rubric + target score + anchor → candidate BIP)
              │
              ▼
         [Module 3] Dual judge
         ├── Check 1: authenticity (style model)
         └── Check 2: rubric alignment (generation model)
              │
         ┌───┴───────────┐
       fail             pass
         │                │
      regenerate     training pool
                          │
                          ▼
                   [Module 4] Assessor training
                   (4 conditions: synthetic / real /
                    balanced-real / hybrid)
                          │
                          ▼
                   Evaluation on 23 gold BIPs
                   (QWK, calibration, difficulty gap)
                          │
                          ▼
                   Scoring deviation audit
                   (apply to 18k, statistical tests,
                    subgroup analysis)
```

---

## Experimental Conditions

| Condition | Training Data | Purpose |
|---|---|---|
| A — synthetic only | Synthetic (verified, balanced) | Primary condition |
| B — real noisy | Real BIPs, supervisor scores | Lower-bound baseline |
| C — balanced real | Real BIPs, reweighted to balance | Distribution control |
| D — hybrid | Synthetic + real combined | Upper-bound test |

---

## Ablations

| Ablation | What is removed | What it isolates |
|---|---|---|
| 1 | Authenticity filter | Contribution of Check 1 |
| 2 | Rubric alignment filter | Contribution of Check 2 |
| 3 | Model diversity (self-judge) | Contribution of architectural separation |
| 4 | Anchor conditioning | Contribution of topical diversity |

---

## Setup

```bash
# clone and enter
git clone <repo>
cd <repo>

# activate your venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# create folder structure (if starting fresh)
bash setup_repo.sh
```

---

## Running the Pipeline

<!-- ```bash
# 1. train style learner (Module 1)
python scripts/run_style_learner.py

# 2. generate + judge synthetic BIPs (Modules 2 & 3)
python scripts/run_generation.py

# 3. train assessor — one condition at a time
python scripts/run_assessor.py --condition synthetic_only
python scripts/run_assessor.py --condition real_noisy
python scripts/run_assessor.py --condition balanced_real
python scripts/run_assessor.py --condition hybrid

# 4. evaluate on gold standard set
python scripts/run_evaluation.py --condition synthetic_only

# 5. run scoring deviation audit
python scripts/run_audit.py

# 6. pipeline diagnostics
python scripts/run_diagnostics.py -->
```

---

## Data

Raw BIPs and gold standard BIPs are not committed to this repository.
See `docs/data_access.md` for access instructions.

Accepted synthetic BIPs and all judge verdict logs will be released
upon publication.

---

## Citation

TBD upon publication.
