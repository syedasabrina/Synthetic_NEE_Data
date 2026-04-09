#!/usr/bin/env bash
set -e

# ── directories ──────────────────────────────────────────────────────
mkdir -p data/{raw,anchors,synthetic/{candidates,accepted,rejected},gold}
mkdir -p src/{data,generation,judges,training,evaluation,audit,utils}
mkdir -p configs/{generation,training,evaluation}
mkdir -p scripts
mkdir -p notebooks
mkdir -p tests/{unit,integration}
mkdir -p logs/{generation,judges,training,audit}
mkdir -p results/{assessor,audit,diagnostics}
mkdir -p docs

# ── touch all __init__.py ─────────────────────────────────────────────
touch src/__init__.py
touch src/data/__init__.py
touch src/generation/__init__.py
touch src/judges/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/audit/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

# ── src/data ─────────────────────────────────────────────────────────
touch src/data/corpus.py          # load + iterate 18k BIP corpus
touch src/data/anchor_sampler.py  # controlled sampling by score level
touch src/data/dataset.py         # HF Dataset wrappers for training
touch src/data/filters.py         # dedup, similarity threshold filtering

# ── src/generation ───────────────────────────────────────────────────
touch src/generation/generator.py       # Module 2: rubric-conditioned generation
touch src/generation/prompt_builder.py  # build prompts from rubric + anchor + score
touch src/generation/pipeline.py        # orchestrate full generation loop

# ── src/judges ───────────────────────────────────────────────────────
touch src/judges/authenticity_judge.py  # Module 3 Check 1: style model judge
touch src/judges/rubric_judge.py        # Module 3 Check 2: rubric alignment judge
touch src/judges/verdict.py             # combine verdicts, log reason codes
touch src/judges/calibration.py         # judge calibration on gold standard set

# ── src/training ─────────────────────────────────────────────────────
touch src/training/style_learner.py  # Module 1: causal LM fine-tune on 18k BIPs
touch src/training/assessor.py       # Module 4: classification head fine-tune
touch src/training/ordinal_loss.py   # ordinal-aware loss function
touch src/training/trainer.py        # HF Trainer wrapper with LoRA / PEFT setup

# ── src/evaluation ───────────────────────────────────────────────────
touch src/evaluation/metrics.py       # QWK, exact/adjacent agreement, bootstrap CI
touch src/evaluation/calibration.py   # assessor calibration analysis on gold set
touch src/evaluation/difficulty.py    # synthetic vs real difficulty gap analysis
touch src/evaluation/error_analysis.py

# ── src/audit ────────────────────────────────────────────────────────
touch src/audit/scorer.py         # apply assessor to full 18k corpus
touch src/audit/deviation.py      # compute supervisor vs model deviation
touch src/audit/statistics.py     # t-test, KS test, regression on deviations
touch src/audit/subgroup.py       # subgroup analysis if metadata available
touch src/audit/report.py         # generate audit summary report

# ── src/utils ────────────────────────────────────────────────────────
touch src/utils/config.py      # dataclasses for all config objects
touch src/utils/logging.py     # structured logging setup
touch src/utils/io.py          # json/jsonl/csv read-write helpers
touch src/utils/embeddings.py  # embedding util (cosine sim, spread metrics)
touch src/utils/reproducibility.py  # seed setting, run ID generation

# ── configs ──────────────────────────────────────────────────────────
touch configs/generation/default.py
touch configs/training/style_learner.py
touch configs/training/assessor.py
touch configs/evaluation/default.py

# ── scripts (CLI entry points) ────────────────────────────────────────
touch scripts/run_generation.py    # end-to-end: sample anchors → generate → judge → save
touch scripts/run_style_learner.py # train Module 1
touch scripts/run_assessor.py      # train Module 4 (one condition at a time)
touch scripts/run_evaluation.py    # evaluate assessor on gold standard set
touch scripts/run_audit.py         # apply assessor to 18k, run deviation analysis
touch scripts/run_diagnostics.py   # pipeline diagnostics: rejection rates, distributions

# ── notebooks ────────────────────────────────────────────────────────
touch notebooks/01_corpus_exploration.ipynb
touch notebooks/02_pilot_review.ipynb
touch notebooks/03_synthetic_distribution.ipynb
touch notebooks/04_model_comparison.ipynb
touch notebooks/05_audit_analysis.ipynb

# ── tests ─────────────────────────────────────────────────────────────
touch tests/unit/test_anchor_sampler.py
touch tests/unit/test_prompt_builder.py
touch tests/unit/test_verdict.py
touch tests/unit/test_metrics.py
touch tests/unit/test_ordinal_loss.py
touch tests/integration/test_generation_pipeline.py
touch tests/integration/test_judge_pipeline.py

echo "Repo skeleton created."
