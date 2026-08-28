# Rubric-Aligned Synthetic BIP Generation via RLHF

Reinforcement learning from rubric feedback for synthetic educational
assessment data generation, applied to auditing principal assessment
scores in the NEE corpus.

---

## The Problem

Building Improvement Plans (BIPs) are annual documents written by
school principals and scored by supervisors against the NEE rubric.
The corpus contains two things that cannot currently be used together:

**Real BIPs carry authentic writing but unreliable scores.** 19,719
raw responses written by 1,086 principals across nine school years.
The text is genuine. The supervisor scores attached to it are not
trustworthy.

**The rubric carries a reliable score specification but no data.** It
defines what each score level means for each of seven elements, but
there is no corpus attached to it.

The gap between these two is measurable and large. Comparing the
supervisor-scored corpus against the expert-adjudicated gold standard
set on the same kind of work:

| Score | Supervisor corpus | Expert gold standard |
|---|---|---|
| 0 | 31 (0.3%) | 21 (12%) |
| 2 | 374 (4%) | 90 (53%) |
| 4 | 8,500 (95%) | 60 (35%) |

Supervisors award the top score on 95% of BIPs. Experts award it on
35%. Experts assign score 2 more than half the time, where supervisors
do so 4% of the time. This 60-point gap at the top score level is the
project's motivating evidence, visible in the data before any model is
trained.

---

## The Approach

The pipeline exploits the two data properties separately and combines
them only through a reward signal, so noisy supervisor scores never
enter as a training label at any stage.

```
Real BIP corpus (12k texts, scores ignored)
        |
        v
  BIPDomainSFT (Qwen2.5-7B + LoRA)
  learns authentic principal writing style
        |
        +---> Authenticity reward model (frozen)
              perplexity + repetition penalty
                          |
NEE rubric text            |
        |                  |
        v                  |
  Rubric judge (Gemma 4 E4B, frozen, blind)
  predicts score independently, few-shot anchored
        |                  |
        +---> Rubric alignment reward
                          |
                          v
              Combined reward r = a*r_auth + (1-a)*r_rubric - b*KL
                          |
                          v
        Generator (Gemma 4 E4B + LoRA)
        SFT warmup, then PPO against combined reward
                          |
                          v
              Synthetic BIP pool
        rubric-aligned AND authentic by construction
                          |
                          v
        Assessor (Qwen2.5-7B, 4 training conditions)
                          |
                          v
   Evaluation on held-out gold set, then corpus-wide audit
```

The key property: the generator optimizes directly for both
objectives rather than being filtered after the fact. Rubric
conditioning comes from the rubric specification, style conditioning
comes from the real corpus, and supervisor scores are used only as
the audit target at the very end.

---

## Model Assignment

| Role | Model | Family | Trained? |
|---|---|---|---|
| BIPDomainSFT / authenticity reward | Qwen2.5-7B | Alibaba | Yes, LoRA |
| Rubric alignment reward | Gemma 4 E4B | Google | No, frozen few-shot |
| Generator (SFT warmup, then PPO) | Gemma 4 E4B | Google | Yes, LoRA |
| Final assessor | Qwen2.5-7B | Alibaba | Yes, from BIPDomainSFT |

Qwen and Gemma are architecturally distinct families. The generator
and rubric judge share a base model but diverge during PPO, and the
judge is always frozen, so the two reward channels remain independent.

**Gemma 4 LoRA caveat.** Gemma 4 is multimodal by architecture even in
text-only use. Its vision and audio towers wrap attention projections
in Linear subclasses that match simple leaf names like `q_proj`. Using
exact-match `target_modules` risks PEFT adapting the wrong layers or
silently aborting adapter injection. All Gemma 4 LoRA configs use an
explicit regex restricted to the text tower:

```
.*language_model.layers.\d+.(self_attn.(q|k|v|o)_proj|mlp.(gate|up|down)_proj)
```

Qwen models keep the simple list form. Two separate config
dataclasses exist for this reason: `LoRAConfig` and `LoRAConfigGemma4`.

---

## Data

### Score mapping

The NEE rubric defines three anchors: 0 (none or little), 2 (vague or
minimal), 4 (extensive or fully described). Supervisors occasionally
use 1 and 3 as informal interpolations. These are mapped to the
nearest defined anchor before any processing:

- 1 maps to 2, a supervisor recording "minimal but not zero"
- 3 maps to 4, a supervisor recording "mostly good"

This is a fact about the rubric, not a tunable parameter. Applied
identically to the training corpus and the gold standard set.

### Gold standard set

`data/gold/combined_annotations_with_SupervisorScore.csv`, 173 rows
covering 24 principals across all seven elements.

**Encoding.** The file is latin-1, not UTF-8. Reading it as UTF-8
fails on byte 0xd0.

**Score column.** `Scaled_Annotator_Rating` is the ground truth, not
`Annotator_Rating`. Annotators worked on a raw 7-point scale (0 to 6)
which was then normalized per annotator onto the supervisor's 0 to 4
scale. The crosstab between the two columns is non-deterministic:
raw rating 3 maps to scaled 2, 3, and 4 in different rows, and both
raw 3 and raw 4 collapse onto scaled 2. This confirms per-annotator
normalization rather than a global formula. Seven annotators appear
with uneven row counts, consistent with correcting for individual
severity. Only the scaled column is directly comparable to
`Supervisor_Score_x` and to model predictions.

**Unscoreable rows.** Two rows carry `Scaled_Annotator_Rating == -1`,
outside the 0 to 4 scale. These are dropped, leaving 171 usable rows.

### Evaluation isolation

All 24 gold standard PersonIds appear in the raw corpus. Without
exclusion, BIPDomainSFT would train on text from the exact principals
used for final evaluation, and PPO could anchor generation on it.

`corpus.load()` excludes these PersonIds by default, removing 840
rows. The `__main__` block runs an explicit leakage check that must
report zero overlap.

The gold set is used in exactly two places, both without gradient
flow: as frozen in-context few-shot examples for the rubric judge,
and as the final evaluation reference. It is never training data.

### Corpus after processing

| Metric | Value |
|---|---|
| Raw rows | 19,719 |
| Excluded as gold principals | 840 |
| Usable after 50-token filter and dedup | 11,953 |
| Anchor pool (scored) | 8,905 |
| Score 0 / 2 / 4 anchors | 31 / 374 / 8,500 |

Element counts in the anchor pool run from 758 (Element 7) to 1,613
(Element 3). Score 0 is severely thin at 31 anchors total, roughly
four per element. This is treated as a genuine property of the domain
rather than a sampling problem: a score of 0 means the principal did
not engage with the process at all, which is legitimately rare. The
synthetic pool preserves that rarity rather than forcing balance.

---

## Design Decisions Worth Recording

These were found during a stress test of code that had been written
but never executed. Each would have silently corrupted results.

**SFT pairing must use distinct texts.** The first implementation put
the anchor text in the prompt and used the same text as the training
target, which teaches verbatim copying. The downstream anchor leakage
filter (cosine similarity above 0.85) would then have rejected nearly
everything PPO produced. `build_sft_pairs()` now pairs each BIP with
a different BIP from the same element. Verified: 9,398 pairs, zero
self-pairs.

**Prompt tokens are masked in the SFT loss.** Only the completion
contributes to the loss. Training on the rubric text and instructions
teaches the model to reproduce its own prompt.

**Authenticity reward needs a repetition penalty.** Raw `exp(-nll)`
under a domain LM rewards repetitive boilerplate, since repeated
phrases have very low perplexity. This is the most predictable
reward-hacking failure mode in the design. The reward now combines
batch-normalized perplexity (z-score then sigmoid, so the signal
reflects relative standing within a batch rather than clustering near
a fixed point) with a distinct-bigram ratio that multiplies down
repetitive candidates.

**The rubric judge scores blind.** The first prompt told the judge the
target score and asked whether the candidate warranted it.
Instruction-tuned models agree reflexively, which would inflate the
rubric reward regardless of quality. The judge now sees all three
rubric levels, picks one independently, and the reward is computed
outside the model by comparing its prediction to the target.

**PPO batches are element-stratified.** Sampling a score then any
element drifts freely and burns through the tiny score-0 pools.
`ElementCycler` shuffles through all seven elements and reshuffles on
exhaustion.

---

## Pipeline Stages

| Stage | Script | Status |
|---|---|---|
| 1. BIPDomainSFT | `scripts/run_BIPDomainSFT.py` | Retraining |
| 2. Authenticity reward | `src/rewards/authenticity_reward.py` | Written |
| 3. Rubric reward | `src/rewards/rubric_reward.py` | Written |
| 4. Generator SFT warmup | `scripts/run_generator_sft.py` | Queued |
| 5. PPO training | `scripts/run_ppo_pilot.py` | Pilots queued |
| 6. Generation run | `src/generation/pipeline.py` | Not written |
| 7. Assessor training | `src/training/assessor.py` | Not written |
| 8. Evaluation | `src/evaluation/` | Not written |
| 9. Audit | `src/audit/` | Not written |

### Experimental conditions

Four assessor models, identical architecture and procedure, differing
only in training data:

- **A, synthetic only.** Primary condition, PPO-generated pool.
- **B, real noisy labels.** Lower-bound baseline on the natural skewed
  distribution. Expected to predict 4 for nearly everything and score
  poorly against the gold set.
- **C, balanced real subset.** Distribution control. If A beats C, the
  improvement comes from data quality, not just from rebalancing. If
  they tie, the contribution reduces to the balancing mechanism,
  which is itself a finding.
- **D, hybrid.** Synthetic plus real combined.

### Ablations

1. No authenticity reward (alpha = 0)
2. No rubric reward (alpha = 1)
3. SFT warmup only, no PPO
4. No anchor conditioning
5. Gemma-based authenticity reward instead of Qwen, testing whether
   architectural diversity between reward channels matters

---

## Running the Pipeline

### Environment

```bash
source /scratch/sakter6/bip-env/bin/activate
cd /scratch/sakter6/synthetic/Synthetic_NEE_Data
export PYTHONPATH=/scratch/sakter6/synthetic/Synthetic_NEE_Data:$PYTHONPATH
export HF_HOME=/scratch/sakter6/.cache/huggingface
export WANDB_PROJECT=synthetic-nee-bip
```

### Job chain

All stages submit as dependent SLURM jobs so the pipeline runs
unattended through long queue waits:

```bash
JOB1=$(sbatch --parsable --export=ALL,SCRIPT=scripts/run_BIPDomainSFT.py,ARGS="--data data/raw/bips.csv --model Qwen/Qwen2.5-7B" scripts/train.slurm)
JOB2=$(sbatch --parsable --export=ALL,SCRIPT=scripts/run_generator_sft.py,ARGS="--data data/raw/bips.csv" scripts/train.slurm)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 --export=ALL,SCRIPT=scripts/test_rewards.py,ARGS="" scripts/train.slurm)

for cfg in "0.5 0.1" "0.7 0.1" "0.3 0.1" "0.5 0.05"; do
  read -r alpha beta <<< "$cfg"
  sbatch --dependency=afterok:$JOB1:$JOB2 \
    --export=ALL,SCRIPT=scripts/run_ppo_pilot.py,ARGS="--alpha $alpha --beta $beta --steps 300" \
    scripts/train.slurm
done
```

Jobs 1 and 2 are independent and run in parallel. The reward test
depends only on BIPDomainSFT, so judge calibration completes before
any PPO compute is spent. The four pilots depend on both.

### Local verification without a GPU

```bash
python src/data/corpus.py data/raw/bips.csv
```

Prints both corpora and runs the leakage check. Expect 11,953 corpus
rows, 8,905 anchors, 171 gold rows, and overlap of zero.

---

## Decision Gates

**After judge calibration (`test_rewards.py`).** The number that
matters is mean signed deviation between the blind rubric judge and
expert scores. If the judge is as lenient as supervisors, the rubric
reward carries no independent signal and PPO would reinforce the same
bias the project exists to detect. Exact agreement near 0.33 means
chance performance and the judge is not reading the rubric usefully.
Either result means revising the judge prompt before spending PPO
compute.

**After PPO pilots.** Each pilot prints measured seconds per step and
extrapolates to a full 5,000-step run, replacing estimates with real
hardware numbers. If reward climbs steadily and KL stays bounded,
proceed to full PPO. If reward flatlines, spikes, or the generator
collapses to repetitive output, fall back to Ablation 3 (SFT-warmup
generator, no PPO) as the primary synthetic source and report the PPO
attempt as a documented limitation. That comparison is already framed
as scientifically meaningful in the ablation design, so it is a
legitimate result rather than a failure.

---

## Operational Notes

**Scratch purges after 90 days.** Files on `/scratch` are removed
after 90 days without access. This project lost the BIPDomainSFT
adapter weights and `data/raw/bips.csv` to a purge between sessions.
Model weights and raw data are gitignored, so git did not protect
them. Reset the clock periodically:

```bash
find /scratch/sakter6/ -type f -exec touch {} + 2>/dev/null
```

**Duplicate job submissions are dangerous, not just wasteful.** Two
jobs writing to the same output directory can interleave checkpoint
writes. Verify the queue before walking away:

```bash
squeue -u sakter6
```

**Dataset caching hides code changes.** HuggingFace `Dataset.map()`
caches to disk and will silently reuse stale tokenization after a
code edit. All dataset builders pass `load_from_cache_file=False`.

---

## Known Limitations

- Gold standard evaluation set is 171 rows across 24 principals.
  Bootstrap confidence intervals and leave-one-out evaluation
  partially compensate.
- Score 0 has 31 anchors total. Generation and evaluation at this
  level are limited in diversity and reported with that caveat.
- Reward hacking remains possible. The KL penalty, repetition
  penalty, and human spot check are the mitigations, not a proof.
- The rubric judge is few-shot, not fine-tuned. Its rubric
  interpretation may diverge from expert interpretation
  systematically. Calibration on the gold set characterizes this
  before PPO begins.
- Higher-scoring BIPs are systematically longer in the real corpus,
  so length is a confound. Generation prompts control for it, but
  residual confounding is possible.
- The audit is correlational. Structured deviations indicate
  inconsistency with rubric-aligned predictions, not causally
  established bias. The term bias is reserved for deviations that
  correlate with subgroup variables.
- PPO is stochastic. Distributional reproducibility is claimed;
  exact dataset reproducibility is not.

---

## Ethical Scope

This operates in a high-stakes domain, the performance evaluation of
school principals.

The assessor is a research instrument, not a scoring tool. It is a
proxy for rubric-consistent scoring, not an oracle. When it disagrees
with a supervisor, that establishes inconsistency with rubric-aligned
prediction, not that the supervisor was wrong. Two error sources
exist, supervisor inconsistency and model miscalibration, and the
evaluation design characterizes both.

The rurality finding from earlier analysis, a 37-point gap in score 4
rates between the most urban and most rural categories, suggests the
corpus may encode structural inequities. The audit is designed to
surface such patterns, not reproduce them. Rurality is an explicit
subgroup variable in all deviation analyses.

No individual principal is identified in any reported result. Audit
findings are reported at aggregate or subgroup level only. Results
are scoped to the NEE rubric, this corpus, and the 2015 to 2024
period.