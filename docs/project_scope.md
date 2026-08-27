# Project Scope

## Project Title
Reinforcement Learning from Rubric Feedback: Authentic Synthetic BIP
Generation for Principal Assessment Auditing


## One-Line Summary
A synthetic data generation pipeline that uses RLHF to train a
generator producing rubric-aligned, authentically-written Building
Improvement Plans (BIPs) -- optimizing simultaneously for rubric
consistency and principal writing style -- to train and evaluate an
LLM-based proxy for detecting systematic scoring deviations in the
real supervisor-assessed corpus.


---


## Motivation and Problem Statement

Evaluating school principal performance through Building Improvement
Plans (BIPs) is a high-stakes process governed by the NEE rubric.
Prior analysis of the existing corpus of supervisor-assigned BIP scores
reveals systematic skew and inter-rater bias, undermining the
reliability of those scores as ground truth. A robust automated
reference for auditing these scores requires high-quality labeled
training data -- data that does not currently exist in sufficient
quantity or quality.

The core challenge is a data scarcity problem with a compound
constraint: the 18,000+ real BIPs in the corpus carry authentic
writing but unreliable scores, while the 23 expert-adjudicated gold
standard BIPs carry reliable scores but are too few to train on.

This project addresses this tension through two key observations:

First, the real BIP corpus contains high-quality human-authored text
that reflects genuine principal writing patterns -- vocabulary,
sentence structure, domain conventions, and topical diversity -- even
though the scores assigned to this text by supervisors are unreliable.

Second, the NEE rubric provides a reliable specification of what each
score level looks like for each element, independent of any supervisor
judgment.

The pipeline exploits both observations separately and then combines
them through RLHF: a domain language model learns authentic writing
style from the real corpus without ever seeing scores, and a rubric-
conditioned generator is trained via reinforcement learning to optimize
simultaneously for rubric alignment (from the rubric specification) and
authentic style (from the domain language model), never relying on
noisy supervisor scores as a training signal.

The central research question is not only whether such a pipeline can
be built, but whether an assessor trained on its outputs can detect
systematic scoring deviations in the real supervisor-assigned corpus
that are consistent with patterns of evaluator inconsistency --
producing a finding of direct educational policy relevance.


---


## Framing: Assessor as Proxy, Not Ground Truth

A core framing commitment of this project is that the trained assessor
is a proxy for rubric-consistent scoring, not an oracle and not a
replacement for expert judgment. This distinction governs how the
audit findings are interpreted.

When the assessor disagrees with a supervisor score, that disagreement
does not establish that the supervisor is wrong. It establishes that
the supervisor's score is inconsistent with the rubric-aligned
predictions of a model trained on verified synthetic data. Two sources
of error exist: supervisor inconsistency and model miscalibration.
The project's evaluation design is explicitly structured to
characterize both.

Consequently, audit findings are reported as systematic scoring
deviations rather than bias. The term bias is reserved for deviation
patterns that correlate with identifiable subgroup variables
(supervisor identity, district, year, school type), where the subgroup
correlation itself provides additional evidence beyond model
disagreement alone.


---


## The NEE Rubric and BIP Structure

The Building Improvement Plan is a structured document completed
annually by school principals. It is organized into seven elements
across three sections:

Section A -- Role of the Principal in BIP Development:
- Element 1 (Leadership Role): describes the principal's personal
  involvement in leading BIP development
- Element 2 (Collaboration): describes the collaborative process
  used to develop the BIP with building stakeholders

Section B -- Major Objectives and Strategies:
- Element 3 (Goal Alignment): describes alignment of BIP objectives
  to district CSIP goals
- Element 4 (Baseline Data): describes measurable objectives with
  baseline data
- Element 5 (Research-Based Strategies): describes implementation
  strategies with citations to credible research sources

Section C -- Monitoring and Sharing Results:
- Element 6 (Monitoring): describes how BIP progress was monitored
  and what corrective actions were taken
- Element 7 (Sharing of Results): describes how BIP results were
  shared with staff, BIP team, and district administration

Each element is scored independently by a supervisor. The NEE rubric
defines three anchor points only: 0 (none or little), 2 (vague or
minimal), and 4 (extensive or fully described). Scores of 1 and 3 are
not defined by the rubric but appear in the corpus as supervisor
interpolations.

An important structural constraint governs evaluation: Elements 1
through 5 address the BIP development process and are not required
for evaluation of new principals who were not in their position at the
time of BIP development. Elements 6 and 7 address monitoring and
reporting and contribute to the evaluation of ALL principals. The
assessor model must be element-aware and treat this constraint
explicitly.


---


## Score Mapping Decision

The rubric defines scores 0, 2, and 4 only. Supervisors occasionally
use 1 and 3 as informal interpolations between anchors. These
intermediate scores are mapped to the nearest defined rubric level
before any training or evaluation:

- Score 1 maps to 2: a supervisor recording "minimal but not zero"
  is describing the minimal anchor
- Score 3 maps to 4: a supervisor recording "mostly good" is
  describing the exemplary anchor

This mapping is a methodological decision grounded in the rubric
definition, not a tunable parameter. It is applied consistently
across all pipeline stages. The final score vocabulary is 0, 2, 4.

Score 0 is intentionally underrepresented in the synthetic pool.
In practice, a score of 0 indicates that the principal did not engage
with the process at all -- either no BIP was submitted or the response
contained no substantive content. This is a genuinely rare condition
and the synthetic pool preserves that rarity rather than forcing
artificial balance at this level. The corpus reflects this: after
filtering, only 31 usable real BIPs carry a score of 0, compared to
406 at score 2 and 8,961 at score 4.


---


## Corpus Statistics (Post-Processing)

The following statistics reflect the corpus after score mapping,
minimum token filtering (50 tokens), and deduplication by text:

| Metric | Value |
|---|---|
| Total BIPs before filtering | 19,719 |
| Usable BIPs (BIPDomainSFT training) | 12,508 |
| Scored BIPs (anchor pool) | 9,398 |
| Score 0 anchors | 31 |
| Score 2 anchors | 406 |
| Score 4 anchors | 8,961 |
| Unique principals | 1,086 |
| Unique districts | 156 |
| School years | 2015-2016 to 2023-2024 |
| Rubric elements | 7 |
| Token count p10 / p50 / p90 / p99 | 65 / 150 / 469 / 1,297 |
| Score imbalance ratio (4 vs 0) | 220x |

Element distribution in the anchor pool is approximately even across
Elements 1-6, with Element 7 being the least represented (810 BIPs).
Higher-scoring BIPs are systematically longer: mean token count
increases from 75 at score 0 to 186 at score 4. Generation prompts
explicitly control for length to prevent the model from using length
as a proxy for score.

Key data quality findings:
- 25% of rows contain duplicate text, indicating the same BIP response
  was used across multiple elements or years. Deduplication by text
  is applied for BIPDomainSFT training. The anchor pool retains
  element-score pairs separately since the same text scored under
  different elements is informative.
- 29.7% of BIPs have no supervisor score. These are included in
  BIPDomainSFT training but excluded from the anchor pool.
- All principals have multiple BIPs (mean 18.2, min 7). Train and
  validation splits use principal-level grouping to prevent leakage.

Subgroup findings from EDA that motivate the audit analysis:
- District mean scores range from 2.18 to 4.00 across 156 districts,
  with a standard deviation of 0.40. This inter-district variance
  is a primary audit target.
- Rurality correlates with score: the most rural category (43) shows
  50% score 4, compared to 87% for the most urban category (11).
  This 37-point gap is reported as an exploratory finding pending
  audit confirmation.
- Mean scores increased sharply from 2021-2022 onward, suggesting
  possible grade inflation over time. This temporal pattern is
  included in the subgroup regression.
- The prob4_ElementX field (median 0.955) is a pre-existing predicted
  probability of score 4 from a prior scoring system. This is used
  as a baseline comparison in the audit analysis.


---


## Primary Research Contribution

The novel contribution of this project is a reinforcement learning
from rubric feedback (RLRF) pipeline for synthetic educational
assessment data generation, combined with its application to
surfacing systematic scoring deviations in a real principal assessment
corpus. Specifically:

- BIPDomainSFT (Qwen2.5-7B) is fine-tuned on 12,508 real BIP texts
  using causal language modeling, learning authentic principal writing
  style without any score supervision. It serves as a frozen reward
  model providing authenticity signal during PPO training.

- A Gemma-2-2B generator is trained via PPO to simultaneously
  optimize two reward signals: rubric alignment (from a frozen
  Gemma-2-2B few-shot rubric judge) and writing authenticity (from
  BIPDomainSFT perplexity). A KL penalty against the SFT warmup
  checkpoint prevents the generator from drifting into incoherent
  outputs.

- By directly optimizing for both objectives, the generator produces
  BIPs that are rubric-aligned and authentically written by
  construction rather than by post-hoc filtering. This is a
  fundamentally stronger claim than generate-then-filter approaches.

- Noisy supervisor scores are never used as a training signal at any
  stage. The rubric specification provides score-level conditioning;
  the real corpus provides style conditioning. These two signals
  are kept separate and combined only through the reward mechanism.

- The trained assessor is applied to the full real BIP corpus to
  surface systematic deviations between supervisor scores and
  rubric-predicted scores, with formal statistical testing to
  distinguish structured deviation from random noise.

Synthetic data generation for educational assessment via RLHF has not
been attempted in the literature. The combination of domain-specific
style reward modeling, rubric-conditioned generation, element-aware
anchor conditioning, and application to principal-level performance
auditing constitutes a novel contribution validated through four-way
model comparison, ablation studies, and a corpus-level deviation audit.


---


## Model Assignment

| Role | Model | Family | Notes |
|---|---|---|---|
| BIPDomainSFT | Qwen2.5-7B | Alibaba | Fine-tuned on 12,508 BIPs. Frozen during PPO. |
| Authenticity reward | Qwen2.5-7B (BIPDomainSFT) | Alibaba | Perplexity signal. Frozen. |
| Generator (SFT warmup) | Gemma-2-2B | Google | SFT on anchor BIPs without score conditioning. |
| Generator (PPO) | Gemma-2-2B | Google | PPO-trained to maximize combined reward. |
| Rubric alignment reward | Gemma-2-2B | Google | Frozen inference. Few-shot rubric judge. |
| Final assessor | Qwen2.5-7B | Alibaba | From BIPDomainSFT checkpoint. Classification head. |

Qwen (Alibaba) and Gemma (Google) are from architecturally distinct
families. BIPDomainSFT and the generator never share weights or
gradients. The rubric judge and generator share the same base model
family but diverge during PPO -- the judge is always frozen. This
separation satisfies the dual-model independence requirement.


---


## Pipeline Overview


### Stage 1 -- BIPDomainSFT (COMPLETE)

Qwen2.5-7B is fine-tuned on 12,508 BIP texts using a causal language
modeling objective. Scores are not used. The model learns authentic
BIP writing patterns across all seven elements. Each training example
is prefixed with its element label so the model learns element-aware
style priors.

Training used LoRA (r=16) on a single A100 80GB GPU for 3 epochs.
Final training loss: 2.198. Novel generation confirmed: all sampled
phrases from a held-out generation test returned zero matches in the
training corpus. Checkpoint saved at models/BIPDomainSFT.

This model is frozen for all subsequent stages and used only as an
inference-time authenticity reward model.


### Stage 2 -- Authenticity Reward Model

BIPDomainSFT is used as a frozen reward model providing an
authenticity signal for PPO training. Given a candidate synthetic BIP,
the reward is computed as the negative mean per-token log-likelihood
(perplexity proxy) under BIPDomainSFT, normalized to [0, 1]:

    r_auth = exp(-mean_nll) normalized across the batch

Lower perplexity under BIPDomainSFT means the candidate looks more
like real principal writing. This signal is computed efficiently by
running the frozen model in inference mode on each candidate.

No additional training is required for this stage.


### Stage 3 -- Rubric Alignment Reward Model

Gemma-2-2B is used as a frozen few-shot rubric judge. It receives
a structured prompt containing:
- The NEE rubric criteria for the target element and score level
- Three to five gold standard BIP examples at the target score level
  as in-context demonstrations
- The candidate synthetic BIP

It is prompted to predict the rubric score (0, 2, or 4) for the
candidate. The reward is:

    r_rubric = 1.0  if predicted score == target score
               0.5  if predicted score is adjacent (one level away)
               0.0  if predicted score is far off

The gold standard BIPs are used only as few-shot examples here, not
as training data. This does not constitute use of the gold standard
set for training -- it is in-context demonstration for a frozen model.
The 23 gold standard BIPs remain held out for final evaluation.

No fine-tuning is performed on Gemma-2-2B for this role.


### Stage 4 -- Generator SFT Warmup

Gemma-2-2B is fine-tuned with supervised learning on the anchor pool
before PPO training begins. This warmup step is critical: PPO training
on a cold language model is extremely unstable because the model
produces incoherent outputs, making reward signal noisy and uninformative.

The SFT warmup trains Gemma-2-2B to produce BIP-like text given a
structured prompt:

    Input:  [Element: Element3]
            [Rubric: The principal fully and clearly aligns...]
            [Anchor: <real BIP text from anchor pool>]
            Generate a BIP response for this element:
    Output: <BIP text>

Crucially, target score is NOT part of the input at this stage.
The warmup only teaches the generator to produce coherent BIP-like
text given an element and anchor. Score conditioning is introduced
in PPO training where it is reinforced by the rubric reward signal.

This avoids using noisy supervisor scores as SFT training labels.
The anchor BIP provides topical grounding; the rubric criteria
provide structural guidance; no score label is required.

Training uses LoRA on one or two A100 80GB GPUs. The SFT checkpoint
is saved and used as both the starting point for PPO training and
the reference model for KL penalty computation.


### Stage 5 -- PPO Training

The core RLHF stage. The Gemma-2-2B generator is updated using
Proximal Policy Optimization (PPO) via the TRL library to maximize
a combined reward signal from both reward models.

**Input to each PPO step:**
A prompt sampled from the anchor pool:

    [Element: Element3]
    [Target Score: 4]
    [Rubric: The principal fully and clearly aligns BIP objectives
     to CSIP goals...]
    [Anchor: <real BIP text>]
    Generate a BIP response that earns a score of 4 for this element:

**Each PPO step:**
1. Generator samples a candidate BIP from the prompt
2. Authenticity reward: r_auth from BIPDomainSFT perplexity
3. Rubric alignment reward: r_rubric from Gemma-2-2B few-shot judge
4. Combined reward:
       r = alpha * r_auth + (1 - alpha) * r_rubric
       alpha is a hyperparameter tuned during pilot runs (start 0.5)
5. KL penalty:
       r_final = r - beta * KL(generator || SFT reference)
       prevents generator from drifting from coherent BIP writing
6. PPO updates generator LoRA weights to increase probability of
   high-reward outputs

**Multi-GPU setup:**
Generator and authenticity reward model (BIPDomainSFT) run on
separate GPUs. Rubric reward model (frozen Gemma) runs on a third
GPU or shares with generator depending on memory budget. Batched
reward computation using Accelerate for efficiency.

**Checkpointing:**
PPO checkpoints saved every 500 steps. Training is resumable from
any checkpoint. Multiple pilot runs with different alpha and beta
values are submitted as separate slurm jobs before the full run.

**Convergence:**
Training runs until the combined reward plateaus across 1,000
consecutive steps or a maximum step budget is reached. The final
checkpoint is the PPO-trained generator.


### Stage 6 -- Generation Run

The PPO-trained generator produces the synthetic BIP pool. For each
generation:
1. Sample (element, target score, anchor BIP) from the anchor pool
2. Generator produces a candidate BIP
3. Both reward models score the candidate -- scores logged as
   diagnostics, not as accept/reject filters
4. Anchor leakage check: embedding cosine similarity between
   candidate and anchor must fall below threshold (0.85)
5. Accepted candidates stored with element label, target score,
   reward scores, and anchor ID

Unlike generate-then-filter pipelines, the PPO-trained generator
produces high-quality outputs by construction. The reward model
scoring in this stage is diagnostic rather than gatekeeping -- it
characterizes the quality of the generated pool rather than
filtering out failures.

Score distribution in anchor sampling:
- Score 0: sampled sparingly, reflecting genuine rarity
- Score 2 and 4: sampled to produce a training distribution
  imbalanced in the same direction as reality but less extreme


### Stage 7 -- Final Assessor Training

The synthetic BIP pool, each example paired with a verified element
label and target score label, forms the supervised training dataset
for the final assessor model. Qwen2.5-7B initialized from the
BIPDomainSFT checkpoint is fine-tuned with a classification head
to predict the NEE rubric score (0, 2, or 4) given a BIP text and
element label as input.

Starting from the BIPDomainSFT checkpoint means the assessor already
has domain writing knowledge before classification training begins.

Ordinal modeling: the ordinal nature of the scoring scale is handled
explicitly using ordinal cross-entropy loss with adjacent-score
penalty weighting or QWK-optimized training objective, selected after
pilot evaluation.

Four training conditions are run in parallel as separate slurm jobs:

Condition A -- Synthetic only (primary): assessor trained on
PPO-generated synthetic BIPs. This is the main experimental condition.

Condition B -- Real noisy labels (lower-bound baseline): assessor
trained on real BIPs with supervisor-assigned scores, preserving the
natural skewed distribution.

Condition C -- Balanced real subset (distribution control): assessor
trained on a downsampled real BIP subset matched to the synthetic
pool's score distribution. Isolates distribution balancing effects
from data quality effects.

Condition D -- Hybrid: assessor trained on synthetic and real BIPs
combined.


---


## Ablation Studies

These isolate the contribution of each pipeline component.

**Ablation 1 -- No authenticity reward**
PPO trained with rubric reward only (alpha = 0). Measures the
independent contribution of the authenticity reward signal.

**Ablation 2 -- No rubric reward**
PPO trained with authenticity reward only (alpha = 1). Measures the
independent contribution of the rubric reward signal.

**Ablation 3 -- SFT warmup only, no PPO**
Use the SFT warmup generator directly for generation without PPO
training. Measures the contribution of the RL training step.

**Ablation 4 -- No anchor conditioning**
PPO trained without real BIP anchors in the prompt (rubric and score
only). Measures the contribution of topical diversity via anchoring.

**Ablation 5 -- Single model family**
Replace BIPDomainSFT authenticity reward with a Gemma-based
perplexity model. Measures the contribution of using architecturally
distinct families for the two reward signals.

Results across all conditions and ablations are reported in a single
comparison table evaluated on the 23 gold standard BIPs.


---


## Job Submission Design

All pipeline stages are submitted as dependent slurm jobs in a single
submission script. Each job runs automatically when its predecessor
completes successfully.

```
JOB1: generator SFT warmup
JOB2: PPO pilot runs (alpha/beta grid, parallel)
JOB3: PPO full training (best hyperparameters from JOB2)
JOB4: generation run
JOB5-8: assessor training (4 conditions, parallel)
JOB9: evaluation on gold standard
JOB10: scoring deviation audit
```

Jobs 5-8 run in parallel after JOB4. JOB9 waits for all four
assessors. JOB10 runs last. The full pipeline is submitted once and
runs to completion without manual intervention.


---


## Datasets

| Dataset | Size | Score Quality | Role in Pipeline |
|---|---|---|---|
| Real BIP corpus | 19,719 raw / 12,508 usable | Noisy / biased | BIPDomainSFT training; generator SFT warmup anchors; PPO anchor pool; audit target |
| Gold standard BIPs | 23 | Expert-adjudicated | Few-shot examples for rubric judge (frozen inference only); final evaluation |
| Synthetic BIP pool | TBD per element-score cell | PPO-optimized | Stage 7 assessor training (primary condition) |


---


## Evaluation


### Primary Metric

Quadratic Weighted Kappa (QWK) between the assessor model's predicted
scores and expert-assigned scores on the 23 gold standard BIPs.


### Supporting Metrics

- Exact agreement: percentage where predicted score exactly matches
  expert score.
- Adjacent agreement: percentage where predicted score is within one
  rubric level of expert score.
- Human baseline: inter-rater agreement between NEE supervisors on
  the same 23 BIPs.
- Bootstrap confidence intervals (1,000 resamples) on QWK.
- Leave-one-out evaluation alongside bootstrap CIs.


### Assessor Calibration Analysis

- Directional bias check: mean signed deviation with confidence
  interval.
- Score-level calibration: per-level agreement rates.
- Prediction confidence: entropy distribution across predictions,
  used to define high-confidence subset for audit.


### Difficulty Calibration

- Model confidence on synthetic held-out BIPs vs real BIPs.
- Performance gap between held-out synthetic test set and gold
  standard BIPs.


### PPO Training Diagnostics

- Combined reward trajectory across PPO steps.
- Authenticity reward and rubric reward reported separately.
- KL divergence from SFT reference across training.
- Reward model agreement rate: how often both rewards agree on
  high vs low quality outputs.


### Synthetic Data Distribution Characterization

- Length distribution (token count, sentence count).
- Vocabulary diversity (type-token ratio, OOV rate).
- Structural variation (paragraph count, section presence).
- Semantic diversity (pairwise cosine distance in embedding space).
- N-gram diversity (distinct unigram and bigram ratios).


---


## Scoring Deviation Audit: Applying the Assessor to Real BIPs

Once the assessor is trained, validated, and characterized, it is
applied to the full real BIP corpus. For each BIP, the model produces
a rubric-predicted score and a confidence estimate.

**Definition of a meaningful finding:** Observed deviations must be
(a) statistically non-random at corpus level, (b) structured rather
than uniformly distributed across score levels, and (c) not fully
explained by known assessor miscalibration.

**Statistical tests:**
- Mean deviation test: bootstrap test of whether
  mean(supervisor score - model score) is not zero.
- Distributional shift: Kolmogorov-Smirnov test between supervisor
  and model-predicted score distributions.
- Subgroup regression:
    deviation ~ supervisor_id + district + year + rurality

**Pre-specified audit targets from EDA:**
- District variance: 62 districts viable for subgroup analysis,
  mean scores ranging 2.18 to 4.00.
- Rurality: 37-point gap in score 4 rates between urban and rural.
- Temporal trend: score inflation from 2021-2022 onward.
- prob4_ElementX as additional baseline.

**Pre-specified interpretation scenarios:**
- No significant deviation: consistent scoring.
- Significant, unstructured: low reliability.
- Significant, directional, not subgroup-correlated: systematic
  scoring inconsistency.
- Significant, subgroup-correlated: potential evaluator bias.


---


## Validation Plan


### PPO Pilot Runs

Before the full PPO training run, a grid of pilot runs varying alpha
(authenticity weight) and beta (KL coefficient) is submitted as
parallel slurm jobs. Each pilot runs for 1,000 steps. The combination
producing the highest combined reward with stable KL divergence is
selected for the full run. Prompt templates are locked after pilots.


### Reward Model Calibration

Both reward models are evaluated on the 23 gold standard BIPs before
PPO training begins. BIPDomainSFT perplexity scores are computed for
each gold standard BIP and compared against score level -- lower
perplexity should correlate with higher scores if the model has
internalized quality patterns. The Gemma rubric judge is tested on
the same BIPs and agreement with expert scores is reported. Any
systematic miscalibration is investigated before PPO training.


### Human Spot-Check

A sample of approximately 100 generated BIPs from the PPO-trained
generator (balanced across elements and score levels) is reviewed by
a human evaluator. The evaluator assesses: (1) does this sound like
a principal wrote it, (2) does this BIP warrant the target score
according to the rubric, (3) is the improvement plan pedagogically
plausible. Results are reported as qualitative validation alongside
quantitative reward scores.


### Reproducibility Stance

Exact dataset reproducibility is not claimed: PPO training has
stochastic elements and generation involves sampling. What is
reproducible is pipeline behavior: given the locked prompts, fixed
random seeds, and documented hyperparameters, the distributional
properties of the synthetic pool are expected to be stable. All
PPO checkpoints, generation logs, and reward scores are archived.


### Final Evaluation Isolation

The 23 gold standard BIPs are used only as frozen few-shot examples
for the rubric judge (inference only, no gradient flow) and for final
assessor evaluation. They are never used as training data at any stage.


---


## Compute Environment

All training and inference runs on the GMU Hopper HPC cluster.
BIPDomainSFT trained on single A100 80GB. Generator SFT warmup and
PPO training use multiple A100 80GB GPUs via Accelerate. All four
assessor conditions run in parallel as separate slurm jobs. The
project environment is a Python venv at /scratch/sakter6/bip-env.
All code, logs, and outputs are stored under
/scratch/sakter6/synthetic/Synthetic_NEE_Data.

Key libraries: HuggingFace Transformers, PEFT, TRL (PPO), Accelerate,
Datasets, WandB for experiment tracking.


---


## Explicit Non-Goals

- The pipeline does not replace human principal assessment.
- The assessor is not deployed as a scoring tool. It is a research
  instrument; predictions are a proxy, not ground truth.
- The project does not claim synthetic BIPs are indistinguishable from
  real BIPs -- only that they optimize for both rubric alignment and
  authentic style simultaneously.
- The audit does not adjudicate which score is correct.
- The assessor is not evaluated beyond the NEE rubric domain.
- Pedagogical validity of generated BIPs is not claimed beyond human
  spot-check validation.


---


## Known Limitations

- The gold standard evaluation set of 23 BIPs is small. Bootstrap CIs
  and leave-one-out evaluation partially compensate.
- PPO training instability: reward hacking is possible if the
  generator finds outputs that game one reward model without genuinely
  improving. KL penalty and human spot-check are the primary
  mitigations.
- The Gemma rubric judge is not fine-tuned -- it relies on few-shot
  prompting. Its rubric interpretation may diverge from expert
  interpretation in systematic ways. Reward model calibration on gold
  standard BIPs characterizes this risk before training.
- Score 0 is severely underrepresented (31 anchors). Generation and
  training at this level are limited in diversity.
- Length confounding: higher-scoring BIPs are systematically longer.
  Generation prompts explicitly control for length.
- The audit is correlational. Structured deviations indicate
  inconsistency with rubric-aligned predictions, not causally
  established bias.
- Subgroup analysis is contingent on metadata completeness.
- PPO hyperparameter sensitivity: alpha and beta require pilot tuning.
  Results may vary across hyperparameter settings.


---


## Ethical Considerations

- Misuse risk: the assessor could be repurposed as an automated scorer.
  This project does not advocate for automated scoring of principals.
- Fairness: the rurality finding suggests the corpus may encode
  structural inequities. The audit is designed to surface, not
  reproduce, such patterns.
- Transparency: all PPO training logs, reward scores, and generation
  logs are archived and traceable.
- Principal privacy: no individual principal is identified in any
  reported result. Audit findings are reported at aggregate or
  subgroup level only.
- Scope of claims: results are scoped to the NEE rubric, the specific
  corpus, and the 2015-2024 time period.


---


## Project Outputs

### Artifacts
- BIPDomainSFT checkpoint (Qwen2.5-7B, COMPLETE).
- PPO-trained generator checkpoint (Gemma-2-2B).
- Synthetic BIP dataset with element labels, target scores, and
  per-candidate reward scores.
- Trained assessor checkpoints for all four training conditions.
- Full pipeline code, prompt templates, PPO training logs, and
  evaluation scripts -- publicly released on GitHub.
- Distribution characterization and difficulty calibration reports.

### Written Output
- A research paper describing the RLHF pipeline for synthetic
  educational assessment data generation, four-way model comparison,
  ablation results, calibration analysis, and scoring deviation audit
  findings, targeting an NLP or educational NLP venue (BEA workshop,
  EMNLP, NAACL, or similar).
- Detailed pipeline documentation for reproducibility including all
  prompt templates, PPO hyperparameters, reward model calibration
  logs, and human spot-check protocol.