# Project Scope

## Project Title (Loosely)
Rubric-Aligned Synthetic BIP Generation with Dual-Model Verification
for Principal Assessment Auditing


## One-Line Summary
A synthetic data generation pipeline that produces rubric-aligned,
authenticity-verified Building Improvement Plans (BIPs) to train and
evaluate an LLM-based proxy for rubric-consistent scoring -- validated
through four-way model comparison, ablation studies, and applied to
detecting systematic scoring deviations in the real supervisor-assessed
corpus.


---


## Motivation and Problem Statement

Evaluating school principal performance through Building Improvement
Plans (BIPs) is a high-stakes process governed by the NEE rubric. Prior
analysis of the existing corpus of supervisor-assigned BIP scores reveals
systematic skew and inter-rater bias, undermining the reliability of
those scores as ground truth. A robust automated reference for auditing
these scores requires high-quality labeled training data -- data that does
not currently exist in sufficient quantity or quality.

The core challenge is a data scarcity problem with a compound constraint:
the 18,000+ real BIPs in the corpus carry authentic writing but unreliable
scores, while the 23 expert-adjudicated gold standard BIPs carry reliable
scores but are too few to train on. This project resolves that tension by
generating a large pool of synthetic BIPs that are simultaneously
rubric-aligned and authentically written, using a two-model pipeline with
independent verification.

The central research question is not only whether such a pipeline can be
built, but whether an assessor trained on its outputs can detect systematic
scoring deviations in the real supervisor-assigned corpus that are
consistent with patterns of evaluator inconsistency -- producing a finding
of direct educational policy relevance.


---


## Framing: Assessor as Proxy, Not Ground Truth

A core framing commitment of this project is that the trained assessor is
a proxy for rubric-consistent scoring, not an oracle and not a replacement
for expert judgment. This distinction governs how the audit findings are
interpreted.

When the assessor disagrees with a supervisor score, that disagreement
does not establish that the supervisor is wrong. It establishes that the
supervisor's score is inconsistent with the rubric-aligned predictions of
a model trained on verified synthetic data. Two sources of error exist:
supervisor inconsistency and model miscalibration. The project's evaluation
design is explicitly structured to characterize both.

Consequently, audit findings are reported as systematic scoring deviations
rather than bias. The term bias is reserved for deviation patterns that
correlate with identifiable subgroup variables (supervisor identity,
district, year, school type), where the subgroup correlation itself
provides additional evidence beyond model disagreement alone.


---


## The NEE Rubric and BIP Structure

The Building Improvement Plan is a structured document completed annually
by school principals. It is organized into seven elements across three
sections:

Section A -- Role of the Principal in BIP Development:
- Element 1 (Leadership Role): describes the principal's personal
  involvement in leading BIP development
- Element 2 (Collaboration): describes the collaborative process used
  to develop the BIP with building stakeholders

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

The novel contribution of this project is the dual-model synthetic data
generation architecture with independent dual-channel verification,
combined with its application to surfacing systematic scoring deviations
in a real principal assessment corpus. Specifically:

- A domain-fine-tuned small language model (BIPDomainSFT) is
  repurposed as an authenticity judge, leveraging its learned prior
  over 12,508 real BIP documents to detect out-of-distribution
  synthetic text -- a role distinct from its training objective.

- A rubric-conditioned generation model produces score-targeted BIPs
  anchored to real BIP topics, separating content diversity (sourced
  from the real corpus) from score-relevant quality (conditioned on
  the rubric). Generation is element-aware: each synthetic BIP
  targets a specific rubric element and score level.

- The two verification channels -- authenticity and rubric alignment --
  are assigned to architecturally distinct model families with different
  information access, avoiding self-referential judgment and producing
  independently auditable accept/reject decisions.

- The trained assessor is applied to the full real BIP corpus to
  surface systematic deviations between supervisor scores and
  rubric-predicted scores, with formal statistical testing to distinguish
  structured deviation from random noise.

Synthetic data generation for educational assessment has been explored
in prior work, but the combination of element-aware topic anchoring
from a real corpus, dual-model independent verification across
architecturally diverse models, difficulty calibration analysis, and
application to principal-level performance rubric auditing has not been
attempted in the literature. The contribution is empirically validated
through four-way model comparison and ablation over pipeline components.


---


## Model Assignment

| Module | Model | Role |
|---|---|---|
| Module 1 | Qwen2.5-7B | BIPDomainSFT -- fine-tuned on 12,508 BIPs via causal LM |
| Module 2 | Gemini 2.0 Flash | Rubric-conditioned generator -- prompted with rubric + anchor |
| Module 3 Check 1 | Qwen2.5-7B (Module 1 checkpoint) | Authenticity judge |
| Module 3 Check 2 | Gemini 2.0 Flash | Rubric alignment judge |
| Module 4 | Qwen2.5-7B (from Module 1 checkpoint) | Final assessor -- classification fine-tune |

Qwen and Gemini are from architecturally distinct families (Alibaba vs
Google). This separation is deliberate and satisfies the dual-model
independence requirement for the verification pipeline.


---


## Pipeline Overview


### Module 1 -- BIPDomainSFT

Qwen2.5-7B is fine-tuned on 12,508 BIP texts using a causal language
modeling objective. Scores are not used at this stage. The model learns
authentic BIP writing patterns: principal voice, improvement plan
vocabulary, structural conventions, and typical document length per
element.

Each training example is prefixed with its element label:
"Element3: Our building improvement objectives are aligned..."
This element-conditioning costs nothing extra and gives the model
element-aware style priors that improve its reliability as an
authenticity judge in Module 3.

Training data is deduplicated by text. Principal-level grouping is used
for any validation split to prevent leakage across the same principal's
responses.

BIPDomainSFT is drawn from a different model family than the generator
used in Module 2. This architectural diversity is deliberate: it reduces
the risk that both models share the same systematic biases or surface
fluency heuristics, which would undermine the independence of the dual
verification step.

This model serves two purposes: it is the primary authenticity judge
in Module 3, and an alternative generator condition in ablation studies.

Fine-tuning is performed with LoRA/PEFT on a single A100 80GB GPU on
the Hopper HPC cluster using HuggingFace Transformers and PEFT.


### Module 2 -- Rubric-Conditioned Generator

Gemini 2.0 Flash receives a structured prompt comprising: (1) the full
NEE rubric criteria for the target element and score level, (2) a
target score (0, 2, or 4), (3) the target element label, and (4) a
single real BIP sampled from the anchor pool as a topical anchor. The
model is instructed to generate a new BIP response for that element
that addresses similar improvement themes as the anchor but demonstrates
competencies at the specified score level.

One real BIP is used as anchor per generation, providing natural
topical diversity across the synthetic pool without requiring manually
curated seeds. Anchors are sampled at the element level to ensure each
generated BIP is conditioned on an element-appropriate real response.

Score distribution in anchor sampling is intentionally non-uniform:
score 0 anchors are used sparingly to reflect the genuine rarity of
non-engagement in the real population. Score 2 and score 4 anchors
are sampled to produce a training distribution that is imbalanced
in the same direction as reality, but less extreme than the raw
corpus skew.

**Anchor leakage control:** Each generated BIP is compared against
its anchor using embedding cosine similarity. Candidates exceeding
a similarity threshold are rejected and regenerated with a fresh
anchor. The distribution of anchor-to-synthetic similarity scores
is reported as a pipeline diagnostic.

**Operational parameters (confirmed after pilot):**
- Target pool size: scaled per element and score level after pilot
- Generation temperature: fixed across all runs for reproducibility
- Stopping criterion: fixed total accepted per element-score cell
- Prompt variants: tested in pilot; final prompt locked before full run
- Generation effort: attempts per accepted sample reported by
  element-score cell to detect diversity degradation


### Module 3 -- LLM-as-Judge (Dual Verification)

Every candidate synthetic BIP is evaluated on two independent axes
before entering the training pool.

**Check 1: Authenticity**
BIPDomainSFT receives the candidate BIP alongside several real BIP
examples for the same element as in-context references. It is prompted
to assess whether the candidate is consistent with the writing style,
structure, and language patterns of authentic principal-authored BIPs
for that element. The model's domain-specific prior -- trained on
12,508 real examples -- provides a stronger authenticity signal than
a general-purpose model would. Because BIPDomainSFT and the generator
are from different model families, the authenticity check is not
trivially passed by fluent outputs of the generator.

**Check 2: Rubric Alignment**
Gemini 2.0 Flash receives the NEE rubric criteria for the target
element, the intended target score, and the candidate BIP. It is
prompted to independently score the BIP according to the rubric and
assess whether its score agrees with the intended label. Candidates
where the model-assigned score disagrees with the intended score by
more than one rubric level are rejected.

**Verdict and Loop**
Both checks must pass for a candidate to enter the training pool.
Rejections are logged with a reason code: authenticity failure, rubric
misalignment, anchor leakage, or both. On rejection, a new anchor is
sampled from the real corpus and the candidate is regenerated.
Rejection rate and reason distribution are recorded as pipeline
diagnostics.

**Judge Reliability**
Judge reliability is empirically evaluated, not assumed:
- Inter-judge agreement: rate at which both judges agree on
  accept/reject for the same candidate.
- Human vs judge agreement: a sample of approximately 100 candidates
  (balanced across accept/reject and score levels) is reviewed by a
  human evaluator blind to judge verdicts. Human verdicts are compared
  against each judge independently, producing human-judge agreement
  rates alongside false positive and false negative estimates per judge.
- This spot-check is framed explicitly as qualitative validation and
  directional estimation, not precise error rate quantification.
Results are reported as judge reliability metrics alongside pipeline
diagnostics.


### Module 4 -- Final Assessor Training

The accepted synthetic BIPs, each paired with a verified element label
and score label, form the supervised training dataset for the final
assessor model. This is a text classification fine-tuning task:
Qwen2.5-7B initialized from the BIPDomainSFT checkpoint is trained
with a classification head to predict the NEE rubric score (0, 2,
or 4) given a BIP text and element label as input. Starting from the
BIPDomainSFT checkpoint means the assessor already has domain writing
knowledge baked in before classification training begins.

**Ordinal modeling:** The ordinal nature of the scoring scale is
handled explicitly using one of the following (selected after pilot):
ordinal cross-entropy loss with adjacent-score penalty weighting, or
QWK-optimized training objective. The specific choice is justified in
the paper against the score distribution of the training data.
Reporting uses QWK as the primary metric throughout.

The model is trained exclusively on synthetic data in the primary
condition; real BIP scores are never used as training labels in this
condition.


---


## Datasets

| Dataset | Size | Score Quality | Role in Pipeline |
|---|---|---|---|
| Real BIP corpus | 19,719 raw / 12,508 usable | Noisy / biased | Module 1 fine-tuning; Module 2 anchor pool; audit target |
| Synthetic BIP pool | TBD per element-score cell | Verified by judges | Module 4 training (primary condition) |
| Gold standard BIPs | 23 | Expert-adjudicated | Final evaluation only -- held out from all pipeline stages |


---


## Experimental Conditions and Ablations


### Four-Way Model Comparison

The central empirical question is whether training on synthetic data
generalizes to real BIPs, and whether any observed improvement is
attributable to data quality or simply to score distribution balancing.
Four assessor models are trained under identical architecture and
training procedure, differing only in training data composition, and
all evaluated on the 23 gold standard BIPs.

**Condition A -- Synthetic only (primary)**
Assessor trained on synthetic BIPs verified by both judges. This is
the main experimental condition.

**Condition B -- Real noisy labels (lower-bound baseline)**
Assessor trained on real BIPs with supervisor-assigned scores,
preserving the natural skewed distribution. Establishes the floor
that motivates the project.

**Condition C -- Balanced real subset (distribution control)**
Assessor trained on a downsampled or reweighted subset of real BIPs
matched to the synthetic pool's score distribution. This condition
isolates the effect of distribution balancing from the effect of
synthetic data quality. If Condition A outperforms Condition C, the
improvement is attributable to verified synthetic data quality, not
merely to balancing. If they perform equivalently, the contribution
reduces to the balancing mechanism, which is itself a finding.

**Condition D -- Hybrid**
Assessor trained on synthetic and real BIPs combined. Tests whether
authentic writing signal from real data complements verified labels
from synthetic data.

Differences in evaluation performance across all four conditions are
attributable solely to training data composition.


### Ablation Studies

These isolate the contribution of each pipeline component and directly
justify the dual-model architecture claim.

**Ablation 1 -- No authenticity filter**
Run generation and rubric alignment check only. Accept all
rubric-passing BIPs regardless of authenticity verdict. Train assessor
and evaluate. Measures the independent contribution of the
authenticity check.

**Ablation 2 -- No rubric alignment filter**
Run generation and authenticity check only. Accept all
authenticity-passing BIPs regardless of rubric alignment verdict.
Train assessor and evaluate. Measures the independent contribution
of the rubric check.

**Ablation 3 -- Single-model self-judge**
Use the same model for both generation and rubric alignment judging.
Train assessor and evaluate. Measures the contribution of
architectural diversity between generator and judge versus a
self-judging setup.

**Ablation 4 -- No anchor conditioning**
Generate synthetic BIPs without anchoring to real BIPs (rubric and
score only, no topical anchor). Train assessor and evaluate. Measures
the contribution of topical diversity via real corpus anchoring.

Results across all conditions and ablations are reported in a single
comparison table.


---


## Evaluation


### Primary Metric

Quadratic Weighted Kappa (QWK) between the assessor model's predicted
scores and expert-assigned scores on the 23 gold standard BIPs. QWK is
the standard metric for automated rubric scoring tasks and penalizes
ordinal distance between predictions and true labels.


### Supporting Metrics

- Exact agreement: percentage of BIPs where predicted score exactly
  matches expert score.
- Adjacent agreement: percentage of BIPs where predicted score is
  within one rubric level of expert score.
- Human baseline: inter-rater agreement between NEE supervisors on
  the same 23 BIPs, providing a human-performance reference point.
- Bootstrap confidence intervals (1,000 resamples) on QWK to account
  for the small evaluation set size.
- Leave-one-out evaluation reported alongside bootstrap CIs to extract
  maximum signal from 23 examples.


### Assessor Calibration Analysis

Beyond aggregate QWK, the assessor's calibration across the scoring
scale is explicitly characterized on the 23 gold standard BIPs:
- Directional bias check: does the model systematically over-predict
  or under-predict relative to expert scores at the corpus level?
  Reported as mean signed deviation with confidence interval.
- Score-level calibration: per-level agreement rates to detect whether
  the model is well-calibrated at some rubric levels but not others.
- Prediction confidence: entropy or softmax confidence distribution
  across predictions, used to define a high-confidence subset for
  the audit analysis.

These calibration results directly inform the interpretation of the
audit: deviations are reported separately for high-confidence and
low-confidence predictions, with the high-confidence subset
constituting the primary evidence.


### Difficulty Calibration

Synthetic data may be easier to classify than real BIPs even if
surface distributions match. This is tested directly:
- Model confidence (prediction entropy) on synthetic held-out BIPs
  vs real BIPs from the corpus.
- Performance gap between held-out synthetic test set and the 23 gold
  standard BIPs, where a large gap indicates difficulty mismatch.
Results are reported and interpreted in context of the overall
findings.


### Error Analysis

Disagreements on the gold standard set are analyzed structurally:
- Breakdown by score level: does the model systematically over- or
  under-predict at specific rubric levels?
- Breakdown by element: does performance differ across the seven
  rubric elements?
- Breakdown by BIP length and vocabulary complexity.
- Qualitative inspection of the highest-confidence errors: cases
  where the predicted score is far from the expert label.
- At least five example-based analyses reported in the paper.


### Pipeline Diagnostics

- Rejection rate per element-score cell (Module 3 output).
- Rejection reason distribution (authenticity / rubric / anchor
  leakage / both).
- Score distribution of accepted synthetic pool vs real corpus.
- Anchor-to-synthetic cosine similarity distribution.
- Generation attempts per accepted sample by element-score cell.
- Inter-judge agreement rate.
- Human vs judge agreement rate (spot-check sample).
- Judge false positive and false negative estimates.


### Synthetic Data Distribution Characterization

Accepted synthetic BIPs are characterized against the real corpus on:
- Length distribution (token count, sentence count).
- Vocabulary diversity (type-token ratio, out-of-vocabulary rate
  relative to real corpus vocabulary).
- Structural variation (paragraph count, section presence).
- Semantic diversity (pairwise cosine distance in embedding space
  across synthetic pool vs within real corpus).
- N-gram diversity (distinct unigram and bigram ratios).

These metrics are reported as a distribution characterization table
and used to flag systematic over-uniformity in the synthetic pool.


---


## Scoring Deviation Audit: Applying the Assessor to Real BIPs

Once the assessor is trained, validated on the gold standard set, and
characterized for calibration and difficulty, it is applied to the
full real BIP corpus. For each BIP, the model produces a
rubric-predicted score and a confidence estimate. The audit is
conducted on the full corpus and separately on the high-confidence
subset.

**Definition of a meaningful finding:** A meaningful audit finding
requires that observed deviations between supervisor scores and
model-predicted scores are (a) statistically non-random at the corpus
level, (b) structured rather than uniformly distributed across score
levels, and (c) not fully explained by known assessor miscalibration
as characterized on the gold standard set.

**Statistical tests applied:**
- Mean deviation test: one-sample t-test or bootstrap test of whether
  mean(supervisor score - model score) is not zero at the corpus level.
- Distributional shift: Kolmogorov-Smirnov test between supervisor
  score distribution and model-predicted score distribution.
- Subgroup analysis: where metadata is available, a regression of
  deviation on supervisor ID, district, year, and rurality:
    deviation ~ supervisor_id + district + year + rurality
  Significant coefficients indicate structured, subgroup-correlated
  deviation -- the condition under which the term "bias" is used.

The following subgroup patterns are pre-specified as audit targets
based on EDA findings:
- District-level variance: 62 districts have at least 30 scored BIPs
  and are viable for subgroup analysis. District mean scores range
  from 2.18 to 4.00, making district the primary subgroup variable.
- Rurality: a 37-point gap in score 4 rates between most urban (87%)
  and most rural (50%) categories is a pre-specified hypothesis.
- Temporal trend: mean scores increased sharply from 2021-2022
  onward; year is included as a regression variable.
- The prob4_ElementX field (a prior model's predicted probability of
  score 4) is used as an additional baseline in deviation analysis.

**Interpretation scenarios are specified in advance:**
- No significant deviation: supervisors are scoring consistently with
  rubric-aligned predictions.
- Significant deviation, unstructured: high variance but no
  directional pattern; suggests low reliability rather than
  systematic inflation or deflation.
- Significant deviation, directional but not subgroup-correlated:
  corpus-level inflation or deflation; reported as systematic scoring
  inconsistency.
- Significant deviation, subgroup-correlated: scoring patterns differ
  across supervisors, districts, rurality, or time in ways that cannot
  be explained by rubric variation alone; reported as potential
  evaluator bias with specific subgroup effects quantified.

The audit is framed throughout as detecting scoring deviations
relative to a rubric-consistent proxy, not as establishing ground
truth. All findings are reported with appropriate uncertainty and
require human interpretation for policy conclusions.


---


## Validation Plan


### Pre-Run Pilot

Before running the full synthetic generation pipeline, a batch of
30-50 synthetic BIPs per element-score cell is generated and reviewed
by at least one domain expert (education policy expert or
NEE-familiar reviewer). The expert assesses: (1) whether BIPs are
plausibly principal-authored, (2) whether score assignments are
consistent with the rubric for the specific element, and (3) whether
the improvement plans described are pedagogically plausible -- not
merely well-written or rubric-aligned. Pilot results are reported
alongside quantitative pipeline metrics. Prompt templates are locked
after the pilot; no further modification occurs during the full run.


### Judge Calibration

Both judges are evaluated on the 23 gold standard BIPs before the
full generation run. Verdicts on known-good examples are recorded to
confirm calibration. A judge that consistently rejects gold standard
BIPs indicates miscalibration and requires investigation before
deployment.


### Human Spot-Check

A sample of approximately 100 candidates (balanced across accept/reject
verdicts and score levels) from the full run is reviewed by a human
evaluator blind to judge verdicts. Human verdicts are compared against
each judge independently, producing human-judge agreement rates
alongside false positive and false negative estimates. This is
explicitly framed as qualitative validation and directional estimation
rather than precise error rate quantification.


### Reproducibility Stance

Exact dataset reproducibility is not claimed: generation model
non-determinism means the precise synthetic pool cannot be guaranteed
across hardware or API versions even at low temperature. What is
reproducible is pipeline behavior: given the locked prompts, fixed
temperature, and documented acceptance criteria, the distributional
properties of the synthetic pool are expected to be stable. All
generation logs, judge verdicts, and accepted BIPs are archived to
support this weaker but honest reproducibility claim.


### Final Evaluation Isolation

The 23 gold standard BIPs are withheld from all pipeline stages
including pilot review, fine-tuning, and judge calibration. They are
used only for final assessor evaluation.


---


## Compute Environment

All training and inference runs on the GMU Hopper HPC cluster.
Fine-tuning uses a single NVIDIA A100 80GB GPU via SLURM batch
submission. Interactive testing uses the normal CPU partition for
data and preprocessing code, and MIG GPU slices (1g.10gb) for
smoke tests. The project environment is a Python venv located at
/scratch/sakter6/bip-env. All code, logs, and outputs are stored
under /scratch/sakter6/synthetic/Synthetic_NEE_Data. Training jobs
are submitted via sbatch using a shared slurm script with the target
Python script passed as an environment variable.


---


## Explicit Non-Goals

- The pipeline does not replace human principal assessment. It surfaces
  scoring deviations relative to rubric-consistent predictions.
- The assessor model is not deployed as a scoring tool. It is a
  research instrument; its predictions are a proxy, not ground truth.
- The project does not claim that synthetic BIPs are indistinguishable
  from real BIPs -- only that they pass explicit authenticity and
  alignment criteria sufficient for training a rubric-consistent proxy.
- The audit does not adjudicate which score is correct. It surfaces
  structured deviations; human interpretation and institutional review
  are required for any policy conclusions.
- The final assessor is not evaluated on its ability to generalize
  beyond the NEE rubric domain or beyond the BIP document type.
- Pedagogical validity of generated BIPs is not claimed beyond expert
  plausibility checks in the pilot.


---


## Known Limitations

- The gold standard evaluation set of 23 BIPs is small. Bootstrap CIs
  and leave-one-out evaluation partially compensate, but assessor
  calibration claims should be interpreted with this constraint
  explicit.
- Circularity risk: the generator and rubric judge reason about the
  same rubric. Using architecturally distinct model families reduces
  but does not eliminate the risk that both share systematic rubric
  interpretation biases. Human spot-checks and expert pilot review
  are the primary mitigations.
- Score 0 is severely underrepresented (31 anchors). Synthetic
  generation at this level is limited in diversity and all score-0
  findings are reported with this constraint explicit.
- Authenticity as measured by BIPDomainSFT reflects writing style
  consistency, not factual accuracy or pedagogical soundness. A BIP
  can be authentic in style and rubric-aligned but describe
  improvement plans that are unrealistic or educationally unsound.
  Expert pilot review partially addresses this.
- The 25% duplicate text rate in the raw corpus indicates widespread
  reuse of BIP text across elements and years. Deduplication is
  applied for BIPDomainSFT training but the underlying reuse pattern
  may affect the diversity of the anchor pool.
- Length confounding: higher-scoring BIPs are systematically longer.
  Generation prompts explicitly control for length to mitigate this,
  but residual confounding is possible and is noted in the paper.
- Judge reliability is estimated via human spot-check on approximately
  100 candidates. This supports directional conclusions but is not
  sufficient for precise error rate quantification.
- The audit is correlational. Structured deviations indicate
  inconsistency with rubric-aligned predictions, not causally
  established bias.
- Subgroup analysis is contingent on metadata completeness. District
  and rurality metadata are available; supervisor-level ID metadata
  availability is to be confirmed.
- Prompt non-determinism: distributional reproducibility is claimed;
  exact dataset reproducibility is not.


---


## Ethical Considerations

This project operates in a high-stakes domain -- the performance
evaluation of school principals. Several ethical considerations apply:

- Misuse risk: the trained assessor could in principle be repurposed
  as an automated scorer rather than an auditor. This project
  explicitly does not advocate for automated scoring of principals.
  The model is a research instrument; deployment decisions require
  human oversight and institutional review.
- Fairness: the rurality finding (37-point gap in score 4 rates
  between urban and rural categories) suggests the corpus may encode
  structural inequities. The audit is designed to surface, not
  reproduce, such patterns. Rurality is included as an explicit
  subgroup variable in all deviation analyses.
- Transparency: all model decisions in the pipeline are logged with
  evidence. Every accepted synthetic BIP and every assessor prediction
  is traceable to its inputs and the criteria applied.
- Principal privacy: the real BIPs are used for model training only.
  No individual principal is identified in any reported result. Audit
  findings are reported at the aggregate or subgroup level, not the
  individual level.
- Scope of claims: results are scoped to the NEE rubric, the specific
  corpus, and the 2015-2024 time period of the data.


---


## Project Outputs

### Artifacts
- BIPDomainSFT checkpoint (Qwen2.5-7B fine-tuned on 12,508 BIPs).
- Synthetic BIP dataset with verified score and element labels,
  accept/reject logs, anchor similarity scores, and per-candidate
  judge verdicts.
- Trained assessor model checkpoints for all four training conditions.
- Full pipeline code, prompt templates, generation logs, and evaluation
  scripts -- publicly released on GitHub.
- Distribution characterization and difficulty calibration reports.

### Written Output
- A research paper describing the pipeline architecture, four-way
  model comparison, ablation results, calibration analysis, and
  scoring deviation audit findings, targeting an NLP or educational
  NLP venue (BEA workshop, EMNLP, NAACL, or similar).
- Detailed pipeline documentation for reproducibility, including all
  prompt templates, judge calibration logs, and human spot-check
  protocol.