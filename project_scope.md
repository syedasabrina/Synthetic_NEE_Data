# Project Scope

## Project Title (Loosely)
Rubric-Aligned Synthetic BIP Generation with Dual-Model Verification
for Principal Assessment Auditing


## One-Line Summary
A synthetic data generation pipeline that produces rubric-aligned,
authenticity-verified Building Improvement Plans (BIPs) to train and
evaluate an LLM-based auditor of principal assessment scores assigned
by NEE supervisors — validated through three-way model comparison,
ablation studies, and applied to systematic bias detection in the
real supervisor-scored corpus.


---


## Motivation and Problem Statement

Evaluating school principal performance through Building Improvement
Plans (BIPs) is a high-stakes process governed by the NEE rubric. Prior
analysis of the existing corpus of supervisor-assigned BIP scores reveals
systematic skew and inter-rater bias, undermining the reliability of
those scores as ground truth. A robust automated auditor of these scores
requires high-quality labeled training data — data that does not
currently exist in sufficient quantity or quality.

The core challenge is a data scarcity problem with a compound constraint:
the 18,000+ real BIPs in the corpus carry authentic writing but unreliable
scores, while the 23 expert-adjudicated gold standard BIPs carry reliable
scores but are too few to train on. This project resolves that tension by
generating a large pool of synthetic BIPs that are simultaneously
rubric-aligned and authentically written, using a two-model pipeline with
independent verification.

The central research question is not only whether such a pipeline can be
built, but whether an assessor trained on its outputs can detect systematic
patterns of score inflation, deflation, or subgroup bias in the real
supervisor-assigned corpus — producing a finding of direct educational
policy relevance.


---


## Primary Research Contribution

The novel contribution of this project is the dual-model synthetic data
generation architecture with independent dual-channel verification.
Specifically:

- A domain-fine-tuned small language model is repurposed as an
  authenticity judge, leveraging its learned prior over 18k real BIP
  documents to detect out-of-distribution synthetic text — a role
  distinct from its training objective.

- A rubric-conditioned generation model produces score-targeted BIPs
  anchored to real BIP topics, separating content diversity (sourced
  from the real corpus) from score-relevant quality (conditioned on
  the rubric).

- The two verification channels — authenticity and rubric alignment —
  are assigned to architecturally distinct model families with different
  information access, avoiding self-referential judgment and producing
  independently auditable accept/reject decisions.

- The trained assessor is applied to the full 18k real BIP corpus to
  surface systematic deviations between supervisor scores and
  rubric-predicted scores, constituting a bias audit with direct policy
  implications.

Synthetic data generation for educational assessment has been explored
in prior work, but the combination of topic anchoring from a real corpus,
dual-model independent verification across architecturally diverse models,
and application to principal-level performance rubric auditing has not
been attempted in the literature. The contribution is empirically
validated through three-way model comparison and ablation over pipeline
components.


---


## Pipeline Overview


### Module 1 — Domain Style Learner

A small open-source language model (e.g., Qwen-2.5-7B or Mistral-7B) is
fine-tuned on the full 18,000+ BIP corpus using a causal language modeling
objective. Scores are not used at this stage. The model learns authentic
BIP writing patterns: principal voice, improvement plan vocabulary,
structural conventions, and typical document length. This model serves two
purposes: it is the primary authenticity judge in Module 3, and an
alternative generator condition in the ablation studies.

The style learner is drawn from a different model family than the generator
used in Module 2. This architectural diversity is deliberate — it reduces
the risk that both models share the same systematic biases or surface
fluency heuristics, which would undermine the independence of the dual
verification step.


### Module 2 — Rubric-Conditioned Generator

A capable language model (from a different family than Module 1) receives
a structured prompt comprising: (1) the full NEE rubric with per-level
criteria, (2) a target score level, and (3) a single real BIP sampled from
the 18k corpus as a topical anchor. The model is instructed to generate a
new BIP that addresses similar improvement themes as the anchor but
demonstrates competencies at the specified score level. One real BIP is
used as anchor per generation, providing natural topical diversity across
the synthetic pool without requiring manually curated seeds.

Score distribution in the anchor sampling is deliberately controlled to
counteract the known skew in the real corpus, producing a balanced
synthetic pool across all score levels.

Operational parameters (confirmed after pilot):
- Target pool size: scaled to produce balance across all score levels
- Generation temperature: fixed across all runs for reproducibility
- Stopping criterion: fixed total per score level, not acceptance rate
- Prompt variants: tested in pilot; final prompt locked before full run


### Module 3 — LLM-as-Judge (Dual Verification)

Every candidate synthetic BIP is evaluated on two independent axes before
entering the training pool.

**Check 1: Authenticity**
The fine-tuned domain model from Module 1 receives the candidate BIP
alongside several real BIP examples as in-context references. It is
prompted to assess whether the candidate is consistent with the writing
style, structure, and language patterns of authentic principal-authored
BIPs. The model's domain-specific prior — trained on 18k real examples —
provides a stronger authenticity signal than a general-purpose model would.
Because Module 1 and Module 2 are from different model families, the
authenticity check is not trivially passed by fluent outputs of the
generator.

**Check 2: Rubric Alignment**
A model from a separate configuration receives the NEE rubric, the
intended target score, and the candidate BIP. It is prompted to
independently score the BIP according to the rubric and assess whether
its score agrees with the intended label. Candidates where the
model-assigned score disagrees with the intended score by more than one
level are rejected.

**Verdict and Loop**
Both checks must pass for a candidate to enter the training pool.
Rejections are logged with a reason code: authenticity failure, rubric
misalignment, or both. On rejection, a new anchor is sampled from the
real corpus and the candidate is regenerated. Rejection rate and reason
distribution are recorded as pipeline diagnostics and reported.

**Judge Reliability**
Judge reliability is not assumed — it is empirically evaluated:
- Inter-judge agreement: rate at which both judges agree on accept/reject
  for the same candidate.
- Human spot-check: a sample of 20–30 accepted and 20–30 rejected
  candidates is reviewed by a human evaluator blind to judge verdicts,
  producing false positive and false negative estimates for each judge
  independently.
Results are reported as judge reliability metrics alongside pipeline
diagnostics.


### Module 4 — Final Assessor Training

The accepted synthetic BIPs, each paired with a verified score label,
form the supervised training dataset for the final assessor model. This
is a standard text classification fine-tuning task: a pretrained language
model with a classification head is trained to predict the NEE rubric
score given a BIP as input. The ordinal nature of the scoring scale is
explicitly handled in the loss function and metric reporting. The model
is trained exclusively on synthetic data in the primary condition; real
BIP scores are never used as training labels in this condition.


---


## Datasets

| Dataset              | Size              | Score Quality        | Role in Pipeline                                                        |
|----------------------|-------------------|----------------------|-------------------------------------------------------------------------|
| Real BIP corpus      | 18,000+           | Noisy / biased       | Module 1 fine-tuning; Module 2 anchor pool; bias audit target           |
| Synthetic BIP pool   | TBD (balanced)    | Verified by judges   | Module 4 training (primary condition)                                   |
| Gold standard BIPs   | 23                | Expert-adjudicated   | Final evaluation only — held out entirely from all pipeline stages      |


---


## Experimental Conditions and Ablations


### Three-Way Model Comparison

The central empirical question is whether training on synthetic data
actually generalizes to real BIPs. To answer this, three assessor models
are trained under identical architecture and training procedure, differing
only in training data composition, and all evaluated on the 23 gold
standard BIPs.

**Condition A — Synthetic only (primary)**
Assessor trained on synthetic BIPs verified by both judges. This is the
main experimental condition.

**Condition B — Real noisy labels (baseline)**
Assessor trained on real BIPs with supervisor-assigned scores, despite
known bias. This is the lower-bound baseline that motivates the project.

**Condition C — Hybrid**
Assessor trained on synthetic and real BIPs combined. This tests whether
authentic writing signal from real data complements the balanced, verified
labels from synthetic data.

Differences in evaluation performance across conditions are attributable
solely to training data composition.


### Ablation Studies

These isolate the contribution of each pipeline component and directly
justify the dual-model architecture claim.

**Ablation 1 — No authenticity filter**
Run generation and rubric alignment check only. Accept all
rubric-passing BIPs regardless of authenticity verdict. Train assessor
on this pool. Measures the contribution of the authenticity check.

**Ablation 2 — No rubric alignment filter**
Run generation and authenticity check only. Accept all authenticity-passing
BIPs regardless of rubric alignment verdict. Train assessor on this pool.
Measures the contribution of the rubric check.

**Ablation 3 — Single-model self-judge**
Use the same model for both generation and rubric alignment judging.
Train assessor and evaluate. Measures the contribution of using
architecturally distinct models rather than a self-judging setup.

**Ablation 4 — No anchor conditioning**
Generate synthetic BIPs without anchoring to real BIPs (rubric and score
only, no topical anchor). Train assessor and evaluate. Measures the
contribution of diversity via real corpus anchoring.

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
- Adjacent agreement: percentage of BIPs where predicted score is within
  one point of expert score.
- Human baseline: inter-rater agreement between NEE supervisors on the
  same 23 BIPs, providing a human-performance reference point.
- Bootstrap confidence intervals (1000 resamples) on QWK to account for
  the small evaluation set size.
- Leave-one-out evaluation reported alongside bootstrap CIs to extract
  maximum signal from 23 examples.


### Error Analysis

Disagreements on the gold standard set are analyzed structurally:
- Breakdown by score level: does the model systematically over- or
  under-predict at specific rubric levels?
- Breakdown by BIP length and vocabulary complexity.
- Qualitative inspection of the highest-confidence errors: cases where
  the predicted score is far from the expert label.
- At least five example-based analyses reported in the paper.


### Pipeline Diagnostics

- Rejection rate per score level (Module 3 output).
- Rejection reason distribution (authenticity / rubric / both).
- Score distribution of accepted synthetic pool vs. real corpus.
- Inter-judge agreement rate.
- Human spot-check false positive and false negative estimates.


### Synthetic Data Distribution Characterization

Accepted synthetic BIPs are characterized against the real corpus on:
- Length distribution (token count, sentence count).
- Vocabulary diversity (type-token ratio, out-of-vocabulary rate relative
  to real corpus vocabulary).
- Structural variation (paragraph count, section presence).
- Semantic diversity (pairwise cosine distance in embedding space across
  synthetic pool vs. within real corpus).
- N-gram diversity (distinct unigram and bigram ratios).

These metrics are reported as a distribution characterization table and
used to flag any systematic over-uniformity in the synthetic pool.


---


## Bias Audit: Applying the Assessor to Real BIPs

A successful audit is defined as: the trained assessor produces
rubric-predicted scores for real BIPs that differ systematically and
non-randomly from supervisor-assigned scores, and those deviations exhibit
interpretable patterns consistent with known sources of evaluator bias.

Once the assessor is trained and validated on the gold standard set, it is
applied to the full 18k real BIP corpus. For each BIP, the model produces
a rubric-predicted score. This is compared against the supervisor-assigned
score to produce a deviation measure. The audit analysis addresses:

- Score inflation / deflation: is there a directional bias in supervisor
  scores relative to model predictions at the corpus level?
- Score level asymmetry: are deviations concentrated at specific rubric
  levels?
- Subgroup analysis: if metadata is available (supervisor ID, district,
  year, school type), do deviation patterns correlate with those variables?
  This tests whether bias is supervisor-specific, institutional, or
  temporal.
- Deviation distribution: what proportion of supervisor scores agree with
  model predictions, are adjacent, or are far off?

This analysis is explicitly framed as exploratory and correlational. The
model does not adjudicate which score is correct; it surfaces patterns
inconsistent with consistent rubric application.


---


## Validation Plan


### Pre-Run Pilot

Before running the full synthetic generation pipeline, a batch of 30–50
synthetic BIPs per score level is generated and reviewed by at least one
domain expert (education policy expert or NEE-familiar reviewer). The
expert assesses: (1) whether BIPs are plausibly principal-authored,
(2) whether score assignments are consistent with the rubric, and
(3) whether the improvement plans described are pedagogically plausible —
not merely well-written or rubric-aligned. Pilot results are reported
alongside quantitative pipeline metrics. Prompt templates are locked after
the pilot; no further modification occurs during the full run.


### Judge Calibration

Both judges are evaluated on the 23 gold standard BIPs before the full
generation run. Verdicts on known-good examples are recorded to confirm
calibration. A judge that consistently rejects gold standard BIPs
indicates miscalibration and requires investigation before deployment.


### Human Spot-Check

A random sample of 20–30 accepted and 20–30 rejected candidates from the
full run is reviewed by a human evaluator blind to judge verdicts. This
produces false positive and false negative estimates for each judge
independently and constitutes the primary evidence for judge reliability
claims in the paper.


### Final Evaluation Isolation

The 23 gold standard BIPs are withheld from all pipeline stages including
pilot review, fine-tuning, and judge calibration. They are used only for
final assessor evaluation.


---


## Explicit Non-Goals

- The pipeline does not replace human principal assessment. It audits
  existing human-assigned scores for consistency with rubric criteria.
- The assessor model is not deployed as a scoring tool. It is a research
  instrument for detecting systematic bias in supervisor scores.
- The project does not claim that synthetic BIPs are indistinguishable
  from real BIPs — only that they pass explicit authenticity and alignment
  criteria sufficient for training an auditor.
- The audit analysis does not adjudicate which score is correct. It
  surfaces deviations and patterns; human interpretation is required.
- The final assessor is not evaluated on its ability to generalize beyond
  the NEE rubric domain or beyond the BIP document type.
- Pedagogical validity of generated BIPs is not claimed beyond expert
  plausibility checks in the pilot.


---


## Known Limitations

- The gold standard evaluation set of 23 BIPs is small. Bootstrap CIs and
  leave-one-out evaluation partially compensate, but results should be
  interpreted with this constraint explicit.
- Circularity risk: the generator and rubric judge reason about the same
  rubric. Using architecturally distinct model families reduces but does
  not eliminate the risk that both share systematic rubric interpretation
  biases. Human spot-checks and expert pilot review are the primary
  mitigations.
- Authenticity as measured by Module 1 reflects writing style consistency,
  not factual accuracy or pedagogical soundness. A BIP can be authentic in
  style and rubric-aligned but describe improvement plans that are
  unrealistic or educationally unsound. This is partially addressed through
  expert pilot review but is an inherent limitation of the automatic
  pipeline.
- Judge reliability is estimated via human spot-check on a sample, not
  exhaustively verified. The sample size is sufficient for directional
  estimates but not for precise error rate quantification.
- The bias audit is correlational. Systematic deviations between supervisor
  scores and model predictions indicate inconsistency but do not establish
  causal mechanisms of bias.
- Subgroup bias analysis is contingent on available metadata. If supervisor
  ID, district, or school metadata is incomplete, subgroup analyses will be
  limited in scope.
- Prompt non-determinism: even at low temperature, generation models
  exhibit variance. All prompts are locked after pilot and all runs are
  logged, but exact reproducibility of the full synthetic pool is not
  guaranteed across different hardware or API versions.


---


## Ethical Considerations

This project operates in a high-stakes domain — the performance evaluation
of school principals. Several ethical considerations apply:

- Misuse risk: the trained assessor could in principle be repurposed as
  an automated scorer rather than an auditor. This project explicitly does
  not advocate for automated scoring of principals. The model is a research
  instrument; deployment decisions require human oversight and institutional
  review.
- Fairness: if the training corpus encodes historical biases against
  principals of particular backgrounds, the assessor may inherit those
  biases. The bias audit is designed to surface, not reproduce, such
  patterns. Subgroup analyses are reported as exploratory with appropriate
  caveats.
- Transparency: all model decisions in the pipeline — generation, judge
  verdicts, assessor predictions — are logged with evidence. Every accepted
  synthetic BIP and every assessor prediction is traceable to its inputs
  and the criteria applied.
- Principal privacy: the 18k real BIPs are used for model training only.
  No individual principal is identified in any reported result. Audit
  findings are reported at the aggregate or subgroup level, not at the
  individual level.
- Scope of claims: results are scoped to the NEE rubric, the specific
  corpus, and the specific time period of the data. No generalization
  claims are made beyond this scope.


---


## Project Outputs

### Artifacts
- Trained domain style model (Module 1 fine-tune checkpoint).
- Synthetic BIP dataset with verified score labels, accept/reject logs,
  and per-candidate judge verdicts.
- Trained assessor model checkpoints for all three training conditions.
- Full pipeline code, prompt templates, generation logs, and evaluation
  scripts — publicly released on GitHub.
- Distribution characterization report for the synthetic pool.

### Written Output
- A research paper describing the pipeline architecture, three-way model
  comparison, ablation results, and bias audit findings, targeting an NLP
  or educational NLP venue (BEA workshop, EMNLP, NAACL, or similar).
- Detailed pipeline documentation for reproducibility, including all
  prompt templates, judge calibration logs, and human spot-check protocol.