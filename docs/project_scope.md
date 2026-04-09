# Project Scope

## Project Title (Loosely)
Rubric-Aligned Synthetic BIP Generation with Dual-Model Verification
for Principal Assessment Auditing


## One-Line Summary
A synthetic data generation pipeline that produces rubric-aligned,
authenticity-verified Building Improvement Plans (BIPs) to train and
evaluate an LLM-based proxy for rubric-consistent scoring — validated
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
these scores requires high-quality labeled training data — data that does
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
consistent with patterns of evaluator inconsistency — producing a finding
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


## Primary Research Contribution

The novel contribution of this project is the dual-model synthetic data
generation architecture with independent dual-channel verification,
combined with its application to surfacing systematic scoring deviations
in a real principal assessment corpus. Specifically:

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
  rubric-predicted scores, with formal statistical testing to distinguish
  structured deviation from random noise.

Synthetic data generation for educational assessment has been explored
in prior work, but the combination of topic anchoring from a real corpus,
dual-model independent verification across architecturally diverse models,
difficulty calibration analysis, and application to principal-level
performance rubric auditing has not been attempted in the literature. The
contribution is empirically validated through four-way model comparison
and ablation over pipeline components.


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

**Anchor leakage control:** To prevent synthetic outputs from paraphrasing
or closely mirroring their anchor, each generated BIP is filtered by
embedding cosine similarity against its anchor. Candidates exceeding a
similarity threshold are rejected and regenerated with a fresh anchor.
The distribution of anchor-to-synthetic similarity scores is reported as
a pipeline diagnostic.

**Operational parameters (confirmed after pilot):**
- Target pool size: scaled to produce balance across all score levels
- Generation temperature: fixed across all runs for reproducibility
- Stopping criterion: fixed total accepted per score level, not attempts
- Prompt variants: tested in pilot; final prompt locked before full run
- Generation effort: attempts-per-accepted-sample reported by score level
  to detect diversity degradation at high-rejection levels


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
misalignment, anchor leakage, or both. On rejection, a new anchor is
sampled from the real corpus and the candidate is regenerated. Rejection
rate and reason distribution are recorded as pipeline diagnostics.

**Judge Reliability**
Judge reliability is empirically evaluated, not assumed:
- Inter-judge agreement: rate at which both judges agree on accept/reject
  for the same candidate.
- Human vs judge agreement: a sample of ~100 candidates (balanced across
  accept/reject and score levels) is reviewed by a human evaluator blind
  to judge verdicts. Human verdicts are compared against each judge
  independently, producing human-judge agreement rates alongside
  false positive and false negative estimates per judge.
- This spot-check is framed explicitly as qualitative validation and
  directional estimation, not precise error rate quantification.
Results are reported as judge reliability metrics alongside pipeline
diagnostics.


### Module 4 — Final Assessor Training

The accepted synthetic BIPs, each paired with a verified score label,
form the supervised training dataset for the final assessor model. This
is a text classification fine-tuning task: a pretrained language model
with a classification head is trained to predict the NEE rubric score
given a BIP as input.

**Ordinal modeling:** The ordinal nature of the scoring scale is handled
explicitly using one of the following (selected after pilot): ordinal
cross-entropy loss with adjacent-score penalty weighting, or QWK-optimized
training objective. The specific choice is justified in the paper against
the score distribution of the training data. Reporting uses QWK as the
primary metric throughout to maintain consistency with the ordinal
assumption.

The model is trained exclusively on synthetic data in the primary
condition; real BIP scores are never used as training labels in this
condition.


---


## Datasets

| Dataset                    | Size            | Score Quality       | Role in Pipeline                                              |
|----------------------------|-----------------|---------------------|---------------------------------------------------------------|
| Real BIP corpus            | 18,000+         | Noisy / biased      | Module 1 fine-tuning; Module 2 anchor pool; audit target      |
| Synthetic BIP pool         | TBD (balanced)  | Verified by judges  | Module 4 training (primary condition)                         |
| Gold standard BIPs         | 23              | Expert-adjudicated  | Final evaluation only — held out from all pipeline stages     |


---


## Experimental Conditions and Ablations


### Four-Way Model Comparison

The central empirical question is whether training on synthetic data
generalizes to real BIPs, and whether any observed improvement is
attributable to data quality or simply to score distribution balancing.
Four assessor models are trained under identical architecture and training
procedure, differing only in training data composition, and all evaluated
on the 23 gold standard BIPs.

**Condition A — Synthetic only (primary)**
Assessor trained on synthetic BIPs verified by both judges, with balanced
score distribution. This is the main experimental condition.

**Condition B — Real noisy labels (lower-bound baseline)**
Assessor trained on real BIPs with supervisor-assigned scores, preserving
the natural skewed distribution. Establishes the floor that motivates
the project.

**Condition C — Balanced real subset (distribution control)**
Assessor trained on a downsampled or reweighted subset of real BIPs
matched to the synthetic pool's score distribution. This condition
isolates the effect of distribution balancing from the effect of synthetic
data quality. If Condition A outperforms Condition C, the improvement is
attributable to verified synthetic data quality, not merely to balancing.
If they perform equivalently, the contribution reduces to the balancing
mechanism, which is itself a finding.

**Condition D — Hybrid**
Assessor trained on synthetic and real BIPs combined. Tests whether
authentic writing signal from real data complements verified labels from
synthetic data.

Differences in evaluation performance across all four conditions are
attributable solely to training data composition.


### Ablation Studies

These isolate the contribution of each pipeline component and directly
justify the dual-model architecture claim.

**Ablation 1 — No authenticity filter**
Run generation and rubric alignment check only. Accept all rubric-passing
BIPs regardless of authenticity verdict. Train assessor and evaluate.
Measures the independent contribution of the authenticity check.

**Ablation 2 — No rubric alignment filter**
Run generation and authenticity check only. Accept all
authenticity-passing BIPs regardless of rubric alignment verdict. Train
assessor and evaluate. Measures the independent contribution of the
rubric check.

**Ablation 3 — Single-model self-judge**
Use the same model for both generation and rubric alignment judging.
Train assessor and evaluate. Measures the contribution of architectural
diversity between generator and judge versus a self-judging setup.

**Ablation 4 — No anchor conditioning**
Generate synthetic BIPs without anchoring to real BIPs (rubric and score
only, no topical anchor). Train assessor and evaluate. Measures the
contribution of topical diversity via real corpus anchoring.

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


### Assessor Calibration Analysis

Beyond aggregate QWK, the assessor's calibration across the scoring scale
is explicitly characterized on the 23 gold standard BIPs:
- Directional bias check: does the model systematically over-predict or
  under-predict relative to expert scores at the corpus level? Reported
  as mean signed deviation with confidence interval.
- Score-level calibration: per-level agreement rates to detect whether
  the model is well-calibrated at some rubric levels but not others.
- Prediction confidence: entropy or softmax confidence distribution across
  predictions, used to define a high-confidence subset for the audit
  analysis.

These calibration results directly inform the interpretation of the bias
audit: deviations in the audit are reported separately for
high-confidence and low-confidence predictions, with the high-confidence
subset constituting the primary evidence.


### Difficulty Calibration

Synthetic data may be easier to classify than real BIPs even if surface
distributions match. This is tested directly:
- Model confidence (prediction entropy) on synthetic held-out BIPs vs
  real BIPs from the 18k corpus.
- Performance gap between held-out synthetic test set and the 23 gold
  standard BIPs, where a large gap indicates difficulty mismatch.
Results are reported and interpreted in context of the overall findings.


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
- Rejection reason distribution (authenticity / rubric / anchor leakage /
  both).
- Score distribution of accepted synthetic pool vs. real corpus.
- Anchor-to-synthetic cosine similarity distribution.
- Generation attempts per accepted sample by score level.
- Inter-judge agreement rate.
- Human vs judge agreement rate (spot-check sample).
- Judge false positive and false negative estimates.


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
used to flag systematic over-uniformity in the synthetic pool.


---


## Scoring Deviation Audit: Applying the Assessor to Real BIPs

Once the assessor is trained, validated on the gold standard set, and
characterized for calibration and difficulty, it is applied to the full
18k real BIP corpus. For each BIP, the model produces a rubric-predicted
score and a confidence estimate. The audit is conducted on the full corpus
and separately on the high-confidence subset.

**Definition of a meaningful finding:** A meaningful audit finding
requires that observed deviations between supervisor scores and
model-predicted scores are (a) statistically non-random at the corpus
level, (b) structured rather than uniformly distributed across score
levels, and (c) not fully explained by known assessor miscalibration
as characterized on the gold standard set.

**Statistical tests applied:**
- Mean deviation test: one-sample t-test or bootstrap test of whether
  mean(supervisor score − model score) ≠ 0 at the corpus level.
- Distributional shift: Kolmogorov-Smirnov test between supervisor score
  distribution and model-predicted score distribution.
- Subgroup analysis: where metadata is available, a regression of
  deviation on supervisor ID, district, year, and school type:
    deviation ~ supervisor_id + district + year + school_type
  Significant coefficients indicate structured, subgroup-correlated
  deviation — the condition under which the term "bias" is used.

**Interpretation scenarios are specified in advance:**
- No significant deviation: supervisors are scoring consistently with
  rubric-aligned predictions; the corpus does not exhibit detectable
  systematic inconsistency.
- Significant deviation, unstructured: high variance but no directional
  pattern; suggests low reliability in supervisor scoring rather than
  systematic inflation or deflation.
- Significant deviation, directional but not subgroup-correlated: corpus-
  level inflation or deflation relative to rubric predictions; reported
  as systematic scoring inconsistency.
- Significant deviation, subgroup-correlated: scoring patterns differ
  across supervisors, districts, or time periods in ways that cannot be
  explained by rubric variation alone; reported as potential evaluator
  bias with specific subgroup effects quantified.

The audit is framed throughout as detecting scoring deviations relative
to a rubric-consistent proxy, not as establishing ground truth. All
findings are reported with appropriate uncertainty and require human
interpretation for policy conclusions.


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

A sample of approximately 100 candidates (balanced across accept/reject
verdicts and score levels) from the full run is reviewed by a human
evaluator blind to judge verdicts. Human verdicts are compared against
each judge independently, producing human-judge agreement rates alongside
false positive and false negative estimates. This is explicitly framed as
qualitative validation and directional estimation rather than precise
error rate quantification.


### Reproducibility Stance

Exact dataset reproducibility is not claimed: generation model
non-determinism means the precise synthetic pool cannot be guaranteed
across hardware or API versions even at low temperature. What is
reproducible is pipeline behavior: given the locked prompts, fixed
temperature, and documented acceptance criteria, the distributional
properties of the synthetic pool — score balance, length distribution,
vocabulary diversity, semantic spread — are expected to be stable. All
generation logs, judge verdicts, and accepted BIPs are archived to
support this weaker but honest reproducibility claim.


### Final Evaluation Isolation

The 23 gold standard BIPs are withheld from all pipeline stages including
pilot review, fine-tuning, and judge calibration. They are used only for
final assessor evaluation.


---


## Explicit Non-Goals

- The pipeline does not replace human principal assessment. It surfaces
  scoring deviations relative to rubric-consistent predictions.
- The assessor model is not deployed as a scoring tool. It is a research
  instrument; its predictions are a proxy, not ground truth.
- The project does not claim that synthetic BIPs are indistinguishable
  from real BIPs — only that they pass explicit authenticity and alignment
  criteria sufficient for training a rubric-consistent proxy.
- The audit does not adjudicate which score is correct. It surfaces
  structured deviations; human interpretation and institutional review
  are required for any policy conclusions.
- The final assessor is not evaluated on its ability to generalize beyond
  the NEE rubric domain or beyond the BIP document type.
- Pedagogical validity of generated BIPs is not claimed beyond expert
  plausibility checks in the pilot.


---


## Known Limitations

- The gold standard evaluation set of 23 BIPs is small. Bootstrap CIs and
  leave-one-out evaluation partially compensate, but assessor calibration
  claims should be interpreted with this constraint explicit. The
  calibration analysis characterizes known model error before audit
  interpretation.
- Circularity risk: the generator and rubric judge reason about the same
  rubric. Using architecturally distinct model families reduces but does
  not eliminate the risk that both share systematic rubric interpretation
  biases. Human spot-checks and expert pilot review are the primary
  mitigations; this risk is acknowledged in all claims.
- Authenticity as measured by Module 1 reflects writing style consistency,
  not factual accuracy or pedagogical soundness. A BIP can be authentic in
  style and rubric-aligned but describe improvement plans that are
  unrealistic or educationally unsound. Expert pilot review partially
  addresses this but it remains an inherent limitation of the automatic
  pipeline.
- Judge reliability is estimated via human spot-check on a sample of ~100
  candidates. This supports directional conclusions about judge behavior
  but is not sufficient for precise false positive or false negative rate
  quantification. Claims about judge reliability are scoped accordingly.
- Synthetic data difficulty mismatch: even if surface distributions match,
  synthetic BIPs may over-express rubric signals relative to real BIPs,
  making them easier to classify. Difficulty calibration analysis
  characterizes this gap; any mismatch is reported and factored into the
  interpretation of assessor performance on real data.
- The audit is correlational. Structured deviations between supervisor
  scores and model predictions indicate inconsistency with rubric-aligned
  predictions, not causally established bias. Subgroup-correlated
  deviations are the threshold for using the term bias; all other findings
  are reported as scoring inconsistency.
- Subgroup analysis is contingent on metadata availability. If supervisor
  ID, district, or school metadata is incomplete, subgroup analyses will
  be limited in scope, and the bias threshold may not be reachable
  regardless of corpus-level findings.
- Prompt non-determinism: even at low temperature, generation models
  exhibit variance. Distributional reproducibility is claimed; exact
  dataset reproducibility is not.


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
  biases. The audit is designed to surface, not reproduce, such patterns.
  The difficulty calibration analysis and directional bias check on the
  gold standard set are the mechanisms for detecting assessor-level
  fairness issues before audit findings are reported.
- Transparency: all model decisions in the pipeline — generation, judge
  verdicts, anchor similarity scores, assessor predictions — are logged
  with evidence. Every accepted synthetic BIP and every assessor prediction
  is traceable to its inputs and the criteria applied.
- Principal privacy: the 18k real BIPs are used for model training only.
  No individual principal is identified in any reported result. Audit
  findings are reported at the aggregate or subgroup level, not the
  individual level.
- Scope of claims: results are scoped to the NEE rubric, the specific
  corpus, and the specific time period of the data. No generalization
  claims are made beyond this scope. The pre-specified interpretation
  scenarios for the audit ensure that null and ambiguous results are
  reported honestly rather than over-interpreted.


---


## Project Outputs

### Artifacts
- Trained domain style model (Module 1 fine-tune checkpoint).
- Synthetic BIP dataset with verified score labels, accept/reject logs,
  anchor similarity scores, and per-candidate judge verdicts.
- Trained assessor model checkpoints for all four training conditions.
- Full pipeline code, prompt templates, generation logs, and evaluation
  scripts — publicly released on GitHub.
- Distribution characterization and difficulty calibration reports.

### Written Output
- A research paper describing the pipeline architecture, four-way model
  comparison, ablation results, calibration analysis, and scoring deviation
  audit findings, targeting an NLP or educational NLP venue.
- Detailed pipeline documentation for reproducibility, including all
  prompt templates, judge calibration logs, and human spot-check protocol.