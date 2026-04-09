"""
src/utils/config.py
Central configuration dataclasses for all pipeline modules.
Instantiate directly or load from argparse / a config dict.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ── Data ─────────────────────────────────────────────────────────────

@dataclass
class CorpusConfig:
    corpus_path: str = "data/raw"
    gold_path: str = "data/gold"
    anchor_path: str = "data/anchors"
    score_column: str = "Supervisor_Score_x"
    text_column: str = "Text"
    element_column: str = "Element_numberX"
    score_levels: list[int] = field(default_factory=lambda: [0, 2, 3, 4])
    # supervisors used 0 and 1 interchangeably — merge to class 0
    score_merge_map: dict = field(
        default_factory=lambda: {0: 0, 1: 0, 2: 2, 3: 3, 4: 4}
    )
    min_token_length: int = 50      # drop BIPs below this threshold
    elements: list[str] = field(default_factory=lambda: [
        "Element1", "Element2", "Element3",
        "Element4", "Element5", "Element6", "Element7"
    ])
    # Elements 1-5 should not be evaluated for new principals per rubric
    evaluation_elements: list[str] = field(
        default_factory=lambda: ["Element6", "Element7"]
    )


@dataclass
class AnchorSamplerConfig:
    # usable scored BIPs after 50-token filter, by merged class:
    # class 0 (merged 0+1): ~102  class 2: ~364  class 3: ~1843  class 4: ~7928
    # class 0 is thin — allow multiple generations per anchor for that level
    n_per_level: int = 100          # base anchors per score level
    max_gen_per_anchor: dict = field(
        default_factory=lambda: {0: 5, 2: 3, 3: 2, 4: 1}
    )                               # more generations per anchor for rare classes
    balance_scores: bool = True
    seed: int = 42


# ── Generation ───────────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    model_name: str = "gemini-2.0-flash"
    rubric_path: str = "docs/rubric.tsv"
    bip_form_path: str = "docs/2021_2022BIPProcessOrganizer.pdf"
    target_accepted_per_level: int = 1000
    temperature: float = 0.7
    # target length range from EDA: p50=150, p90=469 tokens
    # instruct model to stay within this range regardless of score level
    # (length correlates with score in real data — control for this)
    target_min_tokens: int = 150
    target_max_tokens: int = 469
    max_tokens: int = 1024          # API max output tokens
    max_retries: int = 5
    anchor_similarity_threshold: float = 0.85   # cosine sim ceiling vs anchor
    output_dir: str = "data/synthetic/candidates"
    accepted_dir: str = "data/synthetic/accepted"
    rejected_dir: str = "data/synthetic/rejected"
    log_dir: str = "logs/generation"
    seed: int = 42


# ── Judges ───────────────────────────────────────────────────────────

@dataclass
class AuthenticityJudgeConfig:
    model_name: str = ""            # filled after Module 1 training
    model_path: str = ""
    n_incontext_examples: int = 5
    threshold: str = "pass"
    log_dir: str = "logs/judges"


@dataclass
class RubricJudgeConfig:
    model_name: str = "gemini-2.0-flash"
    rubric_path: str = "docs/rubric.tsv"
    # rubric defines only 0, 2, 4 — but we keep 3 as a valid target level
    # reject if judge score deviates by more than one rubric step
    # rubric steps: 0 -> 2 -> 3 -> 4  (step = 1 ordinal position)
    max_score_deviation: int = 1
    log_dir: str = "logs/judges"


@dataclass
class JudgeConfig:
    authenticity: AuthenticityJudgeConfig = field(
        default_factory=AuthenticityJudgeConfig
    )
    rubric: RubricJudgeConfig = field(
        default_factory=RubricJudgeConfig
    )
    spot_check_n: int = 100


# ── Training ─────────────────────────────────────────────────────────

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_dropout: float = 0.05
    bias: str = "none"


@dataclass
class StyleLearnerConfig:
    model_name: str = "Qwen/Qwen2.5-7B"
    output_dir: str = "models/style_learner"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    # from EDA: p99=1297 tokens + 256 buffer
    max_seq_length: int = 1553
    # deduplicate by text before training — scores are irrelevant here
    deduplicate_text: bool = True
    # usable BIPs after dedup and 50-token filter: ~13,645
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    fp16: bool = True
    log_dir: str = "logs/training"
    seed: int = 42


@dataclass
class AssessorConfig:
    model_name: str = "Qwen/Qwen2.5-7B"
    output_dir: str = "models/assessor"
    training_condition: str = "synthetic_only"  # synthetic_only | real_noisy |
                                                 # balanced_real | hybrid
    # 4 classes: 0 (merged 0+1), 2, 3, 4
    num_labels: int = 4
    # element is a required input feature — assessor is element-aware
    use_element_prefix: bool = True
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    max_seq_length: int = 1553
    ordinal_loss: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    fp16: bool = True
    log_dir: str = "logs/training"
    seed: int = 42


# ── Evaluation ───────────────────────────────────────────────────────

@dataclass
class EvaluationConfig:
    gold_path: str = "data/gold"
    model_path: str = ""
    results_dir: str = "results/assessor"
    bootstrap_n: int = 1000
    confidence_level: float = 0.95
    run_difficulty_analysis: bool = True
    synthetic_held_out_path: str = ""


# ── Audit ────────────────────────────────────────────────────────────

@dataclass
class AuditConfig:
    corpus_path: str = "data/raw"
    assessor_path: str = ""
    metadata_path: Optional[str] = None
    results_dir: str = "results/audit"
    log_dir: str = "logs/audit"
    high_confidence_entropy_threshold: float = 0.5
    # confirmed viable from EDA: 62 districts with >= 30 scored BIPs
    subgroup_columns: list[str] = field(
        default_factory=lambda: ["DistrictID", "Rurality", "SchoolYear"]
    )
    run_subgroup_analysis: bool = True
    # prob4_ElementX is a pre-existing automated score signal — use as baseline
    existing_model_score_column: str = "prob4_ElementX"


# ── Top-level pipeline config ─────────────────────────────────────────

@dataclass
class PipelineConfig:
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    anchor_sampler: AnchorSamplerConfig = field(default_factory=AnchorSamplerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judges: JudgeConfig = field(default_factory=JudgeConfig)
    style_learner: StyleLearnerConfig = field(default_factory=StyleLearnerConfig)
    assessor: AssessorConfig = field(default_factory=AssessorConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    run_id: str = ""                # set by reproducibility.py at runtime