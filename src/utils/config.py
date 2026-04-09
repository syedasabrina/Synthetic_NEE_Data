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
    corpus_path: str = "data/raw"           # directory of real BIPs
    gold_path: str = "data/gold"            # 23 gold standard BIPs
    anchor_path: str = "data/anchors"       # sampled anchor pool
    score_column: str = "score"
    text_column: str = "text"
    score_levels: list[int] = field(default_factory=lambda: [1, 2, 3, 4])


@dataclass
class AnchorSamplerConfig:
    n_per_level: int = 500          # anchors to sample per score level
    balance_scores: bool = True     # oversample underrepresented levels
    seed: int = 42


# ── Generation ───────────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    model_name: str = "gemini-2.0-flash"
    rubric_path: str = "docs/nee_rubric.txt"
    target_accepted_per_level: int = 1000
    temperature: float = 0.7
    max_tokens: int = 2048
    max_retries: int = 5            # retries per anchor before resampling
    anchor_similarity_threshold: float = 0.85   # cosine sim ceiling
    output_dir: str = "data/synthetic/candidates"
    accepted_dir: str = "data/synthetic/accepted"
    rejected_dir: str = "data/synthetic/rejected"
    log_dir: str = "logs/generation"
    seed: int = 42


# ── Judges ───────────────────────────────────────────────────────────

@dataclass
class AuthenticityJudgeConfig:
    model_name: str = ""            # filled after Module 1 training
    model_path: str = ""            # local checkpoint path
    n_incontext_examples: int = 5   # real BIP examples shown as reference
    threshold: str = "pass"         # verdict label to accept on
    log_dir: str = "logs/judges"


@dataclass
class RubricJudgeConfig:
    model_name: str = "gemini-2.0-flash"
    rubric_path: str = "docs/nee_rubric.txt"
    max_score_deviation: int = 1    # reject if |judge_score - target| > this
    log_dir: str = "logs/judges"


@dataclass
class JudgeConfig:
    authenticity: AuthenticityJudgeConfig = field(
        default_factory=AuthenticityJudgeConfig
    )
    rubric: RubricJudgeConfig = field(
        default_factory=RubricJudgeConfig
    )
    spot_check_n: int = 100         # human spot-check sample size


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
    max_seq_length: int = 2048
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    fp16: bool = True
    log_dir: str = "logs/training"
    seed: int = 42


@dataclass
class AssessorConfig:
    model_name: str = "Qwen/Qwen2.5-7B"
    output_dir: str = "models/assessor"
    training_condition: str = "synthetic_only"   # synthetic_only | real_noisy |
                                                  # balanced_real | hybrid
    num_labels: int = 4
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    max_seq_length: int = 2048
    ordinal_loss: bool = True        # use ordinal-aware loss
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    fp16: bool = True
    log_dir: str = "logs/training"
    seed: int = 42


# ── Evaluation ───────────────────────────────────────────────────────

@dataclass
class EvaluationConfig:
    gold_path: str = "data/gold"
    model_path: str = ""                  # assessor checkpoint to evaluate
    results_dir: str = "results/assessor"
    bootstrap_n: int = 1000
    confidence_level: float = 0.95
    run_difficulty_analysis: bool = True
    synthetic_held_out_path: str = ""     # for difficulty gap analysis


# ── Audit ────────────────────────────────────────────────────────────

@dataclass
class AuditConfig:
    corpus_path: str = "data/raw"
    assessor_path: str = ""               # trained assessor checkpoint
    metadata_path: Optional[str] = None  # supervisor/district/year metadata
    results_dir: str = "results/audit"
    log_dir: str = "logs/audit"
    high_confidence_entropy_threshold: float = 0.5
    subgroup_columns: list[str] = field(
        default_factory=lambda: ["supervisor_id", "district", "year"]
    )
    run_subgroup_analysis: bool = True


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
    run_id: str = ""                      # set by reproducibility.py at runtime
