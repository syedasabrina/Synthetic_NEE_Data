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
    # rubric defines 0, 2, 4 only
    # 1 maps to 2 (minimal anchor), 3 maps to 4 (exemplary anchor)
    score_map: dict = field(
        default_factory=lambda: {0.0: 0, 1.0: 2, 2.0: 2, 3.0: 4, 4.0: 4}
    )
    valid_scores: list[int] = field(default_factory=lambda: [0, 2, 4])
    min_token_length: int = 50
    elements: list[str] = field(default_factory=lambda: [
        "Element1", "Element2", "Element3",
        "Element4", "Element5", "Element6", "Element7"
    ])
    # Elements 1-5 should not be evaluated for new principals per rubric note
    evaluation_elements: list[str] = field(
        default_factory=lambda: ["Element6", "Element7"]
    )


@dataclass
class AnchorSamplerConfig:
    # anchor pool after 50-token filter and score mapping:
    # score 0: 31 BIPs  score 2: 406 BIPs  score 4: 8,961 BIPs
    # score 0 is intentionally thin -- reflects genuine rarity of non-engagement
    # allow reuse for score 0 rather than crashing
    seed: int = 42


# ── Generation ───────────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    model_name: str = "gemini-2.0-flash"
    rubric_path: str = "docs/rubric.tsv"
    bip_form_path: str = "docs/2021_2022BIPProcessOrganizer.pdf"
    target_accepted_per_element_score: dict = field(
        default_factory=lambda: {0: 30, 2: 200, 4: 500}
    )
    temperature: float = 0.7
    # target length range from EDA: p50=150, p90=469 tokens
    # length correlates with score in real data -- control for this in prompt
    target_min_tokens: int = 150
    target_max_tokens: int = 469
    max_api_tokens: int = 1024
    max_retries: int = 5
    anchor_similarity_threshold: float = 0.85
    output_dir: str = "data/synthetic/candidates"
    accepted_dir: str = "data/synthetic/accepted"
    rejected_dir: str = "data/synthetic/rejected"
    log_dir: str = "logs/generation"
    seed: int = 42


# ── Judges ───────────────────────────────────────────────────────────

@dataclass
class AuthenticityJudgeConfig:
    model_name: str = "Qwen/Qwen2.5-7B"
    model_path: str = ""            # filled after Module 1 training
    n_incontext_examples: int = 5
    log_dir: str = "logs/judges"


@dataclass
class RubricJudgeConfig:
    model_name: str = "gemini-2.0-flash"
    rubric_path: str = "docs/rubric.tsv"
    # valid scores are 0, 2, 4
    # reject if judge score differs from intended score by more than one level
    # one level means: 0 vs 2 is adjacent, 0 vs 4 is not
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
class BIPDomainSFTConfig:
    model_name: str = "Qwen/Qwen2.5-7B"
    output_dir: str = "models/BIPDomainSFT"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    # from EDA: p99=1297 tokens + 256 buffer, capped at 4096
    max_seq_length: int = 1553
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    fp16: bool = False
    bf16: bool = True
    log_dir: str = "logs/BIPDomainSFT"
    seed: int = 42


@dataclass
class AssessorConfig:
    model_name: str = "Qwen/Qwen2.5-7B"
    output_dir: str = "models/assessor"
    training_condition: str = "synthetic_only"
    # 3 classes matching rubric anchors: 0, 2, 4
    num_labels: int = 3
    label_map: dict = field(
        default_factory=lambda: {0: 0, 2: 1, 4: 2}
    )
    # element is a required input feature -- assessor is element-aware
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
    # rurality gap confirmed: 37-point difference between urban and rural
    # temporal trend confirmed: score inflation from 2021-2022 onward
    subgroup_columns: list[str] = field(
        default_factory=lambda: ["DistrictID", "Rurality", "SchoolYear"]
    )
    run_subgroup_analysis: bool = True
    # prob4_ElementX is a pre-existing model score signal -- use as audit baseline
    existing_model_score_column: str = "prob4_ElementX"


# ── Top-level pipeline config ─────────────────────────────────────────

@dataclass
class PipelineConfig:
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    anchor_sampler: AnchorSamplerConfig = field(default_factory=AnchorSamplerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judges: JudgeConfig = field(default_factory=JudgeConfig)
    bip_domain_sft: BIPDomainSFTConfig = field(default_factory=BIPDomainSFTConfig)
    assessor: AssessorConfig = field(default_factory=AssessorConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    run_id: str = ""