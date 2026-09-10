from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from src.utils.config import AssessorConfig
from src.training.ordinal_loss import OrdinalCrossEntropy, compute_class_weights


# rubric score -> class index. The rubric defines 0, 2, 4 only.
LABEL_MAP = {0: 0, 2: 1, 4: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def build_assessor_dataset(
    texts: list[str],
    elements: list[str],
    scores: list[int],
    tokenizer,
    max_length: int = 1024,
) -> Dataset:
    """
    Tokenizes BIP text for score classification.

    The element label is prefixed because the same response can warrant
    different scores under different elements -- Element 5 requires
    cited research where Element 6 requires monitoring evidence. An
    element-blind assessor cannot represent that.
    """
    prefixed = [f"{e}: {t}" for e, t in zip(elements, texts)]
    labels = [LABEL_MAP[int(s)] for s in scores]

    hf = Dataset.from_dict({"text": prefixed, "labels": labels})

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        out["labels"] = batch["labels"]
        return out

    return hf.map(
        tokenize, batched=True, remove_columns=["text"],
        load_from_cache_file=False,
    )


def load_condition_data(
    condition: str,
    synthetic_path: str | None,
    anchor_df: pd.DataFrame | None,
    seed: int = 42,
) -> tuple[list[str], list[str], list[int]]:
    """
    Assembles training data for one experimental condition.

    A -- synthetic_only: generated text with target scores. Labels come
         from the rubric via reward selection, never from supervisor
         judgement.

    B -- real_noisy: real BIPs with supervisor scores at their natural
         distribution (95% score 4). The lower bound that motivates
         the project.

    C -- balanced_real: real BIPs downsampled to match the synthetic
         score distribution. The control separating two explanations
         for any gain in A. If A beats B only because synthetic data is
         score-balanced, C will match A. If A beats C too, the gain
         comes from label quality.

    D -- hybrid: both pools combined.
    """
    rng = np.random.default_rng(seed)

    def _synthetic():
        df = pd.read_json(synthetic_path, lines=True)
        return (
            df["completion"].tolist(),
            df["element"].tolist(),
            df["target_score"].astype(int).tolist(),
        )

    def _real(balance_to: dict[int, int] | None = None):
        df = anchor_df
        if balance_to:
            parts = []
            for score, n in balance_to.items():
                pool = df[df["score"] == score]
                if len(pool) == 0:
                    continue
                # score 0 has 31 real anchors total, fewer than the
                # synthetic pool's count, so replacement is required to
                # hit the target distribution
                take = pool.sample(
                    n=min(n, len(pool)) if len(pool) >= n else n,
                    replace=len(pool) < n,
                    random_state=int(rng.integers(0, 1_000_000)),
                )
                parts.append(take)
            df = pd.concat(parts)
        return (
            df["Text"].tolist(),
            df["Element_numberX"].tolist(),
            df["score"].astype(int).tolist(),
        )

    if condition == "synthetic_only":
        return _synthetic()

    if condition == "real_noisy":
        return _real()

    if condition == "balanced_real":
        _, _, syn_scores = _synthetic()
        target_dist = pd.Series(syn_scores).value_counts().to_dict()
        return _real(balance_to=target_dist)

    if condition == "hybrid":
        st, se, ss = _synthetic()
        rt, re_, rs = _real()
        return st + rt, se + re_, ss + rs

    raise ValueError(f"unknown condition: {condition}")


def setup_assessor(
    config: AssessorConfig,
    class_weights: torch.Tensor | None = None,
    domain_checkpoint: str | None = "models/BIPDomainSFT",
    score_head_std: float = 1e-3,
):
    """
    Builds the classification model.

    The BIPDomainSFT adapter was trained with a causal LM head and
    cannot attach directly to a sequence classification model. To carry
    the domain knowledge across, the adapter is merged into the base
    weights, saved, and reloaded as a classifier -- the transformer
    trunk is shared between Qwen2ForCausalLM and
    Qwen2ForSequenceClassification, so the merged weights transfer and
    only the score head is randomly initialised.

    Pass domain_checkpoint=None to start from stock Qwen instead, which
    is the cleaner ablation for what the domain fine-tune contributed.

    score_head_std: the head is re-initialised at 1e-3 rather than the
    HuggingFace default of 0.02. Measured at the default, the merged
    Qwen's final hidden states produced logits spanning -12.9 to 26.0
    (std 8.4) before any training. Cross entropy started at 6.16
    instead of the ~1.10 expected for three classes, gradient norms hit
    3700 against max_grad_norm=1.0, and every update was clipped to a
    direction dominated by noise. The result was a model that predicted
    only the extreme classes: recall on score 2 was 0.089 across 90
    examples, and QWK came out at -0.038.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_path = config.model_name

    if domain_checkpoint and Path(domain_checkpoint).exists():
        merged_path = "models/_merged_domain_lm"
        if not Path(merged_path).exists():
            print(f"Merging {domain_checkpoint} into base weights...")
            causal = AutoModelForCausalLM.from_pretrained(
                config.model_name, dtype=torch.bfloat16, device_map="cpu",
            )
            merged = PeftModel.from_pretrained(
                causal, domain_checkpoint
            ).merge_and_unload()
            merged.save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)
            del causal, merged
            torch.cuda.empty_cache()
        base_path = merged_path
        print(f"Assessor initialised from domain LM: {base_path}")
    else:
        print(f"Assessor initialised from stock base: {base_path}")

    model = AutoModelForSequenceClassification.from_pretrained(
        base_path,
        num_labels=config.num_labels,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        model.score.weight.normal_(mean=0.0, std=score_head_std)
        if getattr(model.score, "bias", None) is not None:
            model.score.bias.zero_()
    print(f"Score head re-initialised at std={score_head_std}")

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    return model, tokenizer


class OrdinalTrainer(Trainer):
    """
    Trainer that swaps in the ordinal loss, and logs logit magnitude on
    the first step.

    The magnitude check exists because a scale problem in the
    classification head is invisible in the loss curve alone -- it just
    looks like a large starting loss -- but is obvious from the logits.
    """

    def __init__(self, *args, ordinal_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ordinal_loss = ordinal_loss
        self._logged_init = False

    def compute_loss(
        self, model, inputs, return_outputs=False, **kwargs
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if not self._logged_init:
            with torch.no_grad():
                l = logits.float()
                print(f"[init check] logits min={l.min():.2f} "
                      f"max={l.max():.2f} std={l.std():.2f}  "
                      f"(expect roughly |logit| < 3 at start)")
            self._logged_init = True

        if self.ordinal_loss is not None:
            loss = self.ordinal_loss(logits, labels)
        else:
            loss = torch.nn.functional.cross_entropy(
                logits.float(), labels
            )

        return (loss, outputs) if return_outputs else loss


def train_assessor(
    config: AssessorConfig,
    dataset: Dataset,
    tokenizer,
    class_weights: torch.Tensor | None = None,
    domain_checkpoint: str | None = "models/BIPDomainSFT",
) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    model, _ = setup_assessor(
        config, class_weights, domain_checkpoint=domain_checkpoint
    )

    loss_fn = None
    if config.ordinal_loss:
        loss_fn = OrdinalCrossEntropy(
            num_classes=config.num_labels,
            class_weights=class_weights,
        )
        print(f"Ordinal loss active, class weights: "
              f"{class_weights.tolist() if class_weights is not None else None}")
    else:
        print("Plain cross entropy (ordinal loss disabled)")

    eff = config.per_device_train_batch_size * config.gradient_accumulation_steps
    steps_per_epoch = max(1, len(dataset) // eff)
    total = steps_per_epoch * config.num_train_epochs
    warmup = int(total * config.warmup_ratio)
    print(f"Total steps: {total}, warmup: {warmup}, lr: {config.learning_rate}")

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=warmup,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="wandb",
        run_name=f"Assessor-{config.training_condition}",
        seed=config.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = OrdinalTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=default_data_collator,
        ordinal_loss=loss_fn,
    )

    print(f"Training assessor: condition={config.training_condition}")
    trainer.train()

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Saved to {config.output_dir}")