from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    Gemma4ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from src.data.encoding import encode_many
from src.generation.sampler import fit_anchor
from src.utils.config import GeneratorSFTConfig


def build_sft_prompt(element: str, rubric_text: str, reference_text: str) -> str:
    """
    Builds the SFT training prompt. reference_text is a real BIP shown as a
    topical/stylistic reference. The training target is a DIFFERENT real
    BIP for the same element, never the reference itself, so the model
    cannot learn to copy its input.

    NOT CHANGED in this pass: this template still differs from the one in
    src/generation/sampler.py (no element question, no target score line,
    always the score-4 rubric text). Aligning them is a separate change and
    should be made on its own so its effect can be attributed.
    """
    return f"""You are a school principal writing a Building Improvement Plan.

Element: {element}
Guidance: {rubric_text}

Reference example on a similar topic:
{reference_text}

Write your own BIP response for this element, addressing a similar
theme but in your own words:
"""


def build_sft_pairs(anchor_df, rng: np.random.Generator | None = None) -> list[dict]:
    """
    For each element, pairs every real BIP with a DIFFERENT real BIP from
    the same element to serve as the training target.
    """
    rng = rng or np.random.default_rng(42)
    pairs = []

    for element, group in anchor_df.groupby("Element_numberX"):
        texts = group["Text"].tolist()
        if len(texts) < 2:
            continue
        idxs = list(range(len(texts)))
        for i in idxs:
            others = [j for j in idxs if j != i]
            j = rng.choice(others)
            pairs.append({
                "element": element,
                "reference_text": texts[i],
                "target_text": texts[j],
            })
    return pairs


def build_sft_dataset(
    anchor_df,
    tokenizer,
    max_length: int = 1280,
    rng: np.random.Generator | None = None,
    max_reference_tokens: int = 400,
) -> Dataset:
    """
    Builds the SFT dataset. Prompt tokens are masked, EOS is appended to
    every target and included in the loss, and pairs that do not fit in
    max_length are dropped (see src/data/encoding.py for why).
    """
    from src.rewards.rubric_reward import RubricReward

    pairs = build_sft_pairs(anchor_df, rng=rng)

    prompts, completions = [], []
    for pair in pairs:
        element = pair["element"]
        # generic score-4 guidance during warmup; no score conditioning
        # happens until selection
        rubric_text = RubricReward.RUBRIC[element][4]
        # cap the reference like the BoN anchor, so length-based drops depend
        # on the target alone instead of on two full BIPs
        reference = fit_anchor(tokenizer, pair["reference_text"], max_reference_tokens)
        prompts.append(build_sft_prompt(element, rubric_text, reference))
        completions.append(pair["target_text"])

    cols, dropped = encode_many(tokenizer, prompts, completions, max_length)
    kept = len(cols["input_ids"])
    print(f"SFT pairs: {len(pairs)}  kept: {kept}  dropped for length: "
          f"{dropped} ({100 * dropped / max(len(pairs), 1):.1f}%)")
    n_eos = sum(
        1 for ids, lab in zip(cols["input_ids"], cols["labels"])
        if tokenizer.eos_token_id in [i for i, l in zip(ids, lab) if l != -100]
    )
    print(f"Examples with EOS in the loss: {n_eos}/{kept}")
    if kept == 0 or n_eos != kept:
        raise RuntimeError("EOS is missing from the SFT labels.")

    return Dataset.from_dict(cols)


def setup_generator_model_and_tokenizer(config: GeneratorSFTConfig):
    """
    Loads Gemma 4 E4B and applies LoRA using the Gemma4-safe regex
    target_modules to avoid matching the vision/audio tower wrappers.
    """
    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"eos_token={tokenizer.eos_token!r} id={tokenizer.eos_token_id}  "
          f"pad_token={tokenizer.pad_token!r} id={tokenizer.pad_token_id}")

    print(f"Loading model: {config.model_name}")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        lora_dropout=config.lora.lora_dropout,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(config: GeneratorSFTConfig, dataset: Dataset, tokenizer=None) -> None:
    """SFT warmup for the Gemma 4 E4B generator. Saves the LoRA adapter."""
    os.makedirs(config.output_dir, exist_ok=True)

    model, _tokenizer = setup_generator_model_and_tokenizer(config)
    if tokenizer is None:
        tokenizer = _tokenizer

    # transformers 5.x removed warmup_ratio; compute steps explicitly
    effective_batch = (
        config.per_device_train_batch_size * config.gradient_accumulation_steps
    )
    steps_per_epoch = max(1, len(dataset) // effective_batch)
    total_steps = steps_per_epoch * config.num_train_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    print(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        fp16=False,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb",
        run_name=f"GeneratorSFT-{config.model_name.split('/')[-1]}",
        seed=config.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    print("Starting GeneratorSFT training...")
    trainer.train()

    print(f"Saving adapter to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("Done.")


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data.corpus import load, for_anchor_pool, load_gold_person_ids

    parser = argparse.ArgumentParser(description="Train GeneratorSFT")
    parser.add_argument("--data", required=True, help="Path to BIP CSV file")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--output", default="models/GeneratorSFT")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--max_reference_tokens", type=int, default=400)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    config = GeneratorSFTConfig(
        model_name=args.model,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
    )
    if args.max_seq_length:
        config.max_seq_length = args.max_seq_length

    print("Loading corpus and anchor pool...")
    df = load(args.data)
    anchor_df = for_anchor_pool(df)
    print(f"Anchor pool size: {len(anchor_df):,}")

    # Gold principals must never reach SFT. The earlier SFT log showed
    # 9,398 pairs against an 8,905-row anchor pool today; check directly.
    gold_ids = load_gold_person_ids()
    overlap = set(anchor_df["PersonId"]) & gold_ids
    if gold_ids and overlap:
        raise RuntimeError(f"{len(overlap)} gold PersonIds in the SFT pool.")
    print(f"Gold overlap in SFT pool: {len(overlap)} "
          f"(gold ids loaded: {len(gold_ids)})")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rng = np.random.default_rng(config.seed)
    dataset = build_sft_dataset(
        anchor_df, tokenizer,
        max_length=config.max_seq_length,
        rng=rng,
        max_reference_tokens=args.max_reference_tokens,
    )

    if args.smoke_test:
        print("Smoke test mode: truncating to 50 examples")
        dataset = dataset.select(range(min(50, len(dataset))))
        config.num_train_epochs = 1

    train(config, dataset, tokenizer=tokenizer)