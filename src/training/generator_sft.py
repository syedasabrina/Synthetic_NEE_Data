from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Gemma4ForConditionalGeneration,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.utils.config import GeneratorSFTConfig


def build_sft_prompt(element: str, rubric_text: str, reference_text: str) -> str:
    """
    Builds the SFT training prompt. reference_text is a real BIP shown
    as a topical/stylistic reference. The training target is a
    DIFFERENT real BIP for the same element -- never the reference
    itself. This prevents the model from learning to copy its input,
    which is the failure mode of using the same text as both
    reference and target.
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
    For each element, pairs every real BIP with a DIFFERENT real BIP
    from the same element to serve as the training target. This is
    the core fix: the reference shown in the prompt and the
    completion the model is trained to produce must never be
    identical text, or the model learns to copy verbatim.
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
    max_length: int = 1024,
    rng: np.random.Generator | None = None,
) -> Dataset:
    """
    Builds the SFT training dataset. Prompt tokens are masked with
    -100 so the loss is computed only on the completion, not on the
    rubric guidance or reference text the model is shown.
    """
    from src.rewards.rubric_reward import RubricReward

    pairs = build_sft_pairs(anchor_df, rng=rng)

    prompts, completions, elements = [], [], []
    for pair in pairs:
        element = pair["element"]
        # generic score-4 guidance during warmup -- no score
        # conditioning happens until PPO
        rubric_text = RubricReward.RUBRIC[element][4]
        prompt = build_sft_prompt(element, rubric_text, pair["reference_text"])
        prompts.append(prompt)
        completions.append(pair["target_text"])
        elements.append(element)

    hf_dataset = Dataset.from_dict({
        "prompt": prompts,
        "completion": completions,
    })

    def tokenize(batch):
        input_ids_batch, labels_batch, attn_batch = [], [], []

        for prompt, completion in zip(batch["prompt"], batch["completion"]):
            prompt_ids = tokenizer(
                prompt, add_special_tokens=True, truncation=True,
                max_length=max_length,
            )["input_ids"]

            full_ids = tokenizer(
                prompt + completion, add_special_tokens=True,
                truncation=True, max_length=max_length,
            )["input_ids"]

            # prompt_len is how many leading tokens to mask.
            # Approximate boundary via the prompt-only tokenization --
            # standard practice for causal LM completion masking
            # without a chat template.
            prompt_len = min(len(prompt_ids), len(full_ids))

            pad_len = max_length - len(full_ids)
            input_ids = full_ids + [tokenizer.pad_token_id] * pad_len
            attention_mask = [1] * len(full_ids) + [0] * pad_len

            labels = [-100] * prompt_len + full_ids[prompt_len:]
            labels = labels[:max_length]
            labels = labels + [-100] * (max_length - len(labels))

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
            attn_batch.append(attention_mask)

        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": attn_batch,
        }

    return hf_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["prompt", "completion"],
        load_from_cache_file=False,
    )


def setup_generator_model_and_tokenizer(config: GeneratorSFTConfig):
    """
    Loads Gemma 4 E4B and applies LoRA using the Gemma4-safe regex
    target_modules to avoid matching the vision/audio tower wrappers.
    """
    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {config.model_name}")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
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
    """
    SFT warmup training for the Gemma 4 E4B generator.
    Saves the LoRA adapter checkpoint to config.output_dir.
    """
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

    # default_data_collator works fine here since every example is
    # already padded to max_length with correctly masked labels
    from transformers import default_data_collator

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

    from src.data.corpus import load, for_anchor_pool

    parser = argparse.ArgumentParser(description="Train GeneratorSFT")
    parser.add_argument("--data", required=True, help="Path to BIP CSV file")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--output", default="models/GeneratorSFT")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    config = GeneratorSFTConfig(
        model_name=args.model,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
    )

    print("Loading corpus and anchor pool...")
    df = load(args.data)
    anchor_df = for_anchor_pool(df)
    print(f"Anchor pool size: {len(anchor_df):,}")

    print("Tokenizing SFT dataset with distinct reference/target pairs...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rng = np.random.default_rng(config.seed)
    dataset = build_sft_dataset(
        anchor_df, tokenizer,
        max_length=512 if args.smoke_test else config.max_seq_length,
        rng=rng,
    )

    if args.smoke_test:
        print("Smoke test mode -- truncating to 50 examples")
        dataset = dataset.select(range(min(50, len(dataset))))
        config.num_train_epochs = 1

    train(config, dataset, tokenizer=tokenizer)