from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Gemma4ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
)

from src.utils.config import GeneratorSFTConfig


def build_sft_prompt(element: str, rubric_text: str, anchor_text: str) -> str:
    """
    Builds the SFT training prompt. This teaches Gemma 4 E4B to produce
    BIP-like text given an element and a real anchor BIP as topical
    grounding. No score is included at this stage -- score
    conditioning is introduced during PPO training.
    """
    return f"""You are a school principal writing a Building Improvement Plan.

Element: {element}
Guidance: {rubric_text}

Reference example on a similar topic:
{anchor_text}

Write your own BIP response for this element, addressing a similar
theme but in your own words:
"""


def build_sft_dataset(
    anchor_df,
    tokenizer,
    max_length: int = 1024,
) -> Dataset:
    """
    Builds the SFT training dataset from the anchor pool.
    Input: prompt (element + rubric + anchor)
    Output: the anchor BIP text itself, used as the target completion.

    This trains Gemma 4 E4B to complete BIP-like text given the prompt
    structure it will see during PPO, without ever using scores.
    """
    from src.rewards.rubric_reward import RubricReward

    prompts = []
    completions = []

    for _, row in anchor_df.iterrows():
        element = row["Element_numberX"]
        anchor_text = row["Text"]

        # use score-4 rubric text as generic guidance during warmup
        # since no score conditioning happens at this stage
        rubric_text = RubricReward.RUBRIC[element][4]

        prompt = build_sft_prompt(element, rubric_text, anchor_text)
        prompts.append(prompt)
        completions.append(anchor_text)

    full_texts = [p + c for p, c in zip(prompts, completions)]

    hf_dataset = Dataset.from_dict({"text": full_texts})

    def tokenize(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = [
            [-100 if t == tokenizer.pad_token_id else t for t in ids]
            for ids in tokenized["input_ids"]
        ]
        return tokenized

    return hf_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
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
    # Gemma4ForConditionalGeneration explicitly loads the full
    # multimodal architecture. Vision/audio towers remain present
    # but idle during text-only generation.
    model = Gemma4ForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,  # regex string
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
    os.makedirs(config.log_dir, exist_ok=True)

    model, _tokenizer = setup_generator_model_and_tokenizer(config)
    if tokenizer is None:
        tokenizer = _tokenizer

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=1.0,
        bf16=config.bf16,
        logging_dir=config.log_dir,
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

    from transformers import Trainer
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

    print("Tokenizing SFT dataset...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_sft_dataset(
        anchor_df, tokenizer,
        max_length=512 if args.smoke_test else config.max_seq_length,
    )

    if args.smoke_test:
        print("Smoke test mode -- truncating to 50 examples")
        dataset = dataset.select(range(min(50, len(dataset))))
        config.num_train_epochs = 1

    train(config, dataset, tokenizer=tokenizer)