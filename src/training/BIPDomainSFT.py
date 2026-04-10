from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.utils.config import BIPDomainSFTConfig


def setup_model_and_tokenizer(config: BIPDomainSFTConfig):
    """
    Loads the base model and tokenizer, applies LoRA adapters,
    and returns both ready for training.
    """

    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {config.model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype = torch.float16,
        device_map = "auto",
    )

    # apply LoRA
    lora_config = LoraConfig(
        task_type= TaskType.CAUSAL_LM,
        r=config.lora.r,
        lora_alpha= config.lora.lora_alpha,
        target_modules= config.lora.target_modules,
        bias= config.lora.bias,
        lora_dropout= config.lora.lora_dropout
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(config: BIPDomainSFTConfig, dataset: Dataset) -> None :
    """
    Fine-tunes Qwen on the BIP corpus using causal language modeling.
    Saves the LoRA adapter checkpoint to config.output_dir.
    """

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    model, tokenizer = setup_model_and_tokenizer(config)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        logging_dir=config.log_dir,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb",
        run_name=f"BIPDomainSFT-{config.model_name.split('/')[-1]}",
        seed=config.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset= dataset,
        data_collator= data_collator,
    )

    print("Starting BIPDomainSFT training...")
    trainer.train()

    print(f"Saving adapter to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("Done.")



if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data.corpus import load, for_style_learner
    from src.data.dataset import make_clm_dataset

    parser = argparse.ArgumentParser(description="Train BIPDomainSFT")
    parser.add_argument("--data", required=True, help="Path to BIP CSV file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B", help="Base model name")
    parser.add_argument("--output", default="models/BIPDomainSFT", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run 5 steps only to verify pipeline works")
    args = parser.parse_args()

    config = BIPDomainSFTConfig(
        model_name=args.model,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
    )

    print("Loading corpus...")
    df = load(args.data)
    sl_df = for_style_learner(df)
    print(f"Training examples: {len(sl_df):,}")

    print("Tokenizing dataset...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = make_clm_dataset(sl_df, tokenizer, max_length=config.max_seq_length)

    if args.smoke_test:
        print("Smoke test mode -- truncating to 50 examples")
        dataset = dataset.select(range(50))
        config.num_train_epochs = 1

    train(config, dataset)