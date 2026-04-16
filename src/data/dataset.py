from __future__ import annotations

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizer


def make_clm_dataset(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    text_col: str = "text_with_prefix",
    max_length: int = 1553,
) -> Dataset:

    hf_dataset = Dataset.from_pandas(
        df[[text_col]].reset_index(drop=True)
    )

    def tokenize(batch):
        tokenized = tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = [
            [-100 if t == tokenizer.pad_token_id else t for t in ids]
            for ids in tokenized["input_ids"]
        ]
        return tokenized

    tokenized_dataset = hf_dataset.map(
        tokenize,
        batched=True,
        remove_columns=[text_col],
        load_from_cache_file=False,
    )

    return tokenized_dataset


if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer
    from corpus import load, for_style_learner

    path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-0.5B"

    print(f"Loading corpus from {path}...")
    df = load(path)
    sl_df = for_style_learner(df)

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building dataset...")
    dataset = make_clm_dataset(sl_df, tokenizer)

    print(f"\nDataset size: {len(dataset):,} examples")
    print(f"Features: {dataset.features}")
    print(f"\nFirst example input_ids length: {len(dataset[0]['input_ids'])}")
    print(f"Labels length: {len(dataset[0]['labels'])}")
    print(tokenizer.decode(dataset[0]['input_ids'][:50]))
