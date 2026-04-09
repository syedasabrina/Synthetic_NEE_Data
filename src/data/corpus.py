from __future__ import annotations
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# NEE rubric defines scores 0, 2, 4 only.
# Supervisors sometimes use 1 and 3 as interpolations.
# We map them to the nearest defined anchor.
SCORE_MAP = {
    0.0: 0,
    1.0: 2,
    2.0: 2,
    3.0: 4,
    4.0: 4,
}

VALID_SCORES = [0, 2, 4]

MIN_TOKENS = 50


def load(path: str | Path, text_col: str = "Text", score_col: str = "Supervisor_Score_x",
    element_col: str = "Element_numberX", min_tokens: int = MIN_TOKENS,
    deduplicate: bool = True) -> pd.DataFrame:

    path = Path(path)
    df = pd.read_csv(path, low_memory=False)

    # apply score mapping
    df["score"] = pd.to_numeric(df[score_col], errors="coerce").map(SCORE_MAP)
    df["score"] = df["score"].astype("Int64")  # nullable integer, preserves NaN
    # token count on raw text
    df["token_count"] = df[text_col].fillna("").apply(lambda x: len(x.split()))

    # filter: must have text above min length
    df = df[df["token_count"] >= min_tokens].copy()

    # deduplicate by text
    if deduplicate:
        df = df.drop_duplicates(subset=[text_col]).copy()

    # add element-prefixed text for training
    df["text_with_prefix"] = df[element_col] + ": " + df[text_col].fillna("")

    df = df.reset_index(drop=True)
    return df


def for_style_learner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns all usable BIPs regardless of score.
    Used for Module 1 causal LM fine-tuning.
    Scores are not required — we are learning style, not labels.
    """
    return df[["text_with_prefix", "token_count", "Element_numberX"]].copy()


def for_anchor_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns only scored BIPs with valid rubric scores.
    Used for Module 2 anchor sampling during generation.
    Each row is a real BIP anchored to a verified score level.
    """
    scored = df[df["score"].isin(VALID_SCORES)].copy()
    return scored[["text_with_prefix", "Text", "score", "Element_numberX", "token_count"]].copy()


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    df = load(path)
    print(f"Loaded: {len(df):,} rows after filtering")
    sl = for_style_learner(df)
    print(f"Style learner: {len(sl):,} BIPs")
    ap = for_anchor_pool(df)
    print(f"Anchor pool: {len(ap):,} BIPs")
    print(f"\nScore distribution in anchor pool:")
    print(ap["score"].value_counts().sort_index())
    print(f"\nElement distribution in anchor pool:")
    print(ap["Element_numberX"].value_counts().sort_index())