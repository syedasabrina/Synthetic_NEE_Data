"""
src/data/corpus.py

Loads and cleans the real BIP corpus, and loads the held-out gold
standard evaluation set.

Two isolation guarantees are enforced here:

1. Gold standard PersonIds are excluded from the training corpus and
   anchor pool by default. All 24 gold principals appear in the raw
   corpus, so without this exclusion BIPDomainSFT would train on the
   exact text used for final evaluation, and PPO could anchor
   generation on it. The project scope commits to withholding these
   from every pipeline stage.

2. Score mapping is applied consistently to both corpora. The NEE
   rubric defines 0, 2, 4 only; 1 and 3 are supervisor interpolations
   mapped to the nearest anchor.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
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

GOLD_PATH = "data/gold/combined_annotations_with_SupervisorScore.csv"

# The gold file was exported with Windows-era encoding, not UTF-8.
GOLD_ENCODING = "latin-1"


def load_gold_person_ids(path: str | Path = GOLD_PATH) -> set:
    """
    Returns the set of PersonIds appearing in the gold standard
    evaluation set. These are excluded from all training and anchor
    sampling to preserve evaluation isolation.

    Returns an empty set if the gold file is absent, so the loader
    still works in environments where the gold set is not present.
    """
    path = Path(path)
    if not path.exists():
        print(f"WARNING: gold file not found at {path}. "
              f"No gold exclusion applied.")
        return set()
    gold = pd.read_csv(path, encoding=GOLD_ENCODING)
    return set(gold["PersonId"].unique())


def load(
    path: str | Path,
    text_col: str = "Text",
    score_col: str = "Supervisor_Score_x",
    element_col: str = "Element_numberX",
    min_tokens: int = MIN_TOKENS,
    deduplicate: bool = True,
    exclude_gold: bool = True,
    gold_path: str | Path = GOLD_PATH,
) -> pd.DataFrame:
    """
    Loads the real BIP corpus, applies score mapping, filters short
    responses, removes duplicate text, and (by default) excludes
    principals appearing in the gold standard evaluation set.

    Set exclude_gold=False only for diagnostics where you explicitly
    want the full corpus -- never for training or anchor sampling.
    """
    path = Path(path)
    df = pd.read_csv(path, low_memory=False)

    # apply score mapping; Int64 preserves NaN for unscored rows
    df["score"] = pd.to_numeric(df[score_col], errors="coerce").map(SCORE_MAP)
    df["score"] = df["score"].astype("Int64")

    # exclude gold standard principals before any other processing so
    # all downstream counts reflect the true training corpus
    if exclude_gold:
        gold_ids = load_gold_person_ids(gold_path)
        if gold_ids:
            before = len(df)
            df = df[~df["PersonId"].isin(gold_ids)].copy()
            print(f"Excluded {before - len(df):,} rows belonging to "
                  f"{len(gold_ids)} gold standard PersonIds")

    # token count on raw text
    df["token_count"] = df[text_col].fillna("").apply(lambda x: len(x.split()))

    # filter: must have text above min length
    df = df[df["token_count"] >= min_tokens].copy()

    # deduplicate by text only -- the same text scored under different
    # elements is informative and both rows are kept
    if deduplicate:
        df = df.drop_duplicates(subset=[text_col]).copy()

    # element-prefixed text used for BIPDomainSFT training
    df["text_with_prefix"] = df[element_col] + ": " + df[text_col].fillna("")

    df = df.reset_index(drop=True)
    return df


def for_style_learner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns all usable BIPs regardless of score.
    Used for BIPDomainSFT causal LM fine-tuning.
    Scores are not required -- we are learning style, not labels.
    """
    return df[["text_with_prefix", "token_count", "Element_numberX"]].copy()


def for_anchor_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns only scored BIPs with valid rubric scores.
    Used for anchor sampling during generator SFT and PPO.
    Keeps both raw Text (shown as the anchor in prompts) and the
    prefixed version.
    """
    scored = df[df["score"].isin(VALID_SCORES)].copy()
    return scored[["text_with_prefix", "Text", "score",
                   "Element_numberX", "token_count", "PersonId"]].copy()


def load_gold(
    path: str | Path = GOLD_PATH,
    text_col: str = "Text",
    element_col: str = "Element_numberX",
    score_col: str = "Scaled_Annotator_Rating",
    drop_unscoreable: bool = True,
) -> pd.DataFrame:
    """
    Loads the gold standard evaluation set.

    Scoring column choice: Scaled_Annotator_Rating is used, not
    Annotator_Rating. Annotators worked on a raw 7-point scale (0-6)
    which was then normalized per-annotator onto the supervisor's 0-4
    scale. The crosstab between the two columns is non-deterministic
    (raw 3 maps to scaled 2, 3, and 4 in different rows), confirming
    per-annotator normalization rather than a global formula. Only
    the scaled column is directly comparable to Supervisor_Score_x
    and to model predictions, so it is the evaluation ground truth.

    Rows with Scaled_Annotator_Rating == -1 are dropped by default;
    -1 falls outside the 0-4 scale and indicates an unscoreable
    response.
    """
    path = Path(path)
    df = pd.read_csv(path, encoding=GOLD_ENCODING)

    if drop_unscoreable:
        before = len(df)
        df = df[df[score_col] >= 0].copy()
        if before != len(df):
            print(f"Dropped {before - len(df)} unscoreable gold rows "
                  f"({score_col} == -1)")

    # same rubric mapping as the training corpus so gold labels and
    # model predictions live in the same 0/2/4 space
    df["score"] = pd.to_numeric(df[score_col], errors="coerce").map(SCORE_MAP)
    df["score"] = df["score"].astype("Int64")

    df = df[df["score"].isin(VALID_SCORES)].copy()

    df["token_count"] = df[text_col].fillna("").apply(lambda x: len(x.split()))
    df["text_with_prefix"] = df[element_col] + ": " + df[text_col].fillna("")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/bips.csv"

    print("=" * 60)
    print("TRAINING CORPUS (gold principals excluded)")
    print("=" * 60)
    df = load(path)
    print(f"Loaded:        {len(df):,} rows after filtering")

    sl = for_style_learner(df)
    print(f"Style learner: {len(sl):,} BIPs")

    ap = for_anchor_pool(df)
    print(f"Anchor pool:   {len(ap):,} BIPs")
    print("\nScore distribution in anchor pool:")
    print(ap["score"].value_counts().sort_index())
    print("\nElement distribution in anchor pool:")
    print(ap["Element_numberX"].value_counts().sort_index())

    print()
    print("=" * 60)
    print("GOLD STANDARD EVALUATION SET")
    print("=" * 60)
    try:
        gold = load_gold()
        print(f"Gold rows:     {len(gold):,}")
        print(f"Unique principals: {gold['PersonId'].nunique()}")
        print("\nScore distribution:")
        print(gold["score"].value_counts().sort_index())
        print("\nElement distribution:")
        print(gold["Element_numberX"].value_counts().sort_index())

        overlap = set(df["PersonId"]) & set(gold["PersonId"])
        print(f"\nLEAKAGE CHECK -- overlap with training corpus: "
              f"{len(overlap)}  (MUST be 0)")
    except FileNotFoundError:
        print("Gold file not present; skipping.")