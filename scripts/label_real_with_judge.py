#!/usr/bin/env python
"""
Labels the real scored BIPs with the blind judge.

This produces the training data for the control condition "judge_real":
real BIP text, labelled by the same judge that labels the synthetic data.
If an assessor trained on this does as well as one trained on the
synthetic dataset, the generated text adds nothing beyond running the
judge on real BIPs, and that is worth knowing before spending days on
generation.

The prompt is the same one the selection judge uses (the same rubric text
and the same 21 gold demonstrations, one per element and level). Real BIPs
can be much longer than generated candidates, so the judge's input limit
is raised to avoid cutting off the final instruction line.

Resumable: rows already in the output file are skipped, so a killed job
continues where it stopped.

    sbatch --partition=gpuq --qos=gpu --time=06:00:00 \
      --export=ALL,SCRIPT=scripts/label_real_with_judge.py,ARGS="--out data/derived/judge_labelled_real.jsonl" \
      scripts/train.slurm
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


def label_rows(judge, df: pd.DataFrame, out_path: Path, chunk: int = 200,
               log=print) -> int:
    """Labels every row of df not already in out_path. Returns rows written."""
    done = set()
    if out_path.exists():
        valid = []
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["row"])
                    valid.append(line if line.endswith("\n") else line + "\n")
                except (json.JSONDecodeError, KeyError):
                    continue  # partial last line from a killed job
        # rewrite without the partial line, or the next record would be
        # appended onto it and both would be lost
        with open(out_path, "w") as f:
            f.writelines(valid)
    todo = df[~df.index.isin(done)]
    log(f"Rows: {len(df)}  already labelled: {len(done)}  to do: {len(todo)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "a") as f:
        for start in range(0, len(todo), chunk):
            part = todo.iloc[start:start + chunk]
            preds, raws = judge.predict(
                part["Text"].tolist(), part["Element_numberX"].tolist(),
                return_raw=True,
            )
            for (idx, row), pred, raw in zip(part.iterrows(), preds, raws):
                f.write(json.dumps({
                    "row": int(idx),
                    "PersonId": str(row["PersonId"]),
                    "element": row["Element_numberX"],
                    "text": row["Text"],
                    "supervisor_score": int(row["score"]),
                    "judge_pred": pred,
                    "judge_raw": raw,
                }) + "\n")
            f.flush()
            written += len(part)
            log(f"  labelled {written}/{len(todo)}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/bips.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=8192,
                    help="judge input limit; real BIPs are long")
    ap.add_argument("--limit", type=int, default=None, help="label only the first N rows (test runs)")
    args = ap.parse_args()

    from src.data.corpus import for_anchor_pool, load, load_gold
    from src.rewards.rubric_reward import RubricReward

    df = for_anchor_pool(load(args.data)).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"Real scored BIPs: {len(df):,}")

    few_shot = RubricReward.build_few_shot_examples(load_gold(), max_examples=1)
    judge = RubricReward(
        model_name=args.judge_model, device="cuda",
        few_shot_examples=few_shot, batch_size=args.batch_size,
        max_length=args.max_length,
    )
    label_rows(judge, df, Path(args.out))

    lab = pd.read_json(args.out, lines=True)
    print("\nJudge labels on real BIPs:")
    print(lab["judge_pred"].value_counts(dropna=False).sort_index().to_string())
    print("\nSupervisor labels on the same rows:")
    print(lab["supervisor_score"].value_counts().sort_index().to_string())
    print(f"\nExact match judge vs supervisor: {(lab['judge_pred'] == lab['supervisor_score']).mean():.3f}")


if __name__ == "__main__":
    main()
