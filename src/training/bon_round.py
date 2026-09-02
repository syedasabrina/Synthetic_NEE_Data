from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from src.rewards.authenticity_reward import AuthenticityReward
from src.rewards.rubric_reward import RubricReward
from src.generation.sampler import (
    CandidateSampler,
    ElementCycler,
    build_generation_prompt,
    sample_prompt_spec,
)


class BestOfNRound:
    """
    One round of iterative rejection-sampling fine-tuning.

    For each prompt: generate N candidates, score all N with both
    reward models, keep the top k. Accepted candidates become the
    training set for the next generator checkpoint.

    Why this rather than policy-gradient optimization: the reward is a
    Python function that decodes text and calls two separate language
    models. Selection only requires that the reward RANK candidates
    correctly within a prompt, so the judge's known harshness (mean
    signed deviation -0.47 against expert scores) cancels out. A
    gradient method would propagate that bias into the policy, since
    reward magnitude enters the update directly.
    """

    def __init__(
        self,
        anchor_df,
        gold_df,
        generator_adapter: str,
        output_dir: str,
        n_candidates: int = 8,
        keep_top_k: int = 2,
        alpha: float = 0.5,
        min_reward: float = 0.0,
        anchor_similarity_threshold: float = 0.85,
        seed: int = 42,
        device: str = "cuda",
    ):
        """
        n_candidates / keep_top_k: 8 and 2 gives a 4:1 rejection ratio.
        Higher N raises quality but costs inference linearly.

        min_reward: absolute floor on combined reward. Without it, a
        prompt where all N candidates are poor still contributes its
        top 2 to training, teaching the generator from the best of a
        bad batch. Set above 0 to drop those prompts entirely.

        alpha: weight on authenticity vs rubric alignment in the
        combined reward.
        """
        self.anchor_df = anchor_df
        self.output_dir = Path(output_dir)
        self.n_candidates = n_candidates
        self.keep_top_k = keep_top_k
        self.alpha = alpha
        self.min_reward = min_reward
        self.anchor_similarity_threshold = anchor_similarity_threshold
        self.rng = np.random.default_rng(seed)

        os.makedirs(self.output_dir, exist_ok=True)

        print("Loading reward models...")
        self.auth = AuthenticityReward(device=device)

        few_shot = RubricReward.build_few_shot_examples(gold_df, max_examples=1)
        self.rubric = RubricReward(
            device=device, few_shot_examples=few_shot
        )

        print("Loading generator...")
        self.sampler = CandidateSampler(
            adapter_path=generator_adapter, device=device
        )

        self.element_cycler = ElementCycler(
            anchor_df["Element_numberX"].unique().tolist(), self.rng
        )

    def _anchor_overlap(self, candidate: str, anchor: str) -> float:
        """
        Word-level Jaccard overlap between candidate and anchor.

        Guards against the generator copying its reference rather than
        writing a new response. A lightweight proxy for embedding
        similarity, chosen to avoid loading a third model inside the
        generation loop.
        """
        a = set(candidate.lower().split())
        b = set(anchor.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def run(
        self,
        n_prompts: int = 500,
        log_every: int = 25,
    ) -> pd.DataFrame:
        """
        Generates and scores candidates across n_prompts, returning the
        accepted set as a DataFrame.
        """
        accepted_rows = []
        all_scores = []
        rejected_leakage = 0
        rejected_low = 0

        for i in range(n_prompts):
            spec = sample_prompt_spec(
                self.anchor_df, self.rubric, self.element_cycler, self.rng
            )

            prompt = build_generation_prompt(
                spec["element"],
                spec["target_score"],
                spec["rubric_text"],
                spec["anchor_text"],
            )

            candidates = self.sampler.sample(prompt, n=self.n_candidates)

            # drop empty generations before scoring
            candidates = [c for c in candidates if len(c.split()) >= 10]
            if not candidates:
                continue

            # anchor leakage filter
            keep = []
            for c in candidates:
                if self._anchor_overlap(c, spec["anchor_text"]) \
                        > self.anchor_similarity_threshold:
                    rejected_leakage += 1
                else:
                    keep.append(c)
            if not keep:
                continue
            candidates = keep

            elements = [spec["element"]] * len(candidates)
            targets = [spec["target_score"]] * len(candidates)

            auth_scores = self.auth.score_batch(
                candidates, batch_size=4, normalize=True
            )
            rubric_scores = self.rubric.score(candidates, elements, targets)

            combined = [
                self.alpha * a + (1 - self.alpha) * r
                for a, r in zip(auth_scores, rubric_scores)
            ]

            order = np.argsort(combined)[::-1]
            kept = 0
            for idx in order:
                if kept >= self.keep_top_k:
                    break
                if combined[idx] < self.min_reward:
                    rejected_low += 1
                    continue
                accepted_rows.append({
                    "element": spec["element"],
                    "target_score": spec["target_score"],
                    "rubric_text": spec["rubric_text"],
                    "anchor_text": spec["anchor_text"],
                    "prompt": prompt,
                    "completion": candidates[idx],
                    "auth_reward": auth_scores[idx],
                    "rubric_reward": rubric_scores[idx],
                    "combined_reward": combined[idx],
                })
                kept += 1

            all_scores.extend(combined)

            if (i + 1) % log_every == 0:
                mean_c = float(np.mean(all_scores)) if all_scores else 0.0
                print(f"prompt {i+1}/{n_prompts}  "
                      f"accepted={len(accepted_rows)}  "
                      f"mean_combined={mean_c:.4f}  "
                      f"leakage_rejects={rejected_leakage}  "
                      f"low_rejects={rejected_low}")

        df = pd.DataFrame(accepted_rows)

        stats = {
            "n_prompts": n_prompts,
            "n_accepted": len(df),
            "n_candidates_scored": len(all_scores),
            "mean_combined_all": float(np.mean(all_scores)) if all_scores else 0.0,
            "mean_combined_accepted": float(df["combined_reward"].mean())
                if len(df) else 0.0,
            "mean_auth_accepted": float(df["auth_reward"].mean())
                if len(df) else 0.0,
            "mean_rubric_accepted": float(df["rubric_reward"].mean())
                if len(df) else 0.0,
            "rejected_leakage": rejected_leakage,
            "rejected_low_reward": rejected_low,
        }
        if len(df):
            stats["accepted_by_score"] = (
                df["target_score"].value_counts().to_dict()
            )
            stats["accepted_by_element"] = (
                df["element"].value_counts().to_dict()
            )

        with open(self.output_dir / "round_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

        df.to_json(
            self.output_dir / "accepted.jsonl",
            orient="records", lines=True,
        )

        print("\nRound complete.")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        return df


def build_retrain_dataset(
    accepted_df: pd.DataFrame,
    tokenizer,
    max_length: int = 1024,
) -> Dataset:
    """
    Converts accepted candidates into a tokenized dataset for the next
    generator fine-tune. Prompt tokens are masked with -100 so loss is
    computed only on the completion.

    Mirrors the masking in generator_sft.build_sft_dataset, so each
    round is the same training procedure applied to progressively
    better data.
    """
    hf = Dataset.from_dict({
        "prompt": accepted_df["prompt"].tolist(),
        "completion": accepted_df["completion"].tolist(),
    })

    def tokenize(batch):
        input_ids_b, labels_b, attn_b = [], [], []

        for prompt, completion in zip(batch["prompt"], batch["completion"]):
            prompt_ids = tokenizer(
                prompt, add_special_tokens=True,
                truncation=True, max_length=max_length,
            )["input_ids"]

            full_ids = tokenizer(
                prompt + completion, add_special_tokens=True,
                truncation=True, max_length=max_length,
            )["input_ids"]

            prompt_len = min(len(prompt_ids), len(full_ids))

            pad_len = max_length - len(full_ids)
            input_ids = full_ids + [tokenizer.pad_token_id] * pad_len
            attention_mask = [1] * len(full_ids) + [0] * pad_len

            labels = [-100] * prompt_len + full_ids[prompt_len:]
            labels = labels[:max_length]
            labels = labels + [-100] * (max_length - len(labels))

            input_ids_b.append(input_ids)
            labels_b.append(labels)
            attn_b.append(attention_mask)

        return {
            "input_ids": input_ids_b,
            "labels": labels_b,
            "attention_mask": attn_b,
        }

    return hf.map(
        tokenize, batched=True,
        remove_columns=["prompt", "completion"],
        load_from_cache_file=False,
    )