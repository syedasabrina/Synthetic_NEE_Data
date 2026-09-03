from __future__ import annotations

import torch
import numpy as np
from pathlib import Path

from transformers import Gemma4ForConditionalGeneration, AutoTokenizer
from peft import PeftModel


def build_generation_prompt(
    element: str,
    target_score: int,
    rubric_text: str,
    anchor_text: str,
) -> str:
    """
    Generation prompt. Unlike the SFT warmup, this includes the target
    score explicitly. Score conditioning is introduced here and
    reinforced through reward-based selection, never through supervised
    labels on noisy supervisor scores.
    """
    return f"""You are a school principal writing a Building Improvement Plan.

Element: {element}
Target score: {target_score}
Rubric criteria for this score: {rubric_text}

Reference example on a similar topic:
{anchor_text}

Generate a BIP response for this element that earns a score of
{target_score} according to the rubric criteria above. Write in your
own words, addressing a similar theme to the reference:
"""


class CandidateSampler:
    """
    Samples N candidate BIPs per prompt from the current generator.

    The generation half of best-of-n: produce many candidates, let the
    reward models rank them, keep the best. The generator loads from a
    LoRA checkpoint that advances each round.
    """

    def __init__(
        self,
        base_model_name: str = "google/gemma-4-E4B-it",
        adapter_path: str = "models/GeneratorSFT",
        device: str = "cuda",
        max_new_tokens: int = 320,
        temperature: float = 0.9,
        top_p: float = 0.95,
    ):
        """
        temperature 0.9 and top_p 0.95 are deliberately high. Best-of-n
        depends on candidate diversity; sampling N near-identical
        completions wastes the budget, since selecting the best of
        eight copies of the same text gains nothing.
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"Loading generator: {base_model_name} + {adapter_path} "
              f"on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = Gemma4ForConditionalGeneration.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map=device,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        print("CandidateSampler ready.")

    @torch.no_grad()
    def sample(
        self,
        prompt: str,
        n: int = 8,
        batch_size: int = 4,
    ) -> list[str]:
        """
        Returns n candidate completions for a single prompt, generated
        in sub-batches via num_return_sequences so a large n does not
        blow up memory on long prompts.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[1]
        candidates = []

        remaining = n
        while remaining > 0:
            k = min(batch_size, remaining)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=k,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            for seq in outputs:
                text = self.tokenizer.decode(
                    seq[prompt_len:], skip_special_tokens=True
                ).strip()
                candidates.append(text)
            remaining -= k

        return candidates


class ElementCycler:
    """
    Cycles through all seven elements in shuffled order, reshuffling on
    exhaustion. Prevents element drift when sampling many prompts,
    which matters because per-element anchor counts are uneven.
    """

    def __init__(self, elements: list[str], rng: np.random.Generator):
        self.elements = list(elements)
        self.rng = rng
        self.rng.shuffle(self.elements)
        self.idx = 0

    def next(self) -> str:
        if self.idx >= len(self.elements):
            self.rng.shuffle(self.elements)
            self.idx = 0
        e = self.elements[self.idx]
        self.idx += 1
        return e


def sample_prompt_spec(
    anchor_df,
    rubric_class,
    element_cycler,
    rng: np.random.Generator,
    score_weights: dict[int, float] | None = None,
) -> dict:
    """
    Draws one (element, target_score, anchor) spec.

    Elements come from a cycler for even coverage. Scores are drawn by
    weight, oversampling 2 and 4 because score-0 anchors are scarce
    across every element (31 total, roughly four per element).
    """
    score_weights = score_weights or {0: 0.1, 2: 0.35, 4: 0.55}
    scores = list(score_weights.keys())
    weights = list(score_weights.values())

    element = element_cycler.next()
    target_score = int(rng.choice(scores, p=weights))

    pool = anchor_df[
        (anchor_df["Element_numberX"] == element)
        & (anchor_df["score"] == target_score)
    ]
    if len(pool) == 0:
        pool = anchor_df[
            (anchor_df["Element_numberX"] == element)
            & (anchor_df["score"] == 4)
        ]
    if len(pool) == 0:
        pool = anchor_df[anchor_df["score"] == 4]

    row = pool.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]

    return {
        "element": element,
        "target_score": target_score,
        "anchor_text": row["Text"],
        "rubric_text": rubric_class.RUBRIC[element][target_score],
    }