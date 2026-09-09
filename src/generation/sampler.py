from __future__ import annotations

import torch
import numpy as np
from pathlib import Path

from transformers import Gemma4ForConditionalGeneration, AutoTokenizer
from peft import PeftModel


# What each element actually asks the principal to describe.
#
# Elements 1, 2, and 7 are taken verbatim from the NEE BIP Process
# Organizer form. Elements 3 through 6 are derived from the rubric
# criteria, since the corresponding form pages were not available --
# they state the same requirement the rubric scores against, but are
# not the official prompt wording.
#
# Added because a keyword audit of round 2 found only 63% of
# completions addressed their assigned element, with Element 3 at 39%.
# The prompt previously supplied only "Element3" plus a rubric line
# describing how WELL to do something, never stating WHAT to do, so
# the anchor BIP became the strongest signal and the model drifted
# toward whatever topic the anchor happened to cover.
ELEMENT_QUESTIONS = {
    "Element1":
        "Describe your leadership involvement in the development of "
        "the BIP. Provide evidence, such as the schedule and agendas "
        "of BIP meetings, summary of BIP meetings, and school "
        "performance data reports and resources prepared for use by "
        "the BIP team.",
    "Element2":
        "What collaborative processes were used to address the shared "
        "needs of the building? Who participated in that "
        "collaboration? Provide evidence, such as the BIP team roster "
        "of school stakeholders, meeting agendas revealing degree of "
        "participation and input of all members, and how the BIP was "
        "shared with all building staff and input taken back to the "
        "BIP team.",
    "Element3":
        "Describe how the BIP objectives align to the district's "
        "Comprehensive School Improvement Plan (CSIP) goals. State "
        "which CSIP goal each BIP objective supports.",
    "Element4":
        "State the measurable objectives of the BIP and provide the "
        "baseline data for each one. Include the specific numbers or "
        "percentages the objectives are measured against.",
    "Element5":
        "Describe the implementation strategies chosen for each "
        "objective and cite the credible research sources that "
        "support them.",
    "Element6":
        "Describe how progress toward the BIP objectives was "
        "monitored during the year, and what corrective actions were "
        "taken when the data showed objectives were not being met.",
    "Element7":
        "Describe how and when BIP results were shared with building "
        "staff, the BIP team, and school district administration. "
        "Provide evidence, such as follow-up meeting minutes and "
        "presentations, faculty meeting agendas, building-level data "
        "walls, or BIP reports to district administration.",
}


def build_generation_prompt(
    element: str,
    target_score: int,
    rubric_text: str,
    anchor_text: str,
) -> str:
    """
    Generation prompt. Unlike the SFT warmup, this includes the target
    score explicitly. Score conditioning is introduced here and
    reinforced through reward-based selection, never through
    supervised labels on noisy supervisor scores.

    The element question is stated before the anchor so the task is
    defined before the model sees an example that may cover different
    ground, and the anchor is explicitly framed as a style reference
    rather than a content template.
    """
    question = ELEMENT_QUESTIONS.get(element, "")

    return f"""You are a school principal writing a Building Improvement Plan.

{element} asks: {question}

Target score: {target_score}
Rubric criteria for this score: {rubric_text}

Here is a real BIP response, shown only as an example of how
principals write. Its topic may differ from what you need to write:

{anchor_text}

Now write a response to the {element} question above. Address the
question directly, at the quality level the rubric criteria describe
for a score of {target_score}. Write in your own words:
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
        max_new_tokens: int = 400,
        temperature: float = 0.9,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 4,
    ):
        """
        max_new_tokens 400 sits between two observed failure modes. At
        320, round 1 completions were all clipped at ~300 words with
        the maximum pinned across every score level, so the model was
        trained on systematically truncated text. At 512, round 2
        produced 358-word completions with a distinct-bigram ratio of
        0.559 and 74% of accepted candidates below 0.7 -- the model
        filled the extra budget by looping. Real BIPs have p50 at 150
        tokens and p90 at 469, so most responses finish before 400.

        repetition_penalty and no_repeat_ngram_size stop looping at the
        decoder rather than penalizing it afterward. The round 2 worst
        case repeated one sentence eight times to fill its budget;
        no_repeat_ngram_size=4 makes that structurally impossible. With
        these set, accepted-set repetition went from 0.559 to 0.994.

        temperature 0.9 and top_p 0.95 stay high on purpose. Best-of-n
        depends on candidate diversity; sampling N near-identical
        completions wastes the budget.
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

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

        print(f"CandidateSampler ready (max_new_tokens={max_new_tokens}, "
              f"no_repeat_ngram={no_repeat_ngram_size}).")

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
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
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