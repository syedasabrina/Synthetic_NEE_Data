from __future__ import annotations

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration


# What each element actually asks the principal to describe.
#
# Elements 1, 2, and 7 are taken verbatim from the NEE BIP Process
# Organizer form. Elements 3 through 6 are derived from the rubric
# criteria, since the corresponding form pages were not available. They
# state the same requirement the rubric scores against, but are not the
# official prompt wording.
#
# Added because a keyword audit of round 2 found only 63% of completions
# addressed their assigned element, with Element 3 at 39%.
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


# Same wording as the prompt used in rounds 1 to 4, so changes in output
# can be attributed to the decoder, EOS and gating fixes and not to the
# prompt.
_GEN_TEMPLATE = """You are a school principal writing a Building Improvement Plan.

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


def _render(element, target_score, rubric_text, anchor_text) -> str:
    return _GEN_TEMPLATE.format(
        element=element,
        question=ELEMENT_QUESTIONS.get(element, ""),
        target_score=target_score,
        rubric_text=rubric_text,
        anchor_text=anchor_text,
    )


def _n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def fit_anchor(tokenizer, anchor_text: str, budget: int) -> str:
    """Cuts the anchor to at most `budget` tokens, at a word boundary."""
    ids = tokenizer(anchor_text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return anchor_text
    cut = tokenizer.decode(ids[:max(budget, 0)], skip_special_tokens=True)
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + " ..."


def build_generation_prompt(
    element: str,
    target_score: int,
    rubric_text: str,
    anchor_text: str,
    tokenizer=None,
    max_prompt_tokens: int = 640,
) -> str:
    """
    Generation prompt. Score conditioning enters here and is reinforced
    through selection, never through supervised labels on supervisor scores.

    With a tokenizer, the anchor is cut so the whole prompt fits in
    max_prompt_tokens. Before this, about 5% of prompts exceeded the
    sampler's 1,024-token limit and right-truncation removed the final
    instruction line. The default of 640 leaves room for a completion of
    roughly 600 tokens inside a 1,280-token training sequence.
    """
    prompt = _render(element, target_score, rubric_text, anchor_text)
    if tokenizer is None or _n_tokens(tokenizer, prompt) <= max_prompt_tokens:
        return prompt

    overhead = _n_tokens(tokenizer, _render(element, target_score, rubric_text, ""))
    budget = max_prompt_tokens - overhead - 4
    if budget < 32:
        raise ValueError(
            f"Prompt overhead is {overhead} tokens; max_prompt_tokens="
            f"{max_prompt_tokens} leaves no room for an anchor."
        )

    # decode/encode round trips can shift the count by a few tokens, so
    # shrink until it fits
    for _ in range(6):
        anchor = fit_anchor(tokenizer, anchor_text, budget)
        prompt = _render(element, target_score, rubric_text, anchor)
        n = _n_tokens(tokenizer, prompt)
        if n <= max_prompt_tokens:
            return prompt
        budget -= (n - max_prompt_tokens) + 4
    raise ValueError(f"Could not fit prompt into {max_prompt_tokens} tokens.")


class CandidateSampler:
    """
    Samples N candidate BIPs per prompt from the current generator.

    Defaults changed from rounds 1 to 4:

    repetition_penalty 1.0 and no_repeat_ngram_size 0 (both off). HF
    applies both over the prompt as well as the completion, and the prompt
    holds the rubric, the element question and a real BIP, so the decoder
    was banned from reusing ordinary words and 4-grams that appear in its
    own instructions. Looping is now handled by learned EOS plus the
    quality gate. The decoding probe (scripts/probe_decoding.py) is what
    decides whether any penalty is worth restoring.

    max_new_tokens 512 (was 400). corpus token_count is whitespace words,
    so the real p90 of 469 is words, about 600 model tokens. A 400-token
    cap clipped a large share of real-length completions. With EOS learned
    most completions end well before the cap.
    """

    def __init__(
        self,
        base_model_name: str = "google/gemma-4-E4B-it",
        adapter_path: str = "models/GeneratorSFT",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.9,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        max_input_tokens: int = 1024,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.max_input_tokens = max_input_tokens

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

        # Stop on the EOS id used in training AND on whatever the model's
        # generation_config stops on, so a mismatch between the two cannot
        # silently disable stopping.
        eos = set()
        gc = getattr(self.model, "generation_config", None)
        cfg_eos = getattr(gc, "eos_token_id", None)
        if isinstance(cfg_eos, int):
            eos.add(cfg_eos)
        elif cfg_eos:
            eos.update(int(x) for x in cfg_eos)
        if self.tokenizer.eos_token_id is not None:
            eos.add(int(self.tokenizer.eos_token_id))
        self.eos_ids = sorted(eos)
        self.pad_id = (self.tokenizer.pad_token_id
                       if self.tokenizer.pad_token_id is not None
                       else self.tokenizer.eos_token_id)

        print(f"CandidateSampler ready (max_new_tokens={max_new_tokens}, "
              f"rep_penalty={repetition_penalty}, "
              f"no_repeat_ngram={no_repeat_ngram_size}, "
              f"stop ids={self.eos_ids}).")

    @torch.no_grad()
    def sample_detailed(
        self,
        prompt: str,
        n: int = 8,
        batch_size: int = 4,
    ) -> list[dict]:
        """
        Returns n dicts {text, n_tokens, stopped}. `stopped` is True when
        the model emitted a stop token before max_new_tokens, which is the
        truncation test the quality gate uses.
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=False,
        )
        prompt_len = inputs["input_ids"].shape[1]
        if prompt_len > self.max_input_tokens:
            raise ValueError(
                f"Prompt is {prompt_len} tokens (limit "
                f"{self.max_input_tokens}). Build it with "
                f"build_generation_prompt(..., tokenizer=...)."
            )
        inputs = inputs.to(self.device)

        out = []
        remaining = n
        while remaining > 0:
            k = min(batch_size, remaining)
            gen_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=k,
                eos_token_id=self.eos_ids,
                pad_token_id=self.pad_id,
            )
            if self.repetition_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = self.repetition_penalty
            if self.no_repeat_ngram_size:
                gen_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size

            outputs = self.model.generate(**inputs, **gen_kwargs)
            for seq in outputs:
                gen = seq[prompt_len:].tolist()
                stopped, length = False, len(gen)
                for idx, tok in enumerate(gen):
                    if tok in self.eos_ids:
                        stopped, length = True, idx
                        break
                text = self.tokenizer.decode(
                    gen[:length], skip_special_tokens=True
                ).strip()
                out.append({"text": text, "n_tokens": length,
                            "stopped": stopped})
            remaining -= k
        return out

    def sample(self, prompt: str, n: int = 8, batch_size: int = 4) -> list[str]:
        return [s["text"] for s in self.sample_detailed(prompt, n, batch_size)]


def element_for_index(elements: list[str], seed: int, i: int) -> str:
    """
    Deterministic element schedule: prompt i belongs to cycle i // E, and
    each cycle is a seeded shuffle of all elements. Replaces the stateful
    ElementCycler so a resumed run produces the same specs for the same
    prompt indices.
    """
    elements = sorted(elements)
    e = len(elements)
    perm = np.random.default_rng([seed, 10_000, i // e]).permutation(e)
    return elements[int(perm[i % e])]


def sample_prompt_spec(
    anchor_df,
    rubric_class,
    element: str,
    rng: np.random.Generator,
    score_weights: dict[int, float] | None = None,
) -> dict:
    """
    Draws one (element, target_score, anchor) spec for a given element.

    Scores are drawn by weight. If the (element, score) anchor cell is
    empty it falls back to a score-4 anchor; `anchor_fallback` records
    that so it can be analysed later.
    """
    score_weights = score_weights or {0: 0.1, 2: 0.35, 4: 0.55}
    scores = list(score_weights.keys())
    weights = list(score_weights.values())

    target_score = int(rng.choice(scores, p=weights))

    pool = anchor_df[
        (anchor_df["Element_numberX"] == element)
        & (anchor_df["score"] == target_score)
    ]
    fallback = False
    if len(pool) == 0:
        fallback = True
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
        "anchor_score": int(row["score"]),
        "anchor_fallback": fallback or int(row["score"]) != target_score,
        "anchor_person": str(row["PersonId"]) if "PersonId" in row else None,
        "rubric_text": rubric_class.RUBRIC[element][target_score],
    }
