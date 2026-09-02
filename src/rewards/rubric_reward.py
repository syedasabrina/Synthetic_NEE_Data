from __future__ import annotations

import re
import torch
from pathlib import Path
from transformers import Gemma4ForConditionalGeneration, AutoTokenizer


class RubricReward:
    """
    Computes a rubric alignment reward for candidate synthetic BIPs
    using Gemma 4 E4B as a frozen few-shot rubric judge.

    The judge is BLIND to the target score: it sees all three rubric
    levels and picks one independently. The reward is computed outside
    the model by comparing that blind prediction to the target. Telling
    the judge the target and asking whether it agrees would bias an
    instruction-tuned model toward agreement regardless of quality.

    This model is always frozen -- it is never updated during PPO.
    """

    RUBRIC = {
        "Element1": {
            0: "The principal describes little or no leadership involvement in BIP development.",
            2: "The principal describes vague or minimal leadership involvement in BIP development.",
            4: "The principal describes extensive leadership involvement in BIP development.",
        },
        "Element2": {
            0: "The principal describes a top-down process or the BIP was written by a single author with little effort to actively involve other key stakeholders.",
            2: "The input describes a vague or minimal collaborative process that involves limited stakeholders.",
            4: "The input describes a fully collaborative process that involves a wide variety of building-level stakeholders.",
        },
        "Element3": {
            0: "The principal does not align the BIP objectives to CSIP goals.",
            2: "The principal vaguely or incompletely aligns the BIP objectives to CSIP goals.",
            4: "The principal fully and clearly aligns the BIP objectives to CSIP goals.",
        },
        "Element4": {
            0: "The principal provides no baseline data.",
            2: "The principal provides vague or limited baseline data.",
            4: "The principal provides clear and compelling baseline data for all objectives.",
        },
        "Element5": {
            0: "The principal describes no research-based implementation strategies and sources for each objective.",
            2: "The principal describes some research-based implementation strategies and sources for each objective.",
            4: "The principal fully describes research-based implementation strategies and sources for each objective.",
        },
        "Element6": {
            0: "The principal provides no description of the monitoring process or corrective actions.",
            2: "The principal provides a limited description of the monitoring process or corrective actions.",
            4: "The principal provides an ample and clear description of the monitoring process, and corrective actions if needed.",
        },
        "Element7": {
            0: "The principal provides no description of how BIP results were shared.",
            2: "The principal provides a limited description of how the BIP results were regularly shared with school staff, BIP team, and school district administration.",
            4: "The principal provides an ample and clear description of how the BIP results were regularly shared with school staff, BIP team, and school district administration.",
        },
    }

    def __init__(
        self,
        model_name: str = "google/gemma-4-E4B-it",
        device: str = "cuda",
        few_shot_examples: dict | None = None,
        max_new_tokens: int = 24,
        max_example_chars: int = 900,
    ):
        """
        max_new_tokens defaults to 24 rather than 5. Calibration with a
        5-token budget produced unparseable output on 93 of 171 gold
        BIPs, because the model preambles before emitting a digit and
        gets truncated mid-sentence. 24 tokens reaches the number
        without inviting a full explanation.

        max_example_chars truncates few-shot demonstrations. Real BIPs
        run to hundreds of tokens each; three full examples plus the
        candidate can crowd the instruction out of context, which is a
        second cause of unparseable output.
        """
        self.device = device
        self.few_shot_examples = few_shot_examples or {}
        self.max_new_tokens = max_new_tokens
        self.max_example_chars = max_example_chars

        print(f"Loading Gemma 4 rubric reward model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        print("RubricReward ready.")

    @classmethod
    def build_few_shot_examples(
        cls,
        gold_df,
        text_col: str = "Text",
        element_col: str = "Element_numberX",
        score_col: str = "score",
        max_examples: int = 1,
    ) -> dict:
        """
        Builds few-shot demonstrations from the gold standard set,
        keyed by (element, score).

        max_examples defaults to 1 per cell: the prompt already shows
        three score levels, so three examples each would put nine BIPs
        in context before the candidate appears.

        Used only as frozen in-context demonstrations for a frozen
        model. No gradients, no training. The gold set remains held out
        for final assessor evaluation.
        """
        examples = {}
        for (element, score), group in gold_df.groupby([element_col, score_col]):
            examples[(element, int(score))] = group[text_col].tolist()[:max_examples]
        return examples

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_example_chars:
            return text
        return text[: self.max_example_chars].rsplit(" ", 1)[0] + " ..."

    def _build_prompt(self, element: str, candidate: str) -> str:
        """
        Blind scoring prompt. All three rubric levels shown; the target
        score is never mentioned.
        """
        criteria_block = "\n".join(
            f"Score {score}: {text}"
            for score, text in sorted(self.RUBRIC[element].items())
        )

        prompt = f"""You are an expert evaluator of school principal Building Improvement Plans (BIPs).

Score the BIP response below for {element} using the NEE rubric. Judge only on the content of the response.

Rubric criteria for {element}:
{criteria_block}
"""

        example_block = ""
        for score in (0, 2, 4):
            examples = self.few_shot_examples.get((element, score), [])
            if examples:
                example_block += (
                    f"\nExample of a response scoring {score}:\n"
                    f"{self._truncate(examples[0])}\n"
                )
        if example_block:
            prompt += "\nReference examples at each score level:\n" + example_block

        prompt += f"""
BIP response to score:
{candidate}

Which score does this response earn? Answer with a single number: 0, 2, or 4.
Score:"""

        return prompt

    def _parse_score(self, generated: str) -> int | None:
        """
        Extracts the first 0, 2, or 4 appearing anywhere in the output.

        The previous implementation split on whitespace and required an
        exact token match, so output like "Score: 2." or "**2**" or
        "I would say 2" failed to parse. Regex over the raw string
        matches how instruction-tuned models actually format short
        answers.
        """
        m = re.search(r"[024]", generated)
        return int(m.group(0)) if m else None

    @torch.no_grad()
    def predict(
        self,
        candidates: list[str],
        elements: list[str],
        return_raw: bool = False,
    ):
        """
        Blind score prediction, with no knowledge of any target score.

        return_raw=True also returns the decoded model output per
        candidate, so unparseable cases can be inspected rather than
        silently counted.
        """
        predictions, raw_outputs = [], []

        for candidate, element in zip(candidates, elements):
            prompt = self._build_prompt(element, candidate)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3072,
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            predictions.append(self._parse_score(generated))
            raw_outputs.append(generated)

        if return_raw:
            return predictions, raw_outputs
        return predictions

    def score(
        self,
        candidates: list[str],
        elements: list[str],
        target_scores: list[int],
    ) -> list[float]:
        """
        Rubric alignment reward per candidate:

            1.0  predicted score matches target exactly
            0.5  predicted score is one rubric level away
            0.0  far off, or unparseable
        """
        predicted = self.predict(candidates, elements)
        return [
            self._compute_reward(p, t)
            for p, t in zip(predicted, target_scores)
        ]

    def _compute_reward(self, predicted: int | None, target: int) -> float:
        if predicted is None:
            return 0.0
        if predicted == target:
            return 1.0
        if abs(predicted - target) == 2:
            return 0.5
        return 0.0