from __future__ import annotations

import re
import torch
from pathlib import Path
from transformers import Gemma4ForConditionalGeneration, AutoTokenizer


class RubricReward:
    """
    Rubric alignment reward using Gemma 4 E4B as a frozen few-shot judge.

    The judge is BLIND to the target score: it sees all three rubric
    levels and picks one independently. The reward is computed outside
    the model by comparing that blind prediction to the target.

    Calibration on the 171-row gold set: 171/171 parsed, exact
    agreement 0.661, adjacent 0.988, mean signed deviation +0.000.

    Frozen at all times -- never updated during training.
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
        use_chat_template: bool = True,
        batch_size: int = 8,
    ):
        """
        device: accepts "cuda:2" and similar so the judge can sit on a
        GPU separate from the model being trained. Three large models
        on one device caused OOM; the interface here is text in, float
        out, so nothing crosses device boundaries as tensors.

        use_chat_template: Gemma 4 E4B is instruction tuned. Feeding a
        raw string outside its chat format caused 93 of 171 gold BIPs
        to return empty output and skewed the predictions that did
        parse. Applying the template fixed both the parse rate and a
        -0.949 score bias.

        batch_size: judge calls dominated round-1 wall time at roughly
        70 s per prompt, largely because eight candidates were scored
        one at a time.
        """
        self.device = device
        self.few_shot_examples = few_shot_examples or {}
        self.max_new_tokens = max_new_tokens
        self.max_example_chars = max_example_chars
        self.use_chat_template = use_chat_template
        self.batch_size = batch_size

        print(f"Loading Gemma 4 rubric reward model: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # decoder-only batched generation requires left padding, or
        # shorter sequences end up with pad tokens sitting between the
        # prompt and the generated continuation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        if self.use_chat_template and not getattr(
            self.tokenizer, "chat_template", None
        ):
            print("WARNING: tokenizer has no chat_template; "
                  "falling back to raw prompting.")
            self.use_chat_template = False

        print(f"RubricReward ready (chat_template={self.use_chat_template}, "
              f"batch_size={batch_size}).")

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
        Few-shot demonstrations from the gold standard set, keyed by
        (element, score). Frozen in-context demonstrations only; no
        gradients, no training. The gold set stays held out for final
        assessor evaluation.
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

Which score does this response earn? Reply with only the number 0, 2, or 4."""

        return prompt

    def _format(self, prompt: str) -> str:
        if self.use_chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt + "\nScore:"

    def _parse_score(self, generated: str) -> int | None:
        """
        First 0, 2, or 4 anywhere in the output. Regex rather than
        whitespace token matching, since instruction tuned models
        format short answers as "Score: 2." or "**2**".
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
        Blind score prediction, batched. No knowledge of any target
        score. return_raw also returns decoded output per candidate.
        """
        texts = [
            self._format(self._build_prompt(e, c))
            for c, e in zip(candidates, elements)
        ]

        predictions, raw_outputs = [], []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=3072,
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # left padding means every sequence in the batch shares the
            # same prompt length, so one slice point covers all of them
            prompt_len = inputs["input_ids"].shape[1]
            for seq in outputs:
                gen = self.tokenizer.decode(
                    seq[prompt_len:], skip_special_tokens=True
                ).strip()
                predictions.append(self._parse_score(gen))
                raw_outputs.append(gen)

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
        Rubric alignment reward:
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