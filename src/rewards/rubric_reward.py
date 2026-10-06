from __future__ import annotations

import re
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Gemma4ForConditionalGeneration,
)


class RubricReward:
    """
    Rubric alignment reward using a frozen few-shot LLM judge.

    The judge is BLIND to the target score: it sees all three rubric
    levels and picks one independently. The reward is computed outside the
    model by comparing that blind prediction to the target.

    Previously reported on the 171-row gold set with Gemma 4 E4B: 171/171
    parsed, exact agreement 0.661, adjacent 0.988, signed deviation 0.000.
    Two things in that number need re-measuring (scripts/eval_judge_gold.py
    does both):

    1. Up to 21 of the 171 rows were few-shot examples inside the prompt,
       so agreement on them is not held out.
    2. The chat-template string may already contain the BOS token, in
       which case tokenizing it again added a second one. At load time this
       version checks whether the template output starts with the
       tokenizer's BOS (printed in the load log) and only then tokenizes
       with add_special_tokens=False. If it does not, tokenization is
       unchanged. Either way the judge's inputs can differ slightly from
       the run that produced 0.661, so that number is not directly
       comparable to new runs.

    Frozen at all times. Never updated during training.
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
        max_length: int = 3072,
        chat_template_kwargs: dict | None = None,
    ):
        """
        device: accepts "cuda:2" and similar so the judge can sit on a GPU
        separate from the model being trained.

        use_chat_template: instruction-tuned judges need their chat format.
        Raw strings gave empty output on 93 of 171 gold BIPs for Gemma 4.

        model_name: any causal LM. Gemma 4 loads through
        Gemma4ForConditionalGeneration, everything else through
        AutoModelForCausalLM. Loading support for the newer Qwen and
        Gemma checkpoints on transformers 5.16.1 is untested.
        """
        self.device = device
        self.few_shot_examples = few_shot_examples or {}
        self.max_new_tokens = max_new_tokens
        self.max_example_chars = max_example_chars
        self.use_chat_template = use_chat_template
        self.batch_size = batch_size
        self.max_length = max_length
        # e.g. {"enable_thinking": False} for Qwen models that think by default.
        # None keeps the prompt exactly as before, so the Gemma judge is unchanged.
        self.chat_template_kwargs = chat_template_kwargs or {}

        print(f"Loading rubric judge: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # decoder-only batched generation requires left padding, or shorter
        # sequences end up with pad tokens between prompt and continuation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_cls = (Gemma4ForConditionalGeneration
                     if "gemma-4" in model_name.lower()
                     else AutoModelForCausalLM)
        try:
            self.model = model_cls.from_pretrained(
                model_name, dtype=torch.bfloat16, device_map=device,
            )
        except (ValueError, KeyError) as e:
            # some multimodal checkpoints are not registered for causal LM loading
            print(f"{model_cls.__name__} could not load {model_name} ({e!r}); "
                  f"trying AutoModelForImageTextToText")
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, dtype=torch.bfloat16, device_map=device,
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

        # Does the rendered chat template already begin with BOS? If so the
        # tokenizer must not add another one. Checked, not assumed: if the
        # template has no BOS, special tokens must stay on or the judge
        # would lose its BOS entirely.
        self._template_has_bos = False
        if self.use_chat_template:
            probe = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "x"}],
                tokenize=False, add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
            bos = getattr(self.tokenizer, "bos_token", None)
            self._template_has_bos = bool(bos) and probe.startswith(bos)
            print(f"Chat template starts with BOS ({bos!r}): "
                  f"{self._template_has_bos}")

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
        return_index: bool = False,
    ):
        """
        Few-shot demonstrations from the gold set, keyed by (element,
        score). Frozen in-context demonstrations only; no gradients.

        return_index=True also returns the set of gold_df index labels that
        were used as demonstrations, so evaluation can exclude them.
        Selection is the first max_examples rows of each cell, so it is
        deterministic for a given gold_df.
        """
        examples = {}
        used = set()
        for (element, score), group in gold_df.groupby([element_col, score_col]):
            chosen = group.index.tolist()[:max_examples]
            examples[(element, int(score))] = group.loc[chosen, text_col].tolist()
            used.update(chosen)
        if return_index:
            return examples, used
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
                **self.chat_template_kwargs,
            )
        return prompt + "\nScore:"

    # A standalone 0, 2 or 4: not glued to letters or digits ("Element4",
    # "2024") and not the start of a decimal ("0.5"). A trailing sentence
    # period ("Score: 2.") is fine.
    _SCORE_RE = re.compile(r"(?<![\w.])([024])(?!\w|\.\d)")

    @classmethod
    def _parse_score(cls, generated: str) -> int | None:
        """
        First standalone 0, 2 or 4 in the output. The old parser took the
        first 0, 2 or 4 character anywhere, so "Element4: 2" parsed as 4
        and "2024" parsed as 2.
        """
        m = cls._SCORE_RE.search(generated)
        return int(m.group(1)) if m else None

    @torch.no_grad()
    def predict(
        self,
        candidates: list[str],
        elements: list[str],
        return_raw: bool = False,
    ):
        """
        Blind score prediction, batched. No knowledge of any target score.
        return_raw also returns the decoded output per candidate.
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
                max_length=self.max_length,
                # avoid a double BOS when the template already has one
                add_special_tokens=not self._template_has_bos,
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
