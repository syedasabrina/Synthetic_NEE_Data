from __future__ import annotations

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


class RubricReward:
    """
    Computes a rubric alignment reward for candidate synthetic BIPs
    using Gemma-2-2B as a frozen few-shot rubric judge.

    The model receives the NEE rubric criteria for the target element
    and score level, plus a few gold standard examples, and predicts
    whether the candidate BIP warrants the target score.

    This model is always frozen -- it is never updated during PPO.
    """

    # rubric criteria text for each element and score level
    # sourced directly from rubric.tsv
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

    SCORE_LABELS = {0: "0", 2: "2", 4: "4"}

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        device: str = "cuda",
        few_shot_examples: dict | None = None,
    ):
        self.device = device
        self.few_shot_examples = few_shot_examples or {}

        print(f"Loading Gemma rubric reward model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        print("RubricReward ready.")

    def _build_prompt(
        self,
        element: str,
        target_score: int,
        candidate: str,
    ) -> str:
        """
        Builds the few-shot prompt for the rubric judge.
        """
        rubric_criteria = self.RUBRIC[element][target_score]
        examples = self.few_shot_examples.get((element, target_score), [])

        prompt = f"""You are an expert evaluator of school principal Building Improvement Plans (BIPs).

You will score a BIP response for {element} according to the NEE rubric.

Rubric criteria for score {target_score}:
{rubric_criteria}

"""
        if examples:
            prompt += "Examples of BIPs that earn this score:\n\n"
            for i, ex in enumerate(examples[:3], 1):
                prompt += f"Example {i}:\n{ex}\n\n"

        prompt += f"""Now score the following BIP response for {element}.
The target score is {target_score}. Does this BIP warrant a score of {target_score}?

BIP response:
{candidate}

Respond with only a number: 0, 2, or 4.
Score:"""

        return prompt

    @torch.no_grad()
    def score(
        self,
        candidates: list[str],
        elements: list[str],
        target_scores: list[int],
    ) -> list[float]:
        """
        Scores a list of candidate BIPs for rubric alignment.

        Returns a reward in [0, 1] for each candidate:
            1.0  -- predicted score matches target score exactly
            0.5  -- predicted score is adjacent (one level away)
            0.0  -- predicted score is far off
        """
        rewards = []

        for candidate, element, target_score in zip(
            candidates, elements, target_scores
        ):
            prompt = self._build_prompt(element, target_score, candidate)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # decode only the generated tokens, not the prompt
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            # extract predicted score from generated text
            predicted_score = self._parse_score(generated)
            reward = self._compute_reward(predicted_score, target_score)
            rewards.append(reward)

        return rewards

    def _parse_score(self, generated: str) -> int | None:
        """
        Extracts a score (0, 2, or 4) from the model's generated text.
        Returns None if no valid score found.
        """
        for token in generated.split():
            cleaned = token.strip(".,;:")
            if cleaned in ("0", "2", "4"):
                return int(cleaned)
        return None

    def _compute_reward(
        self,
        predicted: int | None,
        target: int,
    ) -> float:
        """
        Converts predicted vs target score into a reward value.
        """
        if predicted is None:
            return 0.0
        if predicted == target:
            return 1.0
        # adjacent scores in the rubric: 0 <-> 2 <-> 4
        if abs(predicted - target) == 2:
            return 0.5
        # far off: 0 vs 4
        return 0.0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    judge = RubricReward(
        model_name="google/gemma-2-2b-it",
        device="cuda",
    )

    candidates = [
        # should score high for Element6 at score 4
        "Throughout the school year, we monitored our building "
        "improvement objectives by collecting MAP data every eight "
        "weeks and reviewing results with our grade level teams. "
        "When data showed students were not making expected progress "
        "in reading fluency, we implemented small group intervention "
        "blocks three times per week and adjusted our pacing guide.",

        # should score low for Element6 at score 4
        "We monitored our goals.",

        # should score high for Element3 at score 4
        "Our BIP objectives are directly aligned to the district "
        "CSIP goals for 2023-2024. Objective 1 supports CSIP Goal 2 "
        "around increasing ELA proficiency. Objective 2 supports "
        "CSIP Goal 3 around improving math benchmark scores. "
        "Each objective includes a measurable target tied to "
        "district performance indicators.",
    ]

    elements = ["Element6", "Element6", "Element3"]
    target_scores = [4, 4, 4]

    print("Testing rubric alignment scoring:")
    rewards = judge.score(candidates, elements, target_scores)
    for cand, elem, target, reward in zip(
        candidates, elements, target_scores, rewards
    ):
        print(f"  {elem} | target={target} | reward={reward:.1f} | "
              f"{cand[:60]}...")