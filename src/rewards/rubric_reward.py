from __future__ import annotations

import torch
from pathlib import Path
from transformers import Gemma4ForConditionalGeneration, AutoTokenizer


class RubricReward:
    """
    Computes a rubric alignment reward for candidate synthetic BIPs
    using Gemma 4 E4B as a frozen few-shot rubric judge.

    The judge is BLIND to the target score -- it is shown all three
    rubric levels and asked to independently determine which one the
    candidate earns. The target score is only used afterward, outside
    the model, to compute the reward. This matters because revealing
    the target and asking "does this earn a {target}?" biases an
    instruction-tuned model toward agreement regardless of actual
    quality.

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
    ):
        """
        few_shot_examples: dict keyed by (element, score) -> list[str]
        of real BIP texts known to earn that score. Populate this
        using build_few_shot_examples() on the gold standard set
        before instantiating, e.g.:

            gold_df = load_gold_standard(...)
            fs = RubricReward.build_few_shot_examples(gold_df)
            judge = RubricReward(few_shot_examples=fs)
        """
        self.device = device
        self.few_shot_examples = few_shot_examples or {}

        print(f"Loading Gemma 4 rubric reward model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
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
        max_examples: int = 3,
    ) -> dict:
        """
        Builds the few_shot_examples dict from the 23 gold standard
        BIPs, grouped by (element, score). Used only as frozen
        in-context demonstrations -- never as training data. The
        gold standard set remains held out for final evaluation.
        """
        examples = {}
        for (element, score), group in gold_df.groupby([element_col, score_col]):
            examples[(element, int(score))] = group[text_col].tolist()[:max_examples]
        return examples

    def _build_prompt(self, element: str, candidate: str) -> str:
        """
        Builds a BLIND scoring prompt. All three rubric levels are
        shown; the model must independently pick one. The target
        score is never mentioned here.
        """
        criteria_block = "\n".join(
            f"Score {score}: {text}"
            for score, text in sorted(self.RUBRIC[element].items())
        )

        prompt = f"""You are an expert evaluator of school principal Building Improvement Plans (BIPs).

You will independently score a BIP response for {element} according to the NEE rubric below. Do not assume any particular score -- judge based only on the content of the response.

Rubric criteria for {element}:
{criteria_block}

"""
        # attach few-shot examples across all three score levels if available,
        # so the judge sees calibration anchors without knowing which
        # level the candidate should hit
        any_examples = False
        for score in (0, 2, 4):
            examples = self.few_shot_examples.get((element, score), [])
            if examples:
                if not any_examples:
                    prompt += "Reference examples at each score level:\n\n"
                    any_examples = True
                prompt += f"Example earning score {score}:\n{examples[0]}\n\n"

        prompt += f"""Now read the following BIP response and determine which
score (0, 2, or 4) it earns.

BIP response:
{candidate}

Respond with only a number: 0, 2, or 4.
Score:"""

        return prompt

    @torch.no_grad()
    def predict(self, candidates: list[str], elements: list[str]) -> list[int | None]:
        """
        Returns the judge's independently predicted score for each
        candidate, with no knowledge of any intended target score.
        """
        predictions = []

        for candidate, element in zip(candidates, elements):
            prompt = self._build_prompt(element, candidate)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1536,
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            predictions.append(self._parse_score(generated))

        return predictions

    def score(
        self,
        candidates: list[str],
        elements: list[str],
        target_scores: list[int],
    ) -> list[float]:
        """
        Scores a list of candidate BIPs for rubric alignment. The
        judge predicts blind (see predict()); reward is computed
        afterward by comparing the blind prediction to target_scores.

            1.0  -- predicted score matches target score exactly
            0.5  -- predicted score is adjacent (one rubric step away)
            0.0  -- predicted score is far off or unparseable
        """
        predicted_scores = self.predict(candidates, elements)
        return [
            self._compute_reward(pred, target)
            for pred, target in zip(predicted_scores, target_scores)
        ]

    def _parse_score(self, generated: str) -> int | None:
        for token in generated.split():
            cleaned = token.strip(".,;:")
            if cleaned in ("0", "2", "4"):
                return int(cleaned)
        return None

    def _compute_reward(self, predicted: int | None, target: int) -> float:
        if predicted is None:
            return 0.0
        if predicted == target:
            return 1.0
        if abs(predicted - target) == 2:
            return 0.5
        return 0.0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    judge = RubricReward(model_name="google/gemma-4-E4B-it", device="cuda")

    candidates = [
        "Throughout the school year, we monitored our building "
        "improvement objectives by collecting MAP data every eight "
        "weeks and reviewing results with our grade level teams. "
        "When data showed students were not making expected progress "
        "in reading fluency, we implemented small group intervention "
        "blocks three times per week and adjusted our pacing guide.",

        "We monitored our goals.",

        "Our BIP objectives are directly aligned to the district "
        "CSIP goals for 2023-2024. Objective 1 supports CSIP Goal 2 "
        "around increasing ELA proficiency. Objective 2 supports "
        "CSIP Goal 3 around improving math benchmark scores. "
        "Each objective includes a measurable target tied to "
        "district performance indicators.",
    ]

    elements = ["Element6", "Element6", "Element3"]
    target_scores = [4, 4, 4]

    print("Testing BLIND rubric alignment scoring:")
    predicted = judge.predict(candidates, elements)
    rewards = judge.score(candidates, elements, target_scores)
    for cand, elem, target, pred, reward in zip(
        candidates, elements, target_scores, predicted, rewards
    ):
        print(f"  {elem} | predicted={pred} target={target} "
              f"reward={reward:.1f} | {cand[:50]}...")