from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.rewards.authenticity_reward import AuthenticityReward
from src.rewards.rubric_reward import RubricReward


@dataclass
class RewardOutput:
    """
    Holds the full reward breakdown for one candidate BIP.
    Logged as a diagnostic for every generated example.
    """
    text: str
    element: str
    target_score: int
    authenticity: float
    rubric: float
    combined: float
    kl_penalty: float = 0.0
    final: float = 0.0


class CombinedReward:
    """
    Combines authenticity and rubric alignment rewards into a single
    scalar reward for PPO training.

    r_final = alpha * r_auth + (1 - alpha) * r_rubric - beta * kl

    alpha: weight on authenticity reward (0 to 1)
    beta: weight on KL penalty (prevents generator drift)
    """

    def __init__(
        self,
        authenticity_reward: AuthenticityReward,
        rubric_reward: RubricReward,
        alpha: float = 0.5,
        beta: float = 0.1,
    ):
        self.auth = authenticity_reward
        self.rubric = rubric_reward
        self.alpha = alpha
        self.beta = beta

    def score(
        self,
        candidates: list[str],
        elements: list[str],
        target_scores: list[int],
        kl_penalties: list[float] | None = None,
    ) -> list[RewardOutput]:
        """
        Computes full reward breakdown for a batch of candidates.
        Returns RewardOutput for each candidate for logging and PPO.
        """
        if kl_penalties is None:
            kl_penalties = [0.0] * len(candidates)

        auth_rewards = self.auth.score_batch(candidates)
        rubric_rewards = self.rubric.score(
            candidates, elements, target_scores
        )

        outputs = []
        for text, element, target, auth_r, rubric_r, kl in zip(
            candidates, elements, target_scores,
            auth_rewards, rubric_rewards, kl_penalties
        ):
            combined = (
                self.alpha * auth_r + (1 - self.alpha) * rubric_r
            )
            final = combined - self.beta * kl

            outputs.append(RewardOutput(
                text=text,
                element=element,
                target_score=target,
                authenticity=auth_r,
                rubric=rubric_r,
                combined=combined,
                kl_penalty=kl,
                final=final,
            ))

        return outputs

    def scalar_rewards(
        self,
        candidates: list[str],
        elements: list[str],
        target_scores: list[int],
        kl_penalties: list[float] | None = None,
    ) -> list[float]:
        """
        Returns only the final scalar reward for each candidate.
        Used directly by TRL PPOTrainer.
        """
        outputs = self.score(
            candidates, elements, target_scores, kl_penalties
        )
        return [o.final for o in outputs]