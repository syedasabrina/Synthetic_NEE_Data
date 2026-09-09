from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalCrossEntropy(nn.Module):
    """
    Distance-weighted cross entropy for ordinal rubric scores.

    The NEE rubric levels 0, 2, 4 map to class indices 0, 1, 2. Plain
    cross entropy treats those as unordered: predicting 4 when the
    truth is 0 costs exactly as much as predicting 2 when the truth is
    0. That is wrong for a rubric, where being one level off is a much
    smaller error than being two.

    This matters here more than usual because the evaluation metric is
    quadratic weighted kappa, which penalizes squared ordinal distance.
    Training with plain CE while reporting QWK optimizes a different
    objective than the one being measured.

    Two mechanisms:

    Soft targets. Instead of a one-hot label, probability mass is
    spread onto adjacent classes controlled by `smoothing`. A true
    label of 2 (index 1) puts most mass on index 1 and a little on
    indices 0 and 2, so predicting an adjacent class is partially
    credited rather than fully punished.

    Distance weighting. The loss for each example is scaled by the
    ordinal distance between the predicted and true class, so
    two-level errors contribute more gradient than one-level errors.
    """

    def __init__(
        self,
        num_classes: int = 3,
        smoothing: float = 0.1,
        distance_power: float = 2.0,
        class_weights: torch.Tensor | None = None,
    ):
        """
        smoothing: fraction of probability mass moved off the true
        class onto its ordinal neighbours, distributed by inverse
        distance. 0.0 reduces this to weighted cross entropy.

        distance_power: exponent on |predicted - true|. 2.0 matches the
        quadratic weighting in QWK.

        class_weights: optional per-class weights. Score 0 is severely
        underrepresented (31 real anchors, and roughly 10% of synthetic
        prompts), so without weighting the model can score well by
        never predicting it.
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.distance_power = distance_power
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None
            else torch.ones(num_classes),
        )

        # distance matrix between class indices, used for both the
        # soft targets and the per-example weighting
        idx = torch.arange(num_classes).float()
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        self.register_buffer("distance", dist)

    def _soft_targets(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Builds ordinal soft targets. Mass off the true class is
        allocated to neighbours in inverse proportion to distance, so
        an adjacent class receives more than a distant one.
        """
        n = labels.shape[0]
        targets = torch.zeros(
            n, self.num_classes, device=labels.device, dtype=torch.float
        )

        if self.smoothing <= 0:
            targets.scatter_(1, labels.unsqueeze(1), 1.0)
            return targets

        d = self.distance.to(labels.device)[labels]          # (n, C)
        neighbour = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
        denom = neighbour.sum(dim=1, keepdim=True).clamp(min=1e-8)
        targets = self.smoothing * neighbour / denom
        targets.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)
        return targets

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.float()
        log_probs = F.log_softmax(logits, dim=-1)

        targets = self._soft_targets(labels)
        per_example = -(targets * log_probs).sum(dim=-1)

        # scale by how far the model's current prediction is from truth
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            d = (pred - labels).abs().float()
            weight = 1.0 + d.pow(self.distance_power)

        cw = self.class_weights.to(labels.device)[labels]
        return (per_example * weight * cw).mean()


def compute_class_weights(
    labels: list[int],
    num_classes: int = 3,
    scheme: str = "sqrt_inverse",
) -> torch.Tensor:
    """
    Class weights from an observed label distribution.

    "sqrt_inverse" uses the square root of inverse frequency rather
    than plain inverse frequency. With a 220x imbalance between score 4
    and score 0 in the real corpus, plain inverse weighting hands score
    0 an enormous multiplier and destabilises training. The square root
    corrects the imbalance without letting a handful of examples
    dominate the gradient.
    """
    counts = torch.zeros(num_classes)
    for l in labels:
        counts[l] += 1
    counts = counts.clamp(min=1.0)

    if scheme == "inverse":
        w = counts.sum() / (num_classes * counts)
    elif scheme == "sqrt_inverse":
        w = (counts.sum() / (num_classes * counts)).sqrt()
    elif scheme == "none":
        w = torch.ones(num_classes)
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    return w / w.mean()