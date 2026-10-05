"""
src/evaluation/gold_metrics.py

Metrics for comparing scorers against the gold set, with the two things
src/evaluation/metrics.py lacks:

1. Principal-level bootstrap. The gold rows cluster by principal (24
   principals, 6 to 9 rows each, one annotator or annotator group per
   principal), so resampling rows gives intervals that are too narrow.
   Here whole principals are resampled.
2. Paired comparison. To say "scorer A beats scorer B" the two must be
   scored on the same rows and resampled together, so the interval is on
   the difference.

Labels are rubric points 0, 2, 4.
"""

from __future__ import annotations

import numpy as np

LEVELS = (0, 2, 4)
_IDX = {0: 0, 2: 1, 4: 2}


def qwk(y_true, y_pred) -> float:
    """Quadratic weighted kappa over the 0/2/4 levels. NaN if undefined."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return float("nan")
    k = len(LEVELS)
    obs = np.zeros((k, k))
    for t, p in zip(y_true, y_pred):
        obs[_IDX[int(t)], _IDX[int(p)]] += 1
    n = obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)],
                 dtype=float) / (k - 1) ** 2
    denom = (w * exp).sum()
    if denom == 0:
        return float("nan")
    return float(1.0 - (w * obs).sum() / denom)


def exact(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


STATS = {"exact": exact, "qwk": qwk}


def summarize(y_true, y_pred) -> dict:
    """Point estimates: agreement, adjacent agreement, bias, QWK, recall."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out = {
        "n": int(len(y_true)),
        "exact": exact(y_true, y_pred),
        "adjacent": float((np.abs(y_true - y_pred) <= 2).mean()) if len(y_true) else float("nan"),
        "signed_dev": float((y_pred - y_true).mean()) if len(y_true) else float("nan"),
        "qwk": qwk(y_true, y_pred),
    }
    for lv in LEVELS:
        m = y_true == lv
        out[f"recall_{lv}"] = float((y_pred[m] == lv).mean()) if m.any() else None
        out[f"n_{lv}"] = int(m.sum())
    return out


def always_two(y_true) -> dict:
    """The majority-class baseline for these labels: always answer 2."""
    y_true = np.asarray(y_true, dtype=int)
    return {"exact": exact(y_true, np.full_like(y_true, 2)), "qwk": 0.0}


def _groups(person_ids) -> list[np.ndarray]:
    person_ids = np.asarray(person_ids)
    return [np.flatnonzero(person_ids == p) for p in np.unique(person_ids)]


def _draw(groups, rng) -> np.ndarray:
    pick = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[i] for i in pick])


def principal_ci(y_true, y_pred, person_ids, stat: str = "exact",
                 n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI for one scorer, resampling whole principals."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fn = STATS[stat]
    groups = _groups(person_ids)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = _draw(groups, rng)
        v = fn(y_true[idx], y_pred[idx])
        if v == v:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def paired_principal_diff(y_true, pred_a, pred_b, person_ids,
                          stat: str = "exact", n_boot: int = 2000,
                          seed: int = 0, alpha: float = 0.05) -> dict:
    """
    stat(A) - stat(B) on identical rows, with both scorers evaluated on
    the same principal resamples. Also returns the share of resamples in
    which A is no better than B.
    """
    y_true = np.asarray(y_true)
    pred_a, pred_b = np.asarray(pred_a), np.asarray(pred_b)
    fn = STATS[stat]
    groups = _groups(person_ids)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _draw(groups, rng)
        a = fn(y_true[idx], pred_a[idx])
        b = fn(y_true[idx], pred_b[idx])
        if a == a and b == b:
            diffs.append(a - b)
    diffs = np.asarray(diffs)
    point = fn(y_true, pred_a) - fn(y_true, pred_b)
    if len(diffs) == 0:
        return {"diff": float(point), "lo": float("nan"), "hi": float("nan"),
                "share_a_not_better": float("nan"), "n_boot_valid": 0}
    return {
        "diff": float(point),
        "lo": float(np.percentile(diffs, 100 * alpha / 2)),
        "hi": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
        "share_a_not_better": float((diffs <= 0).mean()),
        "n_boot_valid": int(len(diffs)),
    }
