from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


VALID_SCORES = [0, 2, 4]
LABEL_MAP = {0: 0, 2: 1, 4: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    """
    QWK on class indices, the standard metric for automated rubric
    scoring. Penalises squared ordinal distance, so predicting index 2
    when the truth is 0 costs four times as much as being one level
    off. Returns 0.0 when a single class is present, since kappa is
    undefined there.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(np.unique(np.concatenate([y_true, y_pred]))) < 2:
        return 0.0
    return float(cohen_kappa_score(
        y_true, y_pred, weights="quadratic", labels=list(range(3))
    ))


def bootstrap_qwk(
    y_true,
    y_pred,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Percentile bootstrap CI for QWK.

    The gold set is 171 rows over 24 principals. A point estimate at
    that size carries wide uncertainty, and reporting QWK without an
    interval would overstate precision. Resamples that end up with one
    class present are dropped rather than scored as 0.0, which would
    drag the lower bound down for a reason unrelated to model quality.
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)

    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        t, p = y_true[idx], y_pred[idx]
        if len(np.unique(np.concatenate([t, p]))) < 2:
            continue
        stats.append(quadratic_weighted_kappa(t, p))

    if not stats:
        return {"qwk": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_valid": 0}

    stats = np.array(stats)
    a = (1 - confidence) / 2
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "ci_low": float(np.percentile(stats, a * 100)),
        "ci_high": float(np.percentile(stats, (1 - a) * 100)),
        "bootstrap_mean": float(stats.mean()),
        "n_valid": len(stats),
    }


def leave_one_out_qwk(y_true, y_pred) -> dict:
    """
    QWK recomputed with each example held out in turn.

    With 171 rows, a handful of examples can move the point estimate.
    A large spread here means the score rests on a few cases.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    scores = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        t, p = y_true[mask], y_pred[mask]
        if len(np.unique(np.concatenate([t, p]))) < 2:
            continue
        scores.append(quadratic_weighted_kappa(t, p))
    scores = np.array(scores)
    return {
        "loo_mean": float(scores.mean()) if len(scores) else 0.0,
        "loo_std": float(scores.std()) if len(scores) else 0.0,
        "loo_min": float(scores.min()) if len(scores) else 0.0,
        "loo_max": float(scores.max()) if len(scores) else 0.0,
    }


def agreement_metrics(y_true, y_pred) -> dict:
    """
    Exact and adjacent agreement on class indices, plus mean signed
    deviation in rubric points.

    Signed deviation is the diagnostic that matters for the audit: a
    positive value means the assessor scores higher than the reference,
    which is the same direction supervisors err. An assessor that
    inherits supervisor leniency cannot be used to detect it.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    true_pts = np.array([INV_LABEL_MAP[int(v)] for v in y_true])
    pred_pts = np.array([INV_LABEL_MAP[int(v)] for v in y_pred])
    return {
        "exact_agreement": float((y_true == y_pred).mean()),
        "adjacent_agreement": float((np.abs(y_true - y_pred) <= 1).mean()),
        "mean_signed_deviation": float((pred_pts - true_pts).mean()),
        "mean_absolute_deviation": float(np.abs(pred_pts - true_pts).mean()),
    }


def per_level_calibration(y_true, y_pred) -> dict:
    """
    Recall per rubric level.

    Aggregate accuracy hides the failure that matters most here. The
    corpus is 95% score 4, so a model that always predicts 4 looks
    accurate while being useless. Per-level recall exposes that
    immediately.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {}
    for idx in range(3):
        mask = y_true == idx
        score = INV_LABEL_MAP[idx]
        out[f"recall_score_{score}"] = (
            float((y_pred[mask] == idx).mean()) if mask.sum() else None
        )
        out[f"n_score_{score}"] = int(mask.sum())
    return out


def per_element_metrics(y_true, y_pred, elements) -> dict:
    """
    QWK and exact agreement per rubric element. Elements differ in
    what they demand -- Element 5 requires cited research, Element 6
    requires monitoring evidence -- so uniform aggregate performance
    can still hide an element the assessor cannot read.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    elements = np.asarray(elements)
    out = {}
    for e in sorted(set(elements)):
        m = elements == e
        if m.sum() < 2:
            continue
        out[e] = {
            "n": int(m.sum()),
            "exact": float((y_true[m] == y_pred[m]).mean()),
            "qwk": quadratic_weighted_kappa(y_true[m], y_pred[m]),
        }
    return out


def confusion(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(3)))
    return {
        "matrix": cm.tolist(),
        "rows": [f"true_{INV_LABEL_MAP[i]}" for i in range(3)],
        "cols": [f"pred_{INV_LABEL_MAP[i]}" for i in range(3)],
    }


def evaluate(
    y_true,
    y_pred,
    elements=None,
    bootstrap_n: int = 1000,
    run_loo: bool = True,
) -> dict:
    """Full evaluation report."""
    results = {}
    results.update(bootstrap_qwk(y_true, y_pred, n_resamples=bootstrap_n))
    results.update(agreement_metrics(y_true, y_pred))
    results.update(per_level_calibration(y_true, y_pred))
    if run_loo:
        results.update(leave_one_out_qwk(y_true, y_pred))
    results["confusion"] = confusion(y_true, y_pred)
    if elements is not None:
        results["per_element"] = per_element_metrics(y_true, y_pred, elements)
    return results


def print_report(results: dict, title: str = "EVALUATION") -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"QWK                  {results['qwk']:.4f}  "
          f"[{results['ci_low']:.4f}, {results['ci_high']:.4f}]  95% CI")
    if "loo_mean" in results:
        print(f"QWK leave-one-out    {results['loo_mean']:.4f} "
              f"(sd {results['loo_std']:.4f})")
    print(f"Exact agreement      {results['exact_agreement']:.4f}")
    print(f"Adjacent agreement   {results['adjacent_agreement']:.4f}")
    print(f"Mean signed dev      {results['mean_signed_deviation']:+.4f}  "
          f"(positive = assessor scores higher than reference)")
    print()

    print("Recall per rubric level:")
    for s in VALID_SCORES:
        r = results.get(f"recall_score_{s}")
        n = results.get(f"n_score_{s}", 0)
        print(f"  score {s}: {r if r is None else f'{r:.4f}'}  (n={n})")
    print()

    cm = results["confusion"]
    print("Confusion (rows = reference, cols = prediction):")
    print(f"{'':>10}" + "".join(f"{c:>10}" for c in cm["cols"]))
    for name, row in zip(cm["rows"], cm["matrix"]):
        print(f"{name:>10}" + "".join(f"{v:>10}" for v in row))
    print()

    if "per_element" in results:
        print("Per element:")
        for e, m in sorted(results["per_element"].items()):
            print(f"  {e}: exact={m['exact']:.3f}  qwk={m['qwk']:.3f}  "
                  f"(n={m['n']})")
    print()


def save_results(results: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {path}")


def compare_conditions(results_by_condition: dict[str, dict]) -> pd.DataFrame:
    """
    Side-by-side table across training conditions.

    The comparison the project turns on: does condition A beat B, and
    does it also beat C? A over B alone could be explained by score
    balancing rather than label quality. A over C as well isolates the
    labels.
    """
    rows = []
    for cond, r in results_by_condition.items():
        rows.append({
            "condition": cond,
            "qwk": round(r["qwk"], 4),
            "qwk_ci": f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
            "exact": round(r["exact_agreement"], 4),
            "adjacent": round(r["adjacent_agreement"], 4),
            "signed_dev": round(r["mean_signed_deviation"], 4),
            "recall_0": r.get("recall_score_0"),
            "recall_2": r.get("recall_score_2"),
            "recall_4": r.get("recall_score_4"),
        })
    return pd.DataFrame(rows).sort_values("qwk", ascending=False)