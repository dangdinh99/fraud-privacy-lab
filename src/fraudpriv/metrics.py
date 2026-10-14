"""Evaluation metrics. Every reported number comes from this module.

The primary metric is recall at a fixed alert budget: the top ``k`` fraction of all
transactions in the evaluation period, ranked by score, is flagged for review. The
alert budget is not a false-positive rate. Recall is always reported together with
its ceiling (the best recall any model could reach with that many alerts) and as a
share of that ceiling. Accuracy is never reported.
"""

import math
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score


def n_flagged(n: int, budget: float) -> int:
    """Number of transactions flagged at an alert budget: ``ceil(budget * n)``.

    The product is computed exactly from the decimal form of ``budget``, so for example
    ``budget=0.07, n=100`` gives 7, not the 8 that floating-point ``ceil`` would give.
    """
    if not 0 < budget <= 1:
        raise ValueError(f"budget must be in (0, 1], got {budget}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return math.ceil(Fraction(str(budget)) * n)


def flag_top_budget(scores: ArrayLike, ids: ArrayLike, budget: float) -> np.ndarray:
    """Boolean mask of the transactions flagged within the alert budget.

    Exactly ``ceil(budget * n)`` transactions are flagged. Transactions are ranked by
    score, highest first; ties in score are broken deterministically by ascending
    ``TransactionID``, so a tie at the cutoff always resolves the same way regardless of
    input order.
    """
    scores = np.asarray(scores, dtype=float)
    ids = np.asarray(ids)
    if scores.shape != ids.shape or scores.ndim != 1:
        raise ValueError("scores and ids must be 1-D arrays of the same length")
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("ids must be unique")
    k = n_flagged(len(scores), budget)
    order = np.lexsort((ids, -scores))  # primary: score descending; secondary: id ascending
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def recall_ceiling(n: int, n_fraud: int, budget: float) -> float:
    """Highest recall reachable at the budget: ``min(1, ceil(budget * n) / n_fraud)``.

    Computed from the actual evaluation period, never from an overall fraud rate. The
    numerator is the number of transactions actually flagged, ``ceil(budget * n)``, so
    recall can never exceed the ceiling (see docs/decisions.md).
    """
    if n_fraud <= 0:
        raise ValueError("evaluation period has no fraud; recall is undefined")
    return min(1.0, n_flagged(n, budget) / n_fraud)


def budget_metrics(
    y_true: ArrayLike,
    scores: ArrayLike,
    ids: ArrayLike,
    amounts: ArrayLike,
    budget: float,
    segments: dict[str, ArrayLike] | None = None,
) -> dict[str, Any]:
    """Recall, ceiling, precision, and dollar loss at one alert budget.

    - ``recall``: frauds flagged / all frauds in the evaluation period.
    - ``recall_ceiling``: see :func:`recall_ceiling`.
    - ``recall_share_of_ceiling``: ``recall / recall_ceiling``.
    - ``precision``: frauds flagged / transactions flagged.
    - ``dollar_loss``: total ``TransactionAmt`` of frauds not flagged.
    - ``segments`` (optional, name -> boolean mask): recall within each segment, for
      example seen vs new cards. Flags always come from the budget over the whole
      period; segments never change which rows are flagged. Segments have no ceiling of
      their own because the budget is not allocated per segment. Recall is ``None`` for
      a segment with no fraud.
    """
    y = np.asarray(y_true).astype(int)
    amounts = np.asarray(amounts, dtype=float)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must be binary 0/1")
    if y.shape != amounts.shape:
        raise ValueError("y_true and amounts must have the same length")
    flagged = flag_top_budget(scores, ids, budget)
    fraud = y == 1
    n, n_fraud = len(y), int(fraud.sum())
    ceiling = recall_ceiling(n, n_fraud, budget)
    caught = int((flagged & fraud).sum())
    recall = caught / n_fraud
    out: dict[str, Any] = {
        "budget": budget,
        "n": n,
        "n_fraud": n_fraud,
        "n_flagged": int(flagged.sum()),
        "frauds_flagged": caught,
        "recall": recall,
        "recall_ceiling": ceiling,
        "recall_share_of_ceiling": recall / ceiling,
        "precision": caught / int(flagged.sum()),
        "dollar_loss": float(amounts[fraud & ~flagged].sum()),
        "fraud_dollars_total": float(amounts[fraud].sum()),
    }
    if segments is not None:
        out["segments"] = {}
        for name, seg in segments.items():
            seg = np.asarray(seg, dtype=bool)
            if seg.shape != y.shape:
                raise ValueError(f"segment {name!r} has the wrong length")
            seg_fraud = int((fraud & seg).sum())
            seg_caught = int((flagged & fraud & seg).sum())
            out["segments"][name] = {
                "n": int(seg.sum()),
                "n_fraud": seg_fraud,
                "frauds_flagged": seg_caught,
                "recall": seg_caught / seg_fraud if seg_fraud else None,
            }
    return out


def pr_auc(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Area under the precision-recall curve, as average precision (scikit-learn)."""
    return float(average_precision_score(y_true, scores))


def expected_calibration_error(y_true: ArrayLike, probs: ArrayLike, n_bins: int) -> float:
    """Expected calibration error with ``n_bins`` equal-width bins on [0, 1].

    Bin ``i`` covers ``[i/n_bins, (i+1)/n_bins)``; the last bin also includes 1.0. ECE is
    the average over bins of ``|mean predicted probability - observed fraud rate|``,
    weighted by the fraction of transactions in the bin. Empty bins contribute zero.
    Requires probabilities, not raw scores.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("probs must be probabilities in [0, 1]")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        ece += in_bin.mean() * abs(p[in_bin].mean() - y[in_bin].mean())
    return float(ece)


def evaluate(
    y_true: ArrayLike,
    scores: ArrayLike,
    ids: ArrayLike,
    amounts: ArrayLike,
    budgets: tuple[float, ...],
    ece_bins: int | None,
    segments: dict[str, ArrayLike] | None = None,
    probs: ArrayLike | None = None,
) -> dict[str, Any]:
    """All metrics for one evaluation period.

    Ranking metrics (PR-AUC and :func:`budget_metrics` for each budget, keyed
    ``budget_<k>``, with per-segment recall when ``segments`` is given) use ``scores``.
    ECE is computed only when ``ece_bins`` is set, on ``probs`` if given, else on
    ``scores`` (which must then be probabilities). Passing raw scores plus calibrated
    ``probs`` keeps the ranking free of ties created when probabilities saturate.
    """
    out: dict[str, Any] = {"pr_auc": pr_auc(y_true, scores)}
    if ece_bins is not None:
        p = scores if probs is None else probs
        out["ece"] = expected_calibration_error(y_true, p, ece_bins)
        out["ece_bins"] = ece_bins
    for b in budgets:
        out[f"budget_{b}"] = budget_metrics(y_true, scores, ids, amounts, b, segments)
    return out


def aggregate_windows(windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine :func:`budget_metrics` results from consecutive windows (e.g. replay weeks).

    Each window has its own budget and ceiling. Totals are summed over windows:

    - ``recall``: frauds flagged / all frauds.
    - ``recall_ceiling``: sum over windows of ``min(n_flagged, n_fraud)`` / all frauds, the
      most any model could catch with these per-window budgets.
    - ``recall_share_of_ceiling``: frauds flagged / that maximum.
    - ``precision``, ``dollar_loss``, ``fraud_dollars_total``: pooled over windows.

    Windows with no fraud contribute flags and transactions but no frauds. They are allowed
    here, unlike in :func:`budget_metrics`, which is not called on them.
    """
    if not windows:
        raise ValueError("no windows")
    n_fraud = sum(w["n_fraud"] for w in windows)
    if n_fraud == 0:
        raise ValueError("no fraud in any window; recall is undefined")
    caught = sum(w["frauds_flagged"] for w in windows)
    flagged = sum(w["n_flagged"] for w in windows)
    best = sum(min(w["n_flagged"], w["n_fraud"]) for w in windows)
    return {
        "n_windows": len(windows),
        "n": sum(w["n"] for w in windows),
        "n_fraud": n_fraud,
        "n_flagged": flagged,
        "frauds_flagged": caught,
        "recall": caught / n_fraud,
        "recall_ceiling": best / n_fraud,
        "recall_share_of_ceiling": caught / best,
        "precision": caught / flagged,
        "dollar_loss": float(sum(w["dollar_loss"] for w in windows)),
        "fraud_dollars_total": float(sum(w["fraud_dollars_total"] for w in windows)),
    }
