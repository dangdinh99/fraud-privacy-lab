"""Drift monitoring: population stability index (PSI) between a reference and a later window.

Descriptive only in stage 3: drift is reported alongside replay performance and does not
trigger retraining.
"""

import numpy as np
from numpy.typing import ArrayLike


def psi(reference: ArrayLike, current: ArrayLike, n_bins: int, floor: float = 1e-4) -> float:
    """Population stability index of ``current`` against ``reference``.

    Bins are the reference's quantiles (``n_bins`` equal-frequency bins; repeated edges,
    e.g. from a binary or mostly constant column, are merged). Values outside the
    reference range fall into the first or last bin. Bin shares below ``floor`` are raised
    to ``floor`` so empty bins do not give infinite values.

    ``PSI = sum over bins of (cur - ref) * ln(cur / ref)``. Common reading: below 0.1
    stable, 0.1 to 0.25 moderate shift, above 0.25 large shift.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("reference and current must be non-empty")
    if np.isnan(ref).any() or np.isnan(cur).any():
        raise ValueError("inputs contain NaN")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1))[1:-1])
    ref_share = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1)
    cur_share = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1)
    ref_share = np.maximum(ref_share / len(ref), floor)
    cur_share = np.maximum(cur_share / len(cur), floor)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))
