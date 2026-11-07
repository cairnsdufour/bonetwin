"""Area under the ROC curve with bootstrap confidence intervals.

Ref: Sec. IV, Tables I-III (AUC with 95% CI over 20 seeds x 2000 bootstrap).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def auc_score(y: IntArray, scores: FloatArray) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    _assign_tied_ranks(scores, ranks)
    rank_sum_pos = ranks[y == 1].sum()
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _assign_tied_ranks(scores: FloatArray, ranks: FloatArray) -> None:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    i = 0
    n = scores.size
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0 + 1.0
            ranks[order[i : j + 1]] = avg
        i = j + 1


def bootstrap_auc_ci(
    y: IntArray, scores: FloatArray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    point = auc_score(y, scores)
    estimates = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        estimates[b] = auc_score(y[idx], scores[idx])
    lo = float(np.nanpercentile(estimates, 100.0 * alpha / 2.0))
    hi = float(np.nanpercentile(estimates, 100.0 * (1.0 - alpha / 2.0)))
    return point, lo, hi
