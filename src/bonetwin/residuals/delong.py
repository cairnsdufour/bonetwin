"""Paired DeLong test for the difference of two correlated AUCs.

Ref: Sec. IV-A, IV-B (DeLong-paired p with 95% CIs).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _midrank(x: FloatArray) -> FloatArray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        out[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = out
    return ranks


def _structural_components(
    scores: FloatArray, n_pos: int
) -> tuple[FloatArray, FloatArray, float]:
    pos = scores[:n_pos]
    neg = scores[n_pos:]
    m = pos.shape[0]
    n = neg.shape[0]
    tx = _midrank(pos)
    ty = _midrank(neg)
    tz = _midrank(scores)
    auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    return v01, v10, float(auc)


def delong_test(
    y: IntArray, scores_a: FloatArray, scores_b: FloatArray
) -> tuple[float, float, float]:
    order = np.argsort(-y)
    y_sorted = y[order]
    n_pos = int(np.sum(y_sorted == 1))
    a = scores_a[order]
    b = scores_b[order]
    v01a, v10a, auc_a = _structural_components(a, n_pos)
    v01b, v10b, auc_b = _structural_components(b, n_pos)
    m = n_pos
    n = y.shape[0] - n_pos
    s01 = np.cov(np.stack([v01a, v01b]))
    s10 = np.cov(np.stack([v10a, v10b]))
    cov = s01 / m + s10 / n
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if var <= 0.0:
        return auc_a, auc_b, 1.0
    z = (auc_a - auc_b) / np.sqrt(var)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, float(p)
