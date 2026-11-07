"""Empirical and conditional coverage of the conformal intervals.

Ref: Sec. IV-F (pooled coverage, per-subgroup conditional-coverage gap).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bonetwin.alerts.scoring import smooth_labels

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def empirical_coverage(
    lo: FloatArray, hi: FloatArray, y: IntArray, smoothing: float = 0.01
) -> float:
    target = smooth_labels(y.astype(np.int64), smoothing)
    inside = (target >= lo) & (target <= hi)
    return float(inside.mean())


def conditional_coverage_gap(
    lo: FloatArray, hi: FloatArray, y: IntArray, stratum: IntArray, target: float
) -> float:
    worst = 0.0
    for g in np.unique(stratum):
        mask = stratum == g
        cov = empirical_coverage(lo[mask], hi[mask], y[mask])
        worst = max(worst, abs(cov - target))
    return worst
