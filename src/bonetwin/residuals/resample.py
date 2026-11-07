"""Stratified bootstrap and seed aggregation.

Ref: Sec. IV-A (mean over 20 seeds, 2000-sample stratified bootstrap).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]


def stratified_indices(stratum: IntArray, rng: np.random.Generator) -> IntArray:
    parts: list[IntArray] = []
    for g in np.unique(stratum):
        members = np.where(stratum == g)[0]
        parts.append(rng.choice(members, size=members.shape[0], replace=True))
    return np.concatenate(parts).astype(np.int64)


def seed_mean_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    lo = float(np.percentile(arr, 100.0 * alpha / 2.0))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return mean, lo, hi
