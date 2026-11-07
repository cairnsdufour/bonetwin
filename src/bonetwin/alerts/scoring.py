"""Conformity scores on the logit of the 5-year transition probability.

Ref: Sec. III-C (s = |logit p_hat(5) - logit y*|, label-smoothed y*).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def logit(p: FloatArray, eps: float = 1.0e-6) -> FloatArray:
    clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(x: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-x))


def smooth_labels(y: npt.NDArray[np.int64], smoothing: float) -> FloatArray:
    return y.astype(np.float64) * (1.0 - 2.0 * smoothing) + smoothing


def conformity_score(
    p_hat: FloatArray, y: npt.NDArray[np.int64], smoothing: float = 0.01
) -> FloatArray:
    y_star = smooth_labels(y, smoothing)
    return np.abs(logit(p_hat) - logit(y_star))
