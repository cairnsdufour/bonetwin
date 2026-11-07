"""Theorem 1 verifier: trajectory-level marginal CP validity.

Ref: Appendix I, Theorem 1 (Vovk-Gammerman split-CP guarantee
Pr[E5(x) in C_alpha(x)] >= 1 - alpha).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bonetwin.alerts.scoring import conformity_score, logit, sigmoid, smooth_labels
from bonetwin.alerts.thresholds import conformal_quantile

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def marginal_coverage(
    p_cal: FloatArray,
    y_cal: IntArray,
    p_test: FloatArray,
    y_test: IntArray,
    alpha: float,
    smoothing: float = 0.01,
) -> float:
    scores = conformity_score(p_cal, y_cal, smoothing)
    q = conformal_quantile(scores, alpha)
    center = logit(p_test)
    lo = sigmoid(center - q)
    hi = sigmoid(center + q)
    target = smooth_labels(y_test, smoothing)
    inside = (target >= lo) & (target <= hi)
    return float(inside.mean())
