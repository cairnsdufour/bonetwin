"""Probabilistic calibration scores.

Ref: Sec. III-D, Sec. IV-E (Brier: MI 0.187 vs baseline 0.471).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def brier_score(y: IntArray, p: FloatArray) -> float:
    return float(np.mean((p - y.astype(np.float64)) ** 2))
