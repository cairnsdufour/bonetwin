"""Three-bucket conformal triage funnel.

Ref: Sec. III-C, Alg. 1 L7-8: low if hi <= tau_low, high if lo > tau_high,
uncertain otherwise. tau_low=0.15, tau_high=0.50 (FRAX 20% MOF / 3% hip).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bonetwin.alerts.scoring import FloatArray

LOW = 0
UNCERTAIN = 1
HIGH = 2


def triage(
    lo: FloatArray, hi: FloatArray, tau_low: float, tau_high: float
) -> npt.NDArray[np.int64]:
    bucket = np.full(lo.shape[0], UNCERTAIN, dtype=np.int64)
    bucket[hi <= tau_low] = LOW
    bucket[lo > tau_high] = HIGH
    return bucket
