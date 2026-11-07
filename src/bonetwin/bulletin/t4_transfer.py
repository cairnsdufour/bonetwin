"""T4: cross-dataset generalisation.

Ref: Table III, Sec. IV-D (internal 0.898 / external 0.859, delta -3.9pp).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bonetwin.residuals.discrimination import auc_score

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class TransferReport:
    internal_auc: float
    external_auc: float
    gap: float


def evaluate_transfer(
    internal_endpoint: IntArray,
    internal_p: FloatArray,
    external_endpoint: IntArray,
    external_p: FloatArray,
) -> TransferReport:
    internal = auc_score(internal_endpoint, internal_p)
    external = auc_score(external_endpoint, external_p)
    return TransferReport(internal, external, external - internal)
