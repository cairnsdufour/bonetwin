"""T2: component ablation and compositional synergy (P-34).

Ref: Table II, Sec. IV-C (sum-of-parts counterfactual; synergy +0.030, p=0.003).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bonetwin.bounds.synergy import sum_of_parts
from bonetwin.residuals.delong import delong_test

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class SynergyReport:
    full_auc: float
    sum_of_parts_auc: float
    interaction: float
    p_value: float


def evaluate_synergy(
    baseline_auc: float,
    standalone_deltas: Sequence[float],
    endpoint: IntArray,
    full_scores: FloatArray,
    parts_scores: FloatArray,
) -> SynergyReport:
    auc_full, auc_parts, p = delong_test(endpoint, full_scores, parts_scores)
    expected_parts = sum_of_parts(baseline_auc, standalone_deltas)
    interaction = auc_full - auc_parts
    return SynergyReport(auc_full, expected_parts, interaction, p)
