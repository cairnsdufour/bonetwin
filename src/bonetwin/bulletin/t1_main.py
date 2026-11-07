"""T1: primary 5-year pre-osteoporotic transition discrimination.

Ref: Table I, Sec. IV-B (T1a AUC 0.859 [0.841,0.876]; coverage and calibration).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bonetwin.residuals.calibration import brier_score
from bonetwin.residuals.coverage import conditional_coverage_gap, empirical_coverage
from bonetwin.residuals.discrimination import bootstrap_auc_ci

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class PrimaryReport:
    auc: float
    auc_lo: float
    auc_hi: float
    coverage: float
    conditional_gap: float
    brier: float


def evaluate_primary(
    p_hat: FloatArray,
    endpoint: IntArray,
    stratum: IntArray,
    lo: FloatArray,
    hi: FloatArray,
    target_coverage: float,
    n_boot: int = 2000,
    seed: int = 0,
) -> PrimaryReport:
    auc, auc_lo, auc_hi = bootstrap_auc_ci(endpoint, p_hat, n_boot=n_boot, seed=seed)
    coverage = empirical_coverage(lo, hi, endpoint)
    gap = conditional_coverage_gap(lo, hi, endpoint, stratum, target_coverage)
    brier = brier_score(endpoint, p_hat)
    return PrimaryReport(auc, auc_lo, auc_hi, coverage, gap, brier)
