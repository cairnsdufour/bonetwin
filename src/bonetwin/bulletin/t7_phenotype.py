"""T7: bimodal trajectory phenotype characterisation (C4).

Ref: Sec. IV-E (slow 69.1% / fast 30.9%; transition 62.1% vs 11.6%;
NNT 4.6 vs 24.6 under 35% RRR; I_123=+0.030, p=0.003).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bonetwin.swarms.benefit import number_needed_to_treat
from bonetwin.swarms.mixture import fit_phenotype

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class PhenotypeReport:
    n_components: int
    fast_fraction: float
    slow_fraction: float
    fast_transition_rate: float
    slow_transition_rate: float
    nnt_fast: float
    nnt_slow: float


def evaluate_phenotype(
    delta: FloatArray,
    endpoint: IntArray,
    relative_risk_reduction: float,
    max_components: int = 5,
    seed: int = 0,
) -> PhenotypeReport:
    fit = fit_phenotype(delta, max_components=max_components, seed=seed)
    fast_mask = fit.labels == fit.fast_index
    slow_mask = ~fast_mask
    fast_rate = float(endpoint[fast_mask].mean()) if fast_mask.any() else 0.0
    slow_rate = float(endpoint[slow_mask].mean()) if slow_mask.any() else 0.0
    return PhenotypeReport(
        n_components=fit.n_components,
        fast_fraction=fit.fast_fraction,
        slow_fraction=fit.slow_fraction,
        fast_transition_rate=fast_rate,
        slow_transition_rate=slow_rate,
        nnt_fast=number_needed_to_treat(fast_rate, relative_risk_reduction),
        nnt_slow=number_needed_to_treat(slow_rate, relative_risk_reduction),
    )
