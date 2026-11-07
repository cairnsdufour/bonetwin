"""Stratified split-conformal prediction layer.

Ref: Sec. III-C, Eq. (3); Alg. 1 L3. Calibration scores are pooled per
stratum g in sex x age x vendor with |D_cal(g)| >= 500; strata below that
size borrow the marginal quantile (Sec. III-C exchangeability assumption).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bonetwin.alerts.scoring import FloatArray, conformity_score, logit, sigmoid


def conformal_quantile(scores: FloatArray, alpha: float) -> float:
    n = scores.shape[0]
    if n == 0:
        return float("inf")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


@dataclass
class StratifiedConformal:
    alphas: tuple[float, ...]
    min_stratum: int = 500
    smoothing: float = 0.01
    pooled: dict[float, float] = field(default_factory=dict)
    per_stratum: dict[float, dict[int, float]] = field(default_factory=dict)

    def fit(self, p_hat: FloatArray, y: np.ndarray, stratum: np.ndarray) -> StratifiedConformal:
        scores = conformity_score(p_hat, y.astype(np.int64), self.smoothing)
        for alpha in self.alphas:
            self.pooled[alpha] = conformal_quantile(scores, alpha)
            table: dict[int, float] = {}
            for g in np.unique(stratum):
                mask = stratum == g
                if int(mask.sum()) >= self.min_stratum:
                    table[int(g)] = conformal_quantile(scores[mask], alpha)
            self.per_stratum[alpha] = table
        return self

    def quantile(self, alpha: float, stratum_id: int) -> float:
        return self.per_stratum[alpha].get(stratum_id, self.pooled[alpha])

    def interval(
        self, p_hat: FloatArray, stratum: np.ndarray, alpha: float
    ) -> tuple[FloatArray, FloatArray]:
        q = np.array([self.quantile(alpha, int(g)) for g in stratum], dtype=np.float64)
        center = logit(p_hat)
        return sigmoid(center - q), sigmoid(center + q)
