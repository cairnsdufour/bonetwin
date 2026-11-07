"""Theorem 2 verifier: stratified conditional-coverage gap.

Ref: Appendix I, Theorem 2 (DKW + union bound over K strata:
max_g |c_hat_g - (1-alpha)| <= sqrt(log(2K/delta) / (2 n_min)); with K=4,
delta=0.05, n_min=483 this yields <= 4.4pp).
"""

from __future__ import annotations

import math


def dkw_union_gap(n_strata: int, n_min: int, delta: float) -> float:
    return math.sqrt(math.log(2.0 * n_strata / delta) / (2.0 * n_min))


def gap_within_bound(empirical_gap: float, n_strata: int, n_min: int, delta: float) -> bool:
    return empirical_gap <= dkw_union_gap(n_strata, n_min, delta) + 1.0e-9
