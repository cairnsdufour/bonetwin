"""Theorem 3 verifier: compositional synergy lower bound.

Ref: Appendix I, Theorem 3 (A3 >= A0 + sum_i (A_i - A_{i-1}) + I_123 with
I_123 >= 0; the T2 sum-of-parts counterfactual gives I_123 = +0.030).
"""

from __future__ import annotations

from collections.abc import Sequence


def sum_of_parts(baseline_auc: float, standalone_deltas: Sequence[float]) -> float:
    return baseline_auc + sum(standalone_deltas)


def compositional_interaction(
    full_auc: float, baseline_auc: float, standalone_deltas: Sequence[float]
) -> float:
    return full_auc - sum_of_parts(baseline_auc, standalone_deltas)


def synergy_lower_bound_holds(
    full_auc: float,
    baseline_auc: float,
    standalone_deltas: Sequence[float],
    tol: float = 1.0e-9,
) -> bool:
    return compositional_interaction(full_auc, baseline_auc, standalone_deltas) >= -tol
