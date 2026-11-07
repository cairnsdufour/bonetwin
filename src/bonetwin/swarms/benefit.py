"""Number-needed-to-treat from per-phenotype transition rates.

Ref: Sec. III-D, Sec. IV-E. NNT = 1 / (transition_rate x RRR); the published
fast/slow split is 62.1% / 11.6% under a 35% bisphosphonate RRR, giving
4.6 vs 24.6.
"""

from __future__ import annotations


def number_needed_to_treat(transition_rate: float, relative_risk_reduction: float) -> float:
    denominator = transition_rate * relative_risk_reduction
    if denominator <= 0.0:
        return float("inf")
    return 1.0 / denominator
