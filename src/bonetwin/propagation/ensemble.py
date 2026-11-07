"""Monte-Carlo 5-year transition probability.

Ref: Sec. III-C, Alg. 1 L5-6: p_hat(5) = N_MC^-1 sum_m 1[h_psi(z^(m)(5)) <= -2.5].
"""

from __future__ import annotations

import torch


def transition_probability(tscore_at_horizon: torch.Tensor, threshold: float) -> torch.Tensor:
    crossed = (tscore_at_horizon <= threshold).float()
    return crossed.mean(dim=1)


def soft_transition_probability(
    tscore_at_horizon: torch.Tensor, threshold: float, temperature: float = 0.1
) -> torch.Tensor:
    crossed = torch.sigmoid((threshold - tscore_at_horizon) / temperature)
    return crossed.mean(dim=1)
