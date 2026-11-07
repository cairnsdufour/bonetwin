"""Latent-SDE evidence lower bound.

Ref: Sec. III-C, Fig. 3 (ELBO = E_q[log p(T_hat(t)|z(0:5))] - beta_KL KL(q||p),
with beta_KL annealed over training).
"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch.nn import functional as F


class ElboTerms(TypedDict):
    loss: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor
    beta: float


def gaussian_kl(
    mean: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0
) -> torch.Tensor:
    per_dim = 0.5 * (torch.exp(logvar) + mean.pow(2) - 1.0 - logvar)
    if free_bits > 0.0:
        per_dim = torch.clamp(per_dim, min=free_bits)
    return per_dim.sum(dim=-1)


def kl_weight(epoch: int, warmup_epochs: int, beta_max: float) -> float:
    if warmup_epochs <= 0:
        return beta_max
    return beta_max * min(1.0, (epoch + 1) / warmup_epochs)


def elbo(
    tscore_path: torch.Tensor,
    target_path: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    free_bits: float = 0.0,
    observation_sd: float = 0.25,
) -> ElboTerms:
    var = observation_sd**2
    nll = 0.5 * F.mse_loss(tscore_path, target_path, reduction="none") / var
    reconstruction = nll.sum(dim=-1).mean()
    kl = gaussian_kl(mean, logvar, free_bits).mean()
    loss = reconstruction + beta * kl
    return ElboTerms(loss=loss, reconstruction=reconstruction, kl=kl, beta=beta)
