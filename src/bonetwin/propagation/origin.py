"""Amortised variational posterior q_phi(z(0) | x).

Ref: Sec. III-C (variational posterior q_phi(z(0)|x)); Fig. 3 (centre).
"""

from __future__ import annotations

import torch
from torch import nn


class GaussianPosterior(nn.Module):
    def __init__(self, embed_dim: int, latent_dim: int, hidden: int = 1024) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mean = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(embedding)
        logvar = self.logvar(h).clamp(-8.0, 8.0)
        return self.mean(h), logvar

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)
