"""Latent SDE drift and diffusion fields.

Ref: Sec. III-C, Eq. (2). Drift mu_theta is a 4-layer FiLM-conditioned MLP of
width 2048; diffusion Sigma_phi is a 3-layer MLP of width 1024 with softplus
output giving a diagonal, developmental-pathway-conditioned noise scale.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


class FiLM(nn.Module):
    def __init__(self, context_dim: int, width: int) -> None:
        super().__init__()
        self.to_scale = nn.Linear(context_dim, width)
        self.to_shift = nn.Linear(context_dim, width)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor, (1.0 + self.to_scale(context)) * hidden + self.to_shift(context)
        )


class DriftField(nn.Module):
    def __init__(self, latent_dim: int, context_dim: int, width: int, depth: int) -> None:
        super().__init__()
        self.input = nn.Linear(latent_dim + 1, width)
        self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.film = nn.ModuleList(FiLM(context_dim, width) for _ in range(depth))
        self.output = nn.Linear(width, latent_dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        t_feat = t.expand(z.shape[0], 1)
        h = F.gelu(self.film[0](self.input(torch.cat([z, t_feat], dim=-1)), context))
        for layer, film in zip(self.hidden, self.film[1:], strict=True):
            h = F.gelu(film(layer(h), context))
        return cast(torch.Tensor, self.output(h))


class DiffusionField(nn.Module):
    def __init__(self, latent_dim: int, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(latent_dim, width), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers.append(nn.Linear(width, latent_dim))
        self.net = nn.Sequential(*layers)
        self.floor = 1.0e-3

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(z)) + self.floor
