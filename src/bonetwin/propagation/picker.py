"""Transition decoder h_psi mapping the latent state to a T-score.

Ref: Sec. III-C (decoder h_psi, two layers x 256, T_hat(t)=h_psi(z(t))).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class TransitionDecoder(nn.Module):
    def __init__(self, latent_dim: int, width: int = 256, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(latent_dim, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(z).squeeze(-1))
