"""Dual foundation-model towers producing the 1024-D bone state.

Ref: Sec. III-B, Fig. 3 (left). Merlin is the primary 1024-D encoder; CT-FM is
the secondary tower with a 512->1024 linear adapter onto the shared space.

The pretrained Merlin and CT-FM weights are not included or connected. The current
convolutional encoders use fixed initial weights and require independent validation.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from bonetwin.recorder.schema import SensorConfig


def _frozen_init(module: nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    for param in module.parameters():
        with torch.no_grad():
            param.copy_(torch.empty_like(param).normal_(0.0, 0.02, generator=generator))
        param.requires_grad_(False)
    module.eval()


class _ConvTower(nn.Module):
    def __init__(self, out_dim: int, seed: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Linear(64, out_dim)
        _frozen_init(self, seed)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        pooled = self.stem(volume).flatten(1)
        return cast(torch.Tensor, self.head(pooled))


class DualTowerEncoder(nn.Module):
    def __init__(self, cfg: SensorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.merlin = _ConvTower(cfg.merlin_dim, seed=20260421)
        self.ctfm = _ConvTower(cfg.ctfm_dim, seed=20260422)
        self.ctfm_adapter = nn.Linear(cfg.ctfm_dim, cfg.latent_dim)
        _frozen_init(self.ctfm_adapter, seed=20260423)
        self.norm = nn.LayerNorm(cfg.latent_dim)
        _frozen_init(self.norm, seed=20260424)

    @torch.no_grad()
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        primary = self.merlin(volume)
        if not self.cfg.use_ctfm:
            return cast(torch.Tensor, self.norm(primary))
        secondary = self.ctfm_adapter(self.ctfm(volume))
        return cast(torch.Tensor, self.norm(0.5 * (primary + secondary)))

    @torch.no_grad()
    def encode_secondary(self, volume: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.norm(self.ctfm_adapter(self.ctfm(volume))))
