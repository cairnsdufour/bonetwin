"""Assembled BoneTwin: encode -> latent SDE -> decode.

Ref: Sec. III-B/III-C; Fig. 3. The dual-tower encoder is frozen; the posterior,
drift, diffusion, and decoder are the trainable digital-twin dynamics.
"""

from __future__ import annotations

from typing import TypedDict, cast

import torch
from torch import nn

from bonetwin.propagation.ensemble import transition_probability
from bonetwin.propagation.marching import integrate, time_grid
from bonetwin.propagation.medium import DiffusionField, DriftField
from bonetwin.propagation.origin import GaussianPosterior
from bonetwin.propagation.picker import TransitionDecoder
from bonetwin.recorder.schema import PropagationConfig, SensorConfig
from bonetwin.sensors.seismometers import DualTowerEncoder


class TwinOutput(TypedDict):
    tscore_path: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor


class BoneTwin(nn.Module):
    def __init__(self, sensors: SensorConfig, propagation: PropagationConfig) -> None:
        super().__init__()
        self.cfg = propagation
        self.encoder = DualTowerEncoder(sensors)
        self.posterior = GaussianPosterior(sensors.latent_dim, propagation.latent_dim)
        self.drift = DriftField(
            propagation.latent_dim,
            propagation.context_dim,
            propagation.drift_width,
            propagation.drift_depth,
        )
        self.diffusion = DiffusionField(
            propagation.latent_dim, propagation.diffusion_width, propagation.diffusion_depth
        )
        self.decoder = TransitionDecoder(
            propagation.latent_dim, propagation.decoder_width, propagation.decoder_depth
        )
        self.register_buffer(
            "_grid",
            time_grid(propagation.horizon_years, propagation.dt_years, torch.device("cpu")),
        )

    @property
    def grid(self) -> torch.Tensor:
        return cast(torch.Tensor, self._grid)

    def encode(self, volume: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.encoder(volume))

    def forward(self, volume: torch.Tensor, context: torch.Tensor) -> TwinOutput:
        embedding = self.encode(volume)
        mean, logvar = self.posterior(embedding)
        z0 = GaussianPosterior.reparameterize(mean, logvar)
        grid = self.grid.to(z0.device)
        path = integrate(self.drift, self.diffusion, z0, context, grid)
        tscore = self.decoder(path.reshape(-1, path.shape[-1])).reshape(path.shape[:2])
        return TwinOutput(tscore_path=tscore, mean=mean, logvar=logvar)

    @torch.no_grad()
    def predict(
        self, volume: torch.Tensor, context: torch.Tensor, n_mc: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        paths = n_mc if n_mc is not None else self.cfg.n_mc
        embedding = self.encode(volume)
        mean, logvar = self.posterior(embedding)
        std = torch.exp(0.5 * logvar)
        batch = mean.shape[0]
        mean_rep = mean.repeat_interleave(paths, dim=0)
        std_rep = std.repeat_interleave(paths, dim=0)
        ctx_rep = context.repeat_interleave(paths, dim=0)
        z0 = mean_rep + std_rep * torch.randn_like(std_rep)
        grid = self.grid.to(z0.device)
        path = integrate(self.drift, self.diffusion, z0, ctx_rep, grid)
        tscore = self.decoder(path[:, -1, :]).reshape(batch, paths)
        p5 = transition_probability(tscore, self.cfg.transition_threshold)
        return p5, tscore
