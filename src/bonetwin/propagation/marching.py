"""Euler-Maruyama integration of the latent SDE.

Ref: Sec. III-C (Euler-Maruyama, dt=0.25 yr, 5-yr horizon, N_MC=128 paths;
ELBO gradients flow by direct backprop through the unroll).
"""

from __future__ import annotations

import math

import torch

from bonetwin.propagation.medium import DiffusionField, DriftField


def time_grid(horizon: float, dt: float, device: torch.device) -> torch.Tensor:
    n_steps = int(round(horizon / dt))
    return torch.arange(n_steps + 1, device=device, dtype=torch.float32) * dt


def integrate(
    drift: DriftField,
    diffusion: DiffusionField,
    z0: torch.Tensor,
    context: torch.Tensor,
    grid: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    dt = float(grid[1] - grid[0])
    sqrt_dt = math.sqrt(dt)
    z = z0
    states = [z]
    for k in range(grid.numel() - 1):
        t = grid[k]
        mu = drift(z, t, context)
        sigma = diffusion(z, t)
        noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
        z = z + mu * dt + sigma * sqrt_dt * noise
        states.append(z)
    return torch.stack(states, dim=1)


def monte_carlo_paths(
    drift: DriftField,
    diffusion: DiffusionField,
    z0: torch.Tensor,
    context: torch.Tensor,
    grid: torch.Tensor,
    n_mc: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    batch = z0.shape[0]
    z0_rep = z0.repeat_interleave(n_mc, dim=0)
    ctx_rep = context.repeat_interleave(n_mc, dim=0)
    paths = integrate(drift, diffusion, z0_rep, ctx_rep, grid, generator=generator)
    return paths.reshape(batch, n_mc, grid.numel(), z0.shape[-1])
