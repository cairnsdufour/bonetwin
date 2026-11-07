from __future__ import annotations

import torch

from bonetwin.propagation.marching import integrate, time_grid
from bonetwin.propagation.medium import DiffusionField, DriftField


class _ZeroDiffusion(DiffusionField):
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(z)


def test_zero_diffusion_is_deterministic() -> None:
    torch.manual_seed(0)
    drift = DriftField(8, 3, 16, 4)
    diffusion = _ZeroDiffusion(8, 16, 3)
    z0 = torch.randn(4, 8)
    ctx = torch.randn(4, 3)
    grid = time_grid(5.0, 0.25, torch.device("cpu"))
    a = integrate(drift, diffusion, z0, ctx, grid)
    b = integrate(drift, diffusion, z0, ctx, grid)
    assert torch.allclose(a, b, atol=1e-6)


def test_grid_has_expected_step_count() -> None:
    grid = time_grid(5.0, 0.25, torch.device("cpu"))
    assert grid.numel() == 21
    assert torch.isclose(grid[-1], torch.tensor(5.0))


def test_integration_backpropagates_through_unroll() -> None:
    drift = DriftField(8, 3, 16, 4)
    diffusion = DiffusionField(8, 16, 3)
    z0 = torch.randn(2, 8, requires_grad=True)
    ctx = torch.randn(2, 3)
    grid = time_grid(1.0, 0.25, torch.device("cpu"))
    path = integrate(drift, diffusion, z0, ctx, grid)
    path.sum().backward()
    assert z0.grad is not None
    assert torch.isfinite(z0.grad).all()
