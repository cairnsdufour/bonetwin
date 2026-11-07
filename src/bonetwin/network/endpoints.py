"""Primary endpoint and trajectory sampling.

Ref: Sec. III-A, Eq. (1); Sec. III-D bimodal trajectory phenotype.
"""

from __future__ import annotations

import numpy as np

from bonetwin.network.types import FloatArray, IntArray

OSTEOPOROTIC_THRESHOLD = -2.5


def trajectory_grid(horizon_years: float, dt_years: float) -> FloatArray:
    n_steps = int(round(horizon_years / dt_years))
    return (np.arange(n_steps + 1, dtype=np.float32) * dt_years).astype(np.float32)


def sample_trajectory(baseline: FloatArray, delta: FloatArray, grid: FloatArray) -> FloatArray:
    slope = (delta / grid[-1]).astype(np.float32)
    out: FloatArray = (baseline[:, None] + slope[:, None] * grid[None, :]).astype(np.float32)
    return out


def primary_endpoint(trajectory: FloatArray) -> IntArray:
    crossed = np.asarray((trajectory[:, 1:] <= OSTEOPOROTIC_THRESHOLD).any(axis=1))
    return crossed.astype(np.int64)
