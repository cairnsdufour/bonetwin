from __future__ import annotations

import math

import numpy as np

from bonetwin.network.endpoints import primary_endpoint, sample_trajectory, trajectory_grid
from bonetwin.swarms.benefit import number_needed_to_treat


def test_endpoint_fires_when_trajectory_crosses_threshold() -> None:
    grid = trajectory_grid(5.0, 0.25)
    baseline = np.array([-1.0, -2.4], dtype=np.float32)
    delta = np.array([-0.2, -0.5], dtype=np.float32)
    traj = sample_trajectory(baseline, delta, grid)
    endpoint = primary_endpoint(traj)
    assert endpoint.tolist() == [0, 1]


def test_nnt_matches_published_values() -> None:
    assert math.isclose(number_needed_to_treat(0.621, 0.35), 4.6, abs_tol=0.1)
    assert math.isclose(number_needed_to_treat(0.116, 0.35), 24.6, abs_tol=0.3)
