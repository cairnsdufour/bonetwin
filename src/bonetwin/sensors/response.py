"""HU harmonisation to a 120 kVp reference via monotone piecewise-linear knots.

Ref: Sec. III-B (HU harmonisation, piecewise-linear knots, 120 kVp reference).
"""

from __future__ import annotations

import torch

_REFERENCE_KNOTS = torch.tensor([-1000.0, -100.0, 0.0, 300.0, 1000.0, 2000.0])


def piecewise_linear_remap(
    values: torch.Tensor, src_knots: torch.Tensor, dst_knots: torch.Tensor
) -> torch.Tensor:
    flat = values.reshape(-1)
    idx = torch.searchsorted(src_knots, flat, right=True).clamp(1, src_knots.numel() - 1)
    left = src_knots[idx - 1]
    right = src_knots[idx]
    frac = (flat - left) / (right - left).clamp_min(1e-6)
    mapped = dst_knots[idx - 1] + frac * (dst_knots[idx] - dst_knots[idx - 1])
    return mapped.reshape(values.shape)


def _kvp_knots(kvp: float, reference_kvp: float) -> torch.Tensor:
    gain = (reference_kvp / kvp) ** 0.5
    shift = _REFERENCE_KNOTS.clone()
    bone = shift > 0
    shift[bone] = shift[bone] * gain
    return shift


def harmonise_volume(
    volume: torch.Tensor, kvp: float, reference_kvp: float = 120.0
) -> torch.Tensor:
    if abs(kvp - reference_kvp) < 1e-6:
        return volume
    src = _kvp_knots(kvp, reference_kvp)
    return piecewise_linear_remap(volume, src, _REFERENCE_KNOTS.to(volume.dtype))
