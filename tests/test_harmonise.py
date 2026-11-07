from __future__ import annotations

import torch

from bonetwin.sensors.response import harmonise_volume, piecewise_linear_remap


def test_identity_at_reference_kvp() -> None:
    volume = torch.linspace(-1000.0, 2000.0, 64).reshape(4, 4, 4)
    out = harmonise_volume(volume, kvp=120.0, reference_kvp=120.0)
    assert torch.equal(out, volume)


def test_remap_is_monotone() -> None:
    src = torch.tensor([-1000.0, 0.0, 1000.0, 2000.0])
    dst = torch.tensor([-1000.0, 100.0, 1200.0, 2200.0])
    values = torch.linspace(-1000.0, 2000.0, 200)
    mapped = piecewise_linear_remap(values, src, dst)
    assert torch.all(mapped[1:] - mapped[:-1] >= -1e-5)


def test_harmonise_shifts_bone_window_for_off_reference_kvp() -> None:
    volume = torch.full((2, 2, 2), 800.0)
    out = harmonise_volume(volume, kvp=80.0, reference_kvp=120.0)
    assert out.shape == volume.shape
    assert not torch.equal(out, volume)
