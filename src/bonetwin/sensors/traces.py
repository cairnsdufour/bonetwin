"""Volume materialisation and batching for the encode step.

Ref: Sec. III-A stratifier g=(sex, age, vendor); Sec. III-B 96^3 ROI input.
"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch.utils.data import Dataset

from bonetwin.network.types import Cohort

N_VENDORS = 4


class Sample(TypedDict):
    volume: torch.Tensor
    context: torch.Tensor
    baseline: torch.Tensor
    delta: torch.Tensor
    endpoint: torch.Tensor
    phenotype: torch.Tensor
    stratum: torch.Tensor


class Batch(TypedDict):
    volume: torch.Tensor
    context: torch.Tensor
    baseline: torch.Tensor
    delta: torch.Tensor
    endpoint: torch.Tensor
    phenotype: torch.Tensor
    stratum: torch.Tensor


class BoneVolumes(Dataset[Sample]):
    def __init__(
        self, cohort: Cohort, volume_size: int, volumes: torch.Tensor | None = None
    ) -> None:
        if volumes is None:
            raise ValueError("CT volumes must be supplied for each patient.")
        expected_shape = (len(cohort), 1, volume_size, volume_size, volume_size)
        if volumes.shape != expected_shape:
            raise ValueError("CT volume dimensions do not match the configured cohort.")
        self.cohort = cohort
        self.volumes = volumes
        self._stratum = cohort.stratum_key()

    def __len__(self) -> int:
        return len(self.cohort)

    def __getitem__(self, index: int) -> Sample:
        c = self.cohort
        context = torch.tensor(
            [
                float(c.sex[index]),
                (float(c.age[index]) - 70.0) / 10.0,
                float(c.vendor[index]) / N_VENDORS,
            ],
            dtype=torch.float32,
        )
        return Sample(
            volume=self.volumes[index],
            context=context,
            baseline=torch.tensor(float(c.baseline_tscore[index]), dtype=torch.float32),
            delta=torch.tensor(float(c.delta_tscore[index]), dtype=torch.float32),
            endpoint=torch.tensor(int(c.endpoint[index]), dtype=torch.long),
            phenotype=torch.tensor(int(c.phenotype[index]), dtype=torch.long),
            stratum=torch.tensor(int(self._stratum[index]), dtype=torch.long),
        )


def collate(samples: list[Sample]) -> Batch:
    return Batch(
        volume=torch.stack([s["volume"] for s in samples]),
        context=torch.stack([s["context"] for s in samples]),
        baseline=torch.stack([s["baseline"] for s in samples]),
        delta=torch.stack([s["delta"] for s in samples]),
        endpoint=torch.stack([s["endpoint"] for s in samples]),
        phenotype=torch.stack([s["phenotype"] for s in samples]),
        stratum=torch.stack([s["stratum"] for s in samples]),
    )
