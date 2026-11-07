"""End-to-end orchestration: encode -> train -> calibrate -> triage.

Ref: Algorithm 1 (BoneTwin: encode-train-calibrate-triage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader

from bonetwin.alerts.levels import triage
from bonetwin.alerts.thresholds import StratifiedConformal
from bonetwin.network.types import Cohort
from bonetwin.propagation.station import BoneTwin
from bonetwin.recorder.schema import Config
from bonetwin.recorder.telemetry import resolve_device
from bonetwin.sensors.traces import Batch, BoneVolumes, collate

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass
class CohortPrediction:
    p_hat: FloatArray
    endpoint: IntArray
    stratum: IntArray


@dataclass
class TriageResult:
    p_hat: FloatArray
    lo: FloatArray
    hi: FloatArray
    bucket: IntArray


def make_loader(
    cohort: Cohort, volume_size: int, batch_size: int, shuffle: bool
) -> DataLoader[Batch]:
    return DataLoader(
        BoneVolumes(cohort, volume_size),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )


@torch.no_grad()
def predict_cohort(
    model: BoneTwin, loader: DataLoader[Batch], device: torch.device, n_mc: int
) -> CohortPrediction:
    model.eval()
    probs: list[float] = []
    endpoints: list[int] = []
    strata: list[int] = []
    for batch in loader:
        p5, _ = model.predict(batch["volume"].to(device), batch["context"].to(device), n_mc)
        probs.extend(p5.cpu().tolist())
        endpoints.extend(batch["endpoint"].tolist())
        strata.extend(batch["stratum"].tolist())
    return CohortPrediction(
        p_hat=np.asarray(probs, dtype=np.float64),
        endpoint=np.asarray(endpoints, dtype=np.int64),
        stratum=np.asarray(strata, dtype=np.int64),
    )


class BoneTwinPipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = resolve_device()
        self.model = BoneTwin(cfg.sensors, cfg.propagation)
        self.conformal: StratifiedConformal | None = None

    def run(self) -> dict[str, object]:
        raise NotImplementedError(
            "CT data, longitudinal labels, and model weights must be integrated."
        )

    def triage(self, cohort: Cohort, alpha: float = 0.05) -> TriageResult:
        if self.conformal is None:
            raise RuntimeError("pipeline must be run before triage")
        loader = make_loader(
            cohort, self.cfg.sensors.volume_size, self.cfg.optim.batch_size, False
        )
        pred = predict_cohort(self.model, loader, self.device, self.cfg.propagation.n_mc)
        lo, hi = self.conformal.interval(pred.p_hat, pred.stratum, alpha)
        bucket = triage(lo, hi, self.cfg.alerts.tau_low, self.cfg.alerts.tau_high)
        return TriageResult(p_hat=pred.p_hat, lo=lo, hi=hi, bucket=bucket)
