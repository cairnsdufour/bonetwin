from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Cohort:
    baseline_tscore: FloatArray
    delta_tscore: FloatArray
    trajectory: FloatArray
    endpoint: IntArray
    phenotype: IntArray
    sex: IntArray
    age: FloatArray
    vendor: IntArray
    descriptor: FloatArray

    def __len__(self) -> int:
        return int(self.baseline_tscore.shape[0])

    def stratum_key(self) -> IntArray:
        age_band = np.clip(((self.age - 50.0) // 10.0).astype(np.int64), 0, 3)
        key = (self.sex * 16 + age_band * 4 + self.vendor).astype(np.int64)
        return key
