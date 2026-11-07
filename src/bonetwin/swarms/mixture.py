"""Bimodal trajectory phenotype via BIC-selected Gaussian mixture.

Ref: Sec. III-D, Sec. IV-E, Fig. 1 G-I. Two-component decomposition of the
predicted 5-year increment Delta T_hat: slow-decline (~69%) vs fast-decline
(~31%); K is selected by BIC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.mixture import GaussianMixture

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PhenotypeFit:
    n_components: int
    bic: dict[int, float]
    weights: FloatArray
    means: FloatArray
    labels: npt.NDArray[np.int64]
    fast_index: int

    @property
    def fast_fraction(self) -> float:
        return float(self.weights[self.fast_index])

    @property
    def slow_fraction(self) -> float:
        return float(1.0 - self.weights[self.fast_index])


def select_components(delta: FloatArray, max_components: int, seed: int = 0) -> int:
    x = delta.reshape(-1, 1)
    scores: dict[int, float] = {}
    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
        gmm.fit(x)
        scores[k] = float(gmm.bic(x))
    return min(scores, key=lambda k: scores[k])


def fit_phenotype(delta: FloatArray, max_components: int = 5, seed: int = 0) -> PhenotypeFit:
    x = delta.reshape(-1, 1)
    bic: dict[int, float] = {}
    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
        gmm.fit(x)
        bic[k] = float(gmm.bic(x))
    best_k = min(bic, key=lambda k: bic[k])
    model = GaussianMixture(n_components=best_k, covariance_type="full", random_state=seed)
    labels = model.fit_predict(x)
    means = model.means_.reshape(-1)
    fast_index = int(np.argmin(means))
    return PhenotypeFit(
        n_components=best_k,
        bic=bic,
        weights=model.weights_.astype(np.float64),
        means=means.astype(np.float64),
        labels=labels.astype(np.int64),
        fast_index=fast_index,
    )
