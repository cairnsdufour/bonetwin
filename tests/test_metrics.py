from __future__ import annotations

import numpy as np

from bonetwin.residuals.discrimination import auc_score


def test_auc_perfect_separation() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    assert auc_score(y, scores) == 1.0
