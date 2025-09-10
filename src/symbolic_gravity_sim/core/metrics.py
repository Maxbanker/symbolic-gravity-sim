# core/metrics.py
from __future__ import annotations
import numpy as np
from .field_utils import laplacian


def curvature(psi: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(psi.astype(float))
    gmag = np.hypot(gy, gx)
    lap = laplacian(psi.astype(float))
    c = np.hypot(lap, 0.5 * gmag)
    p1, p99 = np.percentile(c, [1, 99])
    c = (c - p1) / (p99 - p1 + 1e-8)
    return np.clip(c, 0.0, 1.0)
