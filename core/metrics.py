# core/metrics.py
from __future__ import annotations
import numpy as np
from .field_utils import finite_grad, laplacian

def curvature(psi: np.ndarray, normalize=True) -> np.ndarray:
    L = laplacian(psi)
    if not normalize:
        return L
    p1, p99 = np.percentile(L, [1, 99])
    L = (L - p1) / (p99 - p1 + 1e-12)
    return np.clip(L, 0, 1)

def drift_pressure(psi: np.ndarray, normalize=True) -> np.ndarray:
    gx, gy = finite_grad(psi)
    D = np.hypot(gx, gy)  # ||∇Ψ||
    if not normalize:
        return D
    p1, p99 = np.percentile(D, [1, 99])
    D = (D - p1) / (p99 - p1 + 1e-12)
    return np.clip(D, 0, 1)

def recursion_density(psi: np.ndarray) -> np.ndarray:
    """
    Heuristic: density of recursive attraction = max(curvature - drift_pressure, 0)
    """
    C = curvature(psi, normalize=True)
    P = drift_pressure(psi, normalize=True)
    return np.clip(C - P, 0, 1)
