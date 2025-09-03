# core/entropy_field.py
from __future__ import annotations
import numpy as np


def _mix_of_bumps(shape, n_bumps: int = 6, sigma: float = 0.12, seed=None) -> np.ndarray:
    """Create a normalized field built from random Gaussian bumps."""
    H, W = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    field = np.zeros((H, W), dtype=float)
    for _ in range(n_bumps):
        cx, cy = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
        amp = rng.uniform(0.6, 1.0) * (1.0 if rng.random() < 0.5 else -1.0)
        sx = sigma * rng.uniform(0.6, 1.4)
        sy = sigma * rng.uniform(0.6, 1.4)
        field += amp * np.exp(-((xx - cx) ** 2 / (2 * sx**2) + (yy - cy) ** 2 / (2 * sy**2)))
    field -= field.min()
    field /= (np.ptp(field) + 1e-8)
    return field


def generate_entropy_field(
    shape: tuple[int, int] = (120, 160),
    method: str = "hillvalley",
    mean: float = 0.5,
    std: float = 0.12,
    seed=None,
    n_bumps: int = 8,
    sigma: float = 0.10,
) -> np.ndarray:
    """
    Produce a synthetic entropy field ∈ [0,1] for the simulation.
      methods: 'hillvalley' | 'gaussian' | 'uniform' | 'gradient'
    """
    H, W = shape
    rng = np.random.default_rng(seed)

    if method == "hillvalley":
        base = _mix_of_bumps(shape, n_bumps=n_bumps, sigma=sigma, seed=seed)
        yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
        tilt = 0.15 * (xx - 0.5) + 0.15 * (yy - 0.5)
        ent = np.clip(0.6 * base + 0.4 * (tilt - tilt.min()) / (np.ptp(tilt) + 1e-8), 0.0, 1.0)
    elif method == "gaussian":
        ent = rng.normal(loc=mean, scale=std, size=(H, W))
        ent = (ent - ent.min()) / (np.ptp(ent) + 1e-8)
    elif method == "uniform":
        ent = rng.uniform(0.0, 1.0, size=(H, W))
    elif method == "gradient":
        yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
        ent = 0.5 * xx + 0.5 * yy
    else:
        raise ValueError(f"Unknown method: {method}")

    return np.clip(ent, 0.0, 1.0)
