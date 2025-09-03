# core/entropy_field.py
from __future__ import annotations
import numpy as np

def _box_blur3(field: np.ndarray) -> np.ndarray:
    """Light 3×3 blur (edge-padded) to add smooth structure without extra deps."""
    k = np.array([[1, 1, 1],
                  [1, 2, 1],
                  [1, 1, 1]], dtype=float)
    k /= k.sum()
    H, W = field.shape
    pad = np.pad(field, 1, mode="edge")
    out = np.empty_like(field)
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(pad[i:i+3, j:j+3] * k)
    return out

def generate_entropy_field(shape=(100, 100), method="gaussian", seed=None, **kwargs):
    """
    Create an entropy exposure field S(x, t).

    Methods:
      - 'gaussian': clipped normal in [0,1]
      - 'uniform' : uniform in [low, high]
      - 'gradient': linear ramp along x or y
      - 'hillvalley': sum of radial bumps (wells/ridges), then normalized to [0,1]
                      args: n_bumps (int), sigma (float in 0..1 of max(H,W))
    """
    rng = np.random.default_rng(seed)
    H, W = shape

    if method == "gaussian":
        mean = kwargs.get("mean", 0.5)
        std = kwargs.get("std", 0.1)
        field = rng.normal(loc=mean, scale=std, size=shape)
        field = np.clip(field, 0.0, 1.0)
        # slight smoothing helps create usable gradients
        field = _box_blur3(field)

    elif method == "uniform":
        low = kwargs.get("low", 0.0)
        high = kwargs.get("high", 1.0)
        field = rng.uniform(low, high, size=shape)
        field = _box_blur3(field)

    elif method == "gradient":
        direction = kwargs.get("direction", "x")
        if direction == "x":
            field = np.tile(np.linspace(0, 1, W), (H, 1))
        else:
            field = np.tile(np.linspace(0, 1, H), (W, 1)).T

    elif method == "hillvalley":
        n = kwargs.get("n_bumps", 6)
        sigma = kwargs.get("sigma", 0.12) * max(H, W)
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        field = np.zeros((H, W), dtype=float)
        for _ in range(n):
            cx = rng.uniform(0, H)
            cy = rng.uniform(0, W)
            amp = rng.uniform(-0.8, 0.8)
            field += amp * np.exp(-((Y - cx)**2 + (X - cy)**2) / (2.0 * sigma**2))
        # normalize safely for NumPy >= 2.0 (use np.ptp, not ndarray.ptp)
        fmin = field.min()
        rng_ptp = np.ptp(field)  # <- NumPy 2.x safe
        field = (field - fmin) / (rng_ptp + 1e-12)
        # add a little blur to soften sharp spikes
        field = _box_blur3(field)
        # mix to avoid extremes; keep entropy in a useful mid-range
        field = 0.6 * field + 0.4 * (1.0 - field)

    else:
        raise ValueError(f"Unsupported entropy field method: {method}")

    return field
