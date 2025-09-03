# core/entropy_field.py
from __future__ import annotations
import numpy as np

def generate_entropy_field(shape=(100, 100), method="gaussian", seed=None, **kwargs):
    """
    Create an entropy exposure field S(x, t).

    Methods: 'gaussian', 'uniform', 'gradient', 'hillvalley'
    """
    rng = np.random.default_rng(seed)
    H, W = shape

    if method == "gaussian":
        mean = kwargs.get("mean", 0.5)
        std = kwargs.get("std", 0.1)
        field = rng.normal(loc=mean, scale=std, size=shape)
        field = np.clip(field, 0, 1)

    elif method == "uniform":
        low = kwargs.get("low", 0.0); high = kwargs.get("high", 1.0)
        field = rng.uniform(low, high, size=shape)

    elif method == "gradient":
        direction = kwargs.get("direction", "x")
        if direction == "x":
            field = np.tile(np.linspace(0, 1, W), (H, 1))
        else:
            field = np.tile(np.linspace(0, 1, H), (W, 1)).T

    elif method == "hillvalley":
        # mixture of radial bumps to create wells and ridges
        n = kwargs.get("n_bumps", 6)
        xs = rng.uniform(0, H, size=n)
        ys = rng.uniform(0, W, size=n)
        sig = kwargs.get("sigma", 0.12) * max(H, W)
        field = np.zeros(shape, dtype=float)
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        for i in range(n):
            amp = rng.uniform(-0.8, 0.8)
            field += amp * np.exp(-((Y - xs[i])**2 + (X - ys[i])**2) / (2 * sig**2))
        field = (field - field.min()) / (field.ptp() + 1e-12)
        # invert half the map to keep “entropy” high in some regions
        field = 0.6 * field + 0.4 * (1 - field)

    else:
        raise ValueError(f"Unsupported entropy field method: {method}")

    return field
