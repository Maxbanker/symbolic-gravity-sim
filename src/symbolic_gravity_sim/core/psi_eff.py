# core/psi_eff.py
from __future__ import annotations
import numpy as np


def compute_psi_eff(
    info_density: np.ndarray,
    entropy_density: np.ndarray,
    mode: str = "log_ratio",
    focus_weight: np.ndarray | None = None,
    p1: float = 1.0,
    p99: float = 99.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute an effective Ψ field from information and entropy densities.
    Modes:
      - "ratio":      info / (entropy + 1)
      - "log_ratio":  log((info + eps) / (entropy + eps))   [default]
      - "difference": info - entropy
    Result is robust-normalized to [0,1] using percentiles p1/p99.
    """
    info = np.asarray(info_density, dtype=float)
    ent = np.asarray(entropy_density, dtype=float)

    if mode == "ratio":
        psi = info / (ent + 1.0)
    elif mode == "log_ratio":
        psi = np.log((info + eps) / (ent + eps))
    elif mode == "difference":
        psi = info - ent
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if focus_weight is not None:
        W = np.asarray(focus_weight, dtype=float)
        if W.shape != psi.shape:
            raise ValueError("focus_weight shape mismatch")
        W = (W - W.min()) / (np.ptp(W) + eps)
        psi = psi * (0.5 + 0.5 * W)

    p1v, p99v = np.percentile(psi, [p1, p99])
    psi = (psi - p1v) / (p99v - p1v + eps)
    return np.clip(psi, 0.0, 1.0)
