# core/psi_eff.py
from __future__ import annotations
import numpy as np

def compute_psi_eff(info_density, entropy_density, focus_weight=None, mode="log_ratio"):
    """
    Ψ_eff ~ information advantage under entropy exposure with optional attentional gain.

    mode: 'ratio' | 'log_ratio' (recommended)
    """
    I = np.asarray(info_density, dtype=float)
    S = np.asarray(entropy_density, dtype=float)
    eps = 10 * np.finfo(I.dtype).eps

    if mode == "ratio":
        psi = I / (S + eps)
    elif mode == "log_ratio":
        psi = np.log(I + eps) - np.log(S + eps)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if focus_weight is not None:
        W = np.asarray(focus_weight, dtype=float)
        if W.shape != psi.shape:
            raise ValueError("focus_weight shape mismatch")
        W = (W - W.min()) / (W.ptp() + eps)
        psi = psi * (0.5 + 0.5 * W)

    # robust normalization to [0,1] for thresholds/plotting
    p1, p99 = np.percentile(psi, [1, 99])
    psi = (psi - p1) / (p99 - p1 + eps)
    return np.clip(psi, 0.0, 1.0)
