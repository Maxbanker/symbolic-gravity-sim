"""
psi_eff.py
Computes symbolic efficiency field Ψ_eff from structured information and entropy exposure
"""

import numpy as np

def compute_psi_eff(info_density, entropy_density, focus_weight=None, epsilon=1e-9):
    """
    Compute Ψ_eff = (dI/dS) * W

    Args:
        info_density (np.ndarray): Structured information at each grid point (dI)
        entropy_density (np.ndarray): Entropy exposure at each point (dS)
        focus_weight (np.ndarray, optional): Attention/focus weighting W(x, t)
        epsilon (float): Small constant to avoid division by zero

    Returns:
        psi_eff (np.ndarray): Ψ_eff field
    """
    ratio = info_density / (entropy_density + epsilon)
    if focus_weight is not None:
        psi_eff = ratio * focus_weight
    else:
        psi_eff = ratio
    return psi_eff
