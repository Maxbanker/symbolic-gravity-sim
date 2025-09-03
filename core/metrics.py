# core/field_utils.py
from __future__ import annotations
import numpy as np

def bilinear_sample(field: np.ndarray, pos: np.ndarray) -> float:
    H, W = field.shape
    x, y = pos
    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    x1 = min(x0 + 1, H - 1); y1 = min(y0 + 1, W - 1)
    dx = x - x0; dy = y - y0
    v00 = field[x0, y0]; v01 = field[x0, y1]
    v10 = field[x1, y0]; v11 = field[x1, y1]
    return (v00*(1-dx)*(1-dy) + v01*(1-dx)*dy +
            v10*dx*(1-dy) + v11*dx*dy)

def finite_grad(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # axis0=row(x), axis1=col(y)
    return np.gradient(field)

def laplacian(field: np.ndarray) -> np.ndarray:
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float)
    H, W = field.shape
    pad = np.pad(field, 1, mode='edge')
    out = np.empty_like(field)
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(pad[i:i+3, j:j+3] * k)
    return out
