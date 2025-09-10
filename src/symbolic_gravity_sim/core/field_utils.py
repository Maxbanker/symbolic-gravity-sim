# core/field_utils.py
from __future__ import annotations
import numpy as np


def clamp(v, lo, hi):
    return np.minimum(np.maximum(v, lo), hi)


def bilinear_sample(img: np.ndarray, pos):
    H, W = img.shape
    y, x = float(pos[0]), float(pos[1])
    y = clamp(y, 0.0, H - 1.0)
    x = clamp(x, 0.0, W - 1.0)

    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = min(y0 + 1, H - 1)
    x1 = min(x0 + 1, W - 1)
    wy = y - y0
    wx = x - x0
    v00 = img[y0, x0]
    v10 = img[y1, x0]
    v01 = img[y0, x1]
    v11 = img[y1, x1]
    return (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11)


def finite_grad(img: np.ndarray, pos, eps: float = 1e-6):
    y, x = float(pos[0]), float(pos[1])
    dy = bilinear_sample(img, (y + 1, x)) - bilinear_sample(img, (y - 1, x))
    dx = bilinear_sample(img, (y, x + 1)) - bilinear_sample(img, (y, x - 1))
    return np.array([dy, dx]) * 0.5


def box_blur(img: np.ndarray, ksize: int = 3, iters: int = 1) -> np.ndarray:
    if ksize < 2 or iters <= 0:
        return img.copy()
    H, W = img.shape
    r = ksize // 2
    out = img.astype(float).copy()
    for _ in range(iters):
        tmp = np.pad(out, ((0, 0), (r, r)), mode="reflect").cumsum(axis=1)
        out = (tmp[:, 2*r:] - tmp[:, :-2*r]) / (2*r)
        tmp = np.pad(out, ((r, r), (0, 0)), mode="reflect").cumsum(axis=0)
        out = (tmp[2*r:, :] - tmp[:-2*r, :]) / (2*r)
    return out


def laplacian(img: np.ndarray) -> np.ndarray:
    return (
        -4.0 * img
        + np.roll(img, 1, axis=0)
        + np.roll(img, -1, axis=0)
        + np.roll(img, 1, axis=1)
        + np.roll(img, -1, axis=1)
    )
