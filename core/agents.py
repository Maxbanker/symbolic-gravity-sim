# core/agents.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .field_utils import bilinear_sample, finite_grad
from .metrics import curvature

@dataclass
class AgentConfig:
    dt: float = 1.0
    alpha: float = 0.35          # step gain
    beta_momentum: float = 0.85  # momentum
    noise_std: float = 0.0
    boundary: str = "reflect"    # 'clip'|'wrap'|'reflect'
    stagnation_tol: float = 2e-3
    stagnation_steps: int = 10
    collapse_q: float = 0.07     # collapse when Ψ below this quantile
    curvature_thresh: float = 0.18

class SymbolicAgent:
    """
    Continuous agent: x_{t+1} = x_t + dt*(β v_t + α*(-∇Ψ)) + noise
    composite collapse: low-Ψ OR high curvature OR stagnation
    """
    def __init__(self, position, psi_field, cfg: AgentConfig = AgentConfig()):
        self.pos = np.array(position, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.history: list[tuple[float, float]] = [tuple(self.pos)]
        self.cfg = cfg
        self.collapsed = False

        self.psi = psi_field
        self.gx, self.gy = finite_grad(self.psi)
        self.psi_threshold = np.quantile(self.psi, self.cfg.collapse_q)
        self.curv = curvature(self.psi, normalize=True)

    def _apply_boundary(self, p: np.ndarray) -> np.ndarray:
        H, W = self.psi.shape
        mode = self.cfg.boundary
        if mode == "clip":
            return np.clip(p, [0,0], [H-1, W-1])
        if mode == "wrap":
            return np.array([p[0] % H, p[1] % W])
        if mode == "reflect":
            def refl(v, L):
                q = np.abs(v) % (2*(L-1))
                return q if q <= (L-1) else 2*(L-1)-q
            return np.array([refl(p[0], H), refl(p[1], W)])
        raise ValueError(f"Unknown boundary: {mode}")

    def _collapse_check(self, pos: np.ndarray) -> bool:
        psi_here = bilinear_sample(self.psi, pos)
        curv_here = bilinear_sample(self.curv, pos)
        return (psi_here <= self.psi_threshold) or (curv_here >= self.cfg.curvature_thresh)

    def step(self, rng: np.random.Generator | None = None):
        if self.collapsed: return

        grad_here = np.array([
            bilinear_sample(self.gx, self.pos),
            bilinear_sample(self.gy, self.pos)
        ])
        force = -grad_here

        self.vel = self.cfg.beta_momentum * self.vel + self.cfg.alpha * force
        if rng is not None and self.cfg.noise_std > 0:
            self.vel += self.cfg.noise_std * rng.normal(size=2)

        self.pos = self.pos + self.cfg.dt * self.vel
        self.pos = self._apply_boundary(self.pos)
        self.history.append(tuple(self.pos))

        if len(self.history) > self.cfg.stagnation_steps:
            recent = np.diff(np.array(self.history[-self.cfg.stagnation_steps:]), axis=0)
            if np.linalg.norm(recent, axis=1).mean() < self.cfg.stagnation_tol:
                self.collapsed = True
                return

        if self._collapse_check(self.pos):
            self.collapsed = True
