# core/agents.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .field_utils import bilinear_sample, finite_grad
from .metrics import curvature  # kept for potential future use


@dataclass
class AgentConfig:
    dt: float = 1.0
    alpha: float = 0.40
    beta_momentum: float = 0.88
    noise_std: float = 0.001
    orbit_bias: float = 0.55
    warmup: int = 10

    collapse_q: float = 0.15
    curvature_q: float = 0.92
    curvature_consecutive: int = 4
    curvature_hysteresis: float = 0.85

    stagnation_window: int = 12
    stagnation_tol: float = 0.25
    stagnation_std: float = 0.05
    max_age: int = 10_000


class SymbolicAgent:
    def __init__(self, psi: np.ndarray, curv_map: np.ndarray, start_xy, cfg: AgentConfig):
        self.psi = psi
        self.curv = curv_map
        self.cfg = cfg

        self.pos = np.array([float(start_xy[1]), float(start_xy[0])], dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.track = [self.pos.copy()]
        self.alive = True
        self.steps = 0
        self.collapse_reason = None

        H, W = psi.shape
        self.bounds = np.array([H - 1, W - 1], dtype=float)

        self._curv_hits = 0
        self._in_high_curv = False
        self._recent_speeds = []

    def _psi_at(self, pos=None):
        if pos is None:
            pos = self.pos
        return float(bilinear_sample(self.psi, pos))

    def _curv_at(self, pos=None):
        if pos is None:
            pos = self.pos
        return float(bilinear_sample(self.curv, pos))

    def _grad_at(self, pos=None):
        if pos is None:
            pos = self.pos
        return finite_grad(self.psi, pos)

    def _collapse_check(self):
        if not self.alive or self.steps < self.cfg.warmup:
            return False

        if self._psi_at() < self._psi_thresh:
            self.alive = False
            self.collapse_reason = "low_psi"
            return True

        curv_here = self._curv_at()
        if self._in_high_curv:
            if curv_here < self._curv_thresh * self.cfg.curvature_hysteresis:
                self._in_high_curv = False
                self._curv_hits = 0
        else:
            if curv_here >= self._curv_thresh:
                self._in_high_curv = True
                self._curv_hits = 0

        if self._in_high_curv:
            self._curv_hits += 1
            if self._curv_hits >= self.cfg.curvature_consecutive:
                self.alive = False
                self.collapse_reason = "high_curvature"
                return True

        if len(self._recent_speeds) >= self.cfg.stagnation_window:
            arr = np.array(self._recent_speeds[-self.cfg.stagnation_window:])
            if (arr.mean() < self.cfg.stagnation_tol and
                arr.std() < self.cfg.stagnation_std and
                self.steps > self.cfg.warmup and
                self.steps > self.cfg.max_age):
                self.alive = False
                self.collapse_reason = "stagnation"
                return True
        return False

    def _keep_in_bounds(self):
        self.pos = np.minimum(np.maximum(self.pos, 0.0), self.bounds)

    def prime_thresholds(self):
        self._curv_thresh = np.quantile(self.curv, self.cfg.curvature_q)
        self._psi_thresh = np.quantile(self.psi, self.cfg.collapse_q)

    def step(self):
        if not self.alive:
            return
        g = self._grad_at()

        force = self.cfg.alpha * g

        R = np.array([[0, -1], [1, 0]], dtype=float)
        curv_here = self._curv_at()
        lam = self.cfg.orbit_bias * (curv_here / (curv_here + 1e-6))
        force = force + lam * (R @ g)

        self.vel = self.cfg.beta_momentum * self.vel + force
        self.vel += np.random.randn(2) * self.cfg.noise_std
        self.pos = self.pos + self.cfg.dt * self.vel
        self._keep_in_bounds()

        self.steps += 1
        spd = float(np.linalg.norm(self.cfg.dt * self.vel))
        self._recent_speeds.append(spd)
        self.track.append(self.pos.copy())

        self._collapse_check()
