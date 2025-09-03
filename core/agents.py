# core/agents.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .field_utils import bilinear_sample, finite_grad
from .metrics import curvature


@dataclass
class AgentConfig:
    dt: float = 1.0
    alpha: float = 0.40                 # base gain
    beta_momentum: float = 0.88         # momentum
    noise_std: float = 0.001            # tiny exploratory noise
    boundary: str = "reflect"           # 'clip' | 'wrap' | 'reflect'

    # stagnation guard
    stagnation_tol: float = 5e-3
    stagnation_steps: int = 14

    # collapse guards
    min_steps_before_collapse: int = 12
    collapse_q: float = 0.02            # low-Ψ quantile triggers collapse

    # curvature collapse (use quantile + consecutive hits)
    curvature_mode: str = "quantile"    # 'abs' or 'quantile'
    curvature_thresh: float = 0.35      # used if mode == 'abs' (0..1 after normalize)
    curvature_q: float = 0.92           # used if mode == 'quantile'
    curvature_consecutive: int = 6      # require N consecutive high-curv steps

    # dynamics quality
    orbit_bias: float = 0.55            # 0..1, tangential drift (0=straight descent)
    max_force: float = 2.5              # clip force norm
    adapt_force: bool = True            # normalize large forces


class SymbolicAgent:
    """
    Continuous agent with momentum and orbit bias.
    Force = -∇Ψ  +  λ * R * ∇Ψ   (R rotates by +90°)
    Composite collapse:
      - low Ψ (below collapse_q quantile)
      - OR high curvature (above threshold/quantile) for 'curvature_consecutive' steps
      - OR stagnation
    """
    def __init__(self, position, psi_field: np.ndarray, cfg: AgentConfig = AgentConfig()):
        self.pos = np.array(position, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.history: list[tuple[float, float]] = [tuple(self.pos)]
        self.cfg = cfg
        self.steps = 0
        self.collapsed = False
        self.collapse_reason: str | None = None

        # static fields
        self.psi = psi_field
        self.gx, self.gy = finite_grad(self.psi)
        self.curv = curvature(self.psi, normalize=True)

        # thresholds
        self.psi_threshold = np.quantile(self.psi, self.cfg.collapse_q)
        if self.cfg.curvature_mode == "quantile":
            self.curv_threshold = np.quantile(self.curv, self.cfg.curvature_q)
        else:
            self.curv_threshold = self.cfg.curvature_thresh

        # counters
        self._curv_hits = 0

    # ---------- internals ----------
    def _apply_boundary(self, p: np.ndarray) -> np.ndarray:
        H, W = self.psi.shape
        mode = self.cfg.boundary
        if mode == "clip":
            return np.clip(p, [0, 0], [H - 1, W - 1])
        if mode == "wrap":
            return np.array([p[0] % H, p[1] % W])
        if mode == "reflect":
            def refl(v, L):
                q = np.abs(v) % (2 * (L - 1))
                return q if q <= (L - 1) else 2 * (L - 1) - q
            return np.array([refl(p[0], H), refl(p[1], W)])
        raise ValueError(f"Unknown boundary mode: {mode}")

    def _collapse_check(self, pos: np.ndarray) -> tuple[bool, str | None]:
        # low-Ψ check
        psi_here = bilinear_sample(self.psi, pos)
        if psi_here <= self.psi_threshold:
            return True, "low_psi"

        # curvature requires consecutive hits to avoid single-pixel spikes
        curv_here = bilinear_sample(self.curv, pos)
        if curv_here >= self.curv_threshold:
            self._curv_hits += 1
        else:
            self._curv_hits = 0

        if self._curv_hits >= self.cfg.curvature_consecutive:
            return True, "high_curvature"

        return False, None

    # ---------- public ----------
    def step(self, rng: np.random.Generator | None = None):
        if self.collapsed:
            return

        # sample gradient at current position
        grad_here = np.array([
            bilinear_sample(self.gx, self.pos),
            bilinear_sample(self.gy, self.pos)
        ])
        # main descent
        force = -grad_here

        # tangential (orbit) component: rotate gradient by +90°
        if self.cfg.orbit_bias != 0.0:
            R = np.array([[0.0, -1.0], [1.0, 0.0]])
            tang = R @ grad_here
            force = force + self.cfg.orbit_bias * tang

        # clip/normalize force for stability
        norm = np.linalg.norm(force) + 1e-12
        if self.cfg.adapt_force and norm > 0:
            force = force / norm
        if norm > self.cfg.max_force:
            force = force * (self.cfg.max_force / norm)

        # momentum + (optional) noise
        self.vel = self.cfg.beta_momentum * self.vel + self.cfg.alpha * force
        if rng is not None and self.cfg.noise_std > 0:
            self.vel += self.cfg.noise_std * rng.normal(size=2)

        # integrate and bound
        self.pos = self.pos + self.cfg.dt * self.vel
        self.pos = self._apply_boundary(self.pos)
        self.history.append(tuple(self.pos))
        self.steps += 1

        # stagnation: tiny average displacement over recent window
        if len(self.history) > self.cfg.stagnation_steps:
            recent = np.diff(np.array(self.history[-self.cfg.stagnation_steps:]), axis=0)
            if np.linalg.norm(recent, axis=1).mean() < self.cfg.stagnation_tol:
                self.collapsed = True
                self.collapse_reason = "stagnation"
                return

        # collapse checks after warmup
        if self.steps >= self.cfg.min_steps_before_collapse:
            hit, reason = self._collapse_check(self.pos)
            if hit:
                self.collapsed = True
                self.collapse_reason = reason
