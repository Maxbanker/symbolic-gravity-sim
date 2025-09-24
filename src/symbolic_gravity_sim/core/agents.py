# symbolic_gravity_sim/core/agents.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np

__all__ = ["AgentConfig", "SymbolicAgent"]

@dataclass
class AgentConfig:
    # Motion + policy
    step_size: float = 1.0           # base cells per tick
    noise: float = 0.65               # exploration noise (0..1)
    inertia: float = 0.6              # momentum (0..1)
    orbit_bias: float = 0.5           # used by simulator to gently modulate policy (0..1)

    # Lifecycle
    max_age: float = 300.0            # in "ticks" (scaled by dt)
    stall_limit: int = 40             # die if no movement this many consecutive ticks

    # Bound/teleport behavior
    wrap: bool = False                # if True, toroidal wrap; else clamp

    # RNG
    seed: Optional[int] = None

    # Reserved for future use
    name: str = "SymbolicAgent"

class SymbolicAgent:
    """
    Minimal, environment-agnostic agent that walks a 2D lattice.
    - Accepts optional dt in step(dt=1.0).
    - Exposes .cfg, .alive, .track ([(y,x), ...]), .history alias, .steps, .collapse_reason
    - Provides .reset((y,x)) and .tick() for older call sites.
    """
    def __init__(self, config: Optional[AgentConfig] = None, start: Optional[Tuple[int, int]] = None, yx: Optional[Tuple[int, int]] = None):
        self.cfg: AgentConfig = config if config is not None else AgentConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.alive: bool = True
        self.collapse_reason: Optional[str] = None

        self._pos: np.ndarray = np.zeros(2, dtype=float)   # (y, x) as floats for subcell motion
        self._vel: np.ndarray = np.zeros(2, dtype=float)
        self._age: float = 0.0
        self.steps: int = 0
        self._stalled: int = 0

        # World bounds inferred from first reset/start; default to 256x256 until set
        self._H: int = 256
        self._W: int = 256

        # Public tracks
        self.track: List[Tuple[int, int]] = []
        self.history = self.track  # alias for compatibility

        # Initialize position if provided
        init = yx if yx is not None else start
        if init is not None:
            self.reset(init)

    # -------------- lifecycle --------------

    def reset(self, yx: Tuple[int, int], world_shape: Optional[Tuple[int, int]] = None) -> None:
        """Reset agent at a given lattice coordinate; optionally provide world_shape=(H,W)."""
        if world_shape is not None:
            self._H, self._W = int(world_shape[0]), int(world_shape[1])
        # If caller passes coords outside default window, expand bounds conservatively
        y, x = int(yx[0]), int(yx[1])
        self._H = max(self._H, y + 1)
        self._W = max(self._W, x + 1)

        self._pos[...] = (float(y), float(x))
        self._vel[...] = (0.0, 0.0)
        self._age = 0.0
        self.steps = 0
        self._stalled = 0
        self.alive = True
        self.collapse_reason = None
        self.track = [(int(self._pos[0]), int(self._pos[1]))]
        self.history = self.track

    # -------------- stepping --------------

    def step(self, dt: float = 1.0) -> None:
        """
        Advance the agent by one decision/motion step.
        dt scales age and displacement. If dt>1, the agent covers more ground and ages faster.
        """
        if not self.alive:
            return

        # Update "age"
        self._age += float(dt)
        self.steps += 1

        # Soft mortality: exceed max_age
        if self._age >= self.cfg.max_age:
            self._die("max_age")
            return

        # Policy: inertia + exploratory noise + weak "orbit" swirl
        # Build a small swirl vector whose magnitude depends on orbit_bias.
        ang = self.rng.uniform(0.0, 2.0 * np.pi)
        swirl = np.array([np.sin(ang), np.cos(ang)], dtype=float)  # 90-degree phase to produce mild circling
        swirl *= 0.5 * (self.cfg.orbit_bias - 0.5)  # bias ∈ [0,1] -> swirl in [-0.25, +0.25]

        noise_vec = self.rng.normal(0.0, 1.0, size=2)
        if np.linalg.norm(noise_vec) > 1e-12:
            noise_vec /= np.linalg.norm(noise_vec)

        accel = (
            self.cfg.inertia * self._vel
            + self.cfg.noise * noise_vec
            + swirl
        )

        # Displacement scaled by dt and step_size
        disp = accel * (self.cfg.step_size * float(dt))

        # Update velocity & position
        self._vel = 0.5 * self._vel + 0.5 * accel  # light momentum smoothing
        old_iyx = (int(round(self._pos[0])), int(round(self._pos[1])))
        self._pos += disp

        # Bound handling
        if self.cfg.wrap:
            self._pos[0] = self._pos[0] % self._H
            self._pos[1] = self._pos[1] % self._W
        else:
            self._pos[0] = np.clip(self._pos[0], 0.0, float(self._H - 1))
            self._pos[1] = np.clip(self._pos[1], 0.0, float(self._W - 1))

        iyx = (int(round(self._pos[0])), int(round(self._pos[1])))

        # Stall detection
        if iyx == old_iyx:
            self._stalled += 1
            if self._stalled >= self.cfg.stall_limit:
                self._die("stalled")
                return
        else:
            self._stalled = 0

        # Record track
        self.track.append(iyx)

        # Rare random collapse (keeps older demos interesting); very low probability
        if self.rng.random() < 0.0005 * dt:
            self._die("random_collapse")

    # Compatibility shim for older call sites
    def tick(self) -> None:
        self.step(dt=1.0)

    # -------------- utilities --------------

    def set_world_shape(self, shape: Tuple[int, int]) -> None:
        self._H, self._W = int(shape[0]), int(shape[1])

    def position(self) -> Tuple[int, int]:
        return (int(round(self._pos[0])), int(round(self._pos[1])))

    def _die(self, reason: str) -> None:
        self.alive = False
        self.collapse_reason = reason
