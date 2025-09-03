# symbolic_collapse_sim.py
"""
Symbolic Gravity Simulator — longer trajectories
- viable spawning (Ψ high, curvature modest)
- warmup before collapse
- curvature collapse by quantile + consecutive hits
- orbit bias to avoid straight dives

Run:
    python symbolic_collapse_sim.py --agents 4 --steps 300 --field hillvalley --seed 42
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from core.entropy_field import generate_entropy_field
from core.psi_eff import compute_psi_eff
from core.metrics import curvature
from core.agents import SymbolicAgent, AgentConfig


def _sample_starts(rng: np.random.Generator,
                   psi: np.ndarray,
                   curv: np.ndarray,
                   n_agents: int,
                   psi_q: float = 0.35,
                   curv_cap: float = 0.33) -> list[np.ndarray]:
    """Choose starting points where Ψ is decent and curvature not extreme."""
    H, W = psi.shape
    mask = (psi >= np.quantile(psi, psi_q)) & (curv <= curv_cap)
    ys, xs = np.where(mask)
    if len(xs) == 0:  # fallback if mask too strict
        ys, xs = np.where(psi >= np.quantile(psi, 0.25))
    idx = rng.choice(len(xs), size=n_agents, replace=True)
    starts = np.stack([ys[idx] + rng.uniform(0, 1, size=n_agents),
                       xs[idx] + rng.uniform(0, 1, size=n_agents)], axis=1)
    return [starts[i] for i in range(n_agents)]


def run(seed=42, n_agents=4, steps=300, field="hillvalley",
        height=120, width=160, orbit_bias=0.55):
    rng = np.random.default_rng(seed)
    shape = (height, width)

    # 1) Entropy field (structured)
    entropy = generate_entropy_field(
        shape=shape,
        method=field,
        mean=0.4, std=0.12,           # for gaussian (ignored by hillvalley)
        seed=seed,
        n_bumps=8, sigma=0.10
    )

    # 2) Info + Ψ (0..1 normalized)
    info = 1.0 - entropy
    psi = compute_psi_eff(info_density=info, entropy_density=entropy, mode="log_ratio")

    # 3) Curvature (0..1 normalized) for spawn filter + collapse checks
    curv = curvature(psi, normalize=True)

    # 4) Agent config tuned for longer runs
    cfg = AgentConfig(
        dt=1.0, alpha=0.38, beta_momentum=0.90,
        noise_std=0.0015, boundary="reflect",
        stagnation_tol=3.5e-3, stagnation_steps=16,
        min_steps_before_collapse=14,
        collapse_q=0.015,                 # treat only the lowest ~1.5% as "low Ψ"
        curvature_mode="quantile",
        curvature_q=0.94,                 # collapse only in top ~6% most curved zones
        curvature_thresh=0.35,            # ignored in quantile mode
        curvature_consecutive=7,          # require 7 consecutive high-curv steps
        orbit_bias=orbit_bias,            # tangential drift to avoid straight dives
        max_force=2.2, adapt_force=True
    )

    # 5) Spawn in viable zones
    starts = _sample_starts(rng, psi, curv, n_agents, psi_q=0.35, curv_cap=0.33)
    agents = [SymbolicAgent(start, psi, cfg) for start in starts]

    # 6) Simulate
    for _ in range(steps):
        for ag in agents:
            ag.step(rng)

    # 7) Plot
    plt.figure(figsize=(10, 6))
    im = plt.imshow(psi, cmap="viridis", origin="lower")
    try:
        plt.contour(psi, levels=12, colors='k', alpha=0.15, linewidths=0.6)
    except Exception:
        pass

    for i, ag in enumerate(agents):
        xs, ys = zip(*ag.history)
        label = f"agent {i} {'(collapsed)' if ag.collapsed else '(active)'}"
        plt.plot(ys, xs, lw=2.4, alpha=0.95, label=label)
        plt.scatter([ys[0]], [xs[0]], s=42, c="white", edgecolors='k', zorder=3)
        plt.scatter([ys[-1]], [xs[-1]], s=42, c="black", edgecolors='w', zorder=3)

    plt.title("Symbolic Agents in Ψ_eff (log-ratio; curvature-aware collapse)")
    cbar = plt.colorbar(im); cbar.set_label("Ψ_eff (0..1)")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.show()

    # 8) Summary
    for i, ag in enumerate(agents):
        reason = ag.collapse_reason if getattr(ag, "collapse_reason", None) else ("collapsed" if ag.collapsed else "active")
        print(f"[agent {i}] steps={len(ag.history)-1} | status={reason}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--field", type=str, default="hillvalley",
                   choices=["hillvalley", "gaussian", "uniform", "gradient"])
    p.add_argument("--height", type=int, default=120)
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--orbit_bias", type=float, default=0.55)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(seed=args.seed, n_agents=args.agents, steps=args.steps,
        field=args.field, height=args.height, width=args.width,
        orbit_bias=args.orbit_bias)
