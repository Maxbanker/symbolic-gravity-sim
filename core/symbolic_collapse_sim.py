# symbolic_collapse_sim.py
"""
Main runner script for simulating symbolic agent collapse in a Ψ_eff field.
Now using continuous dynamics, curvature-aware collapse, and multi-agent runs.
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from core.entropy_field import generate_entropy_field
from core.psi_eff import compute_psi_eff
from core.agents import SymbolicAgent, AgentConfig

def run(seed=42, n_agents=3, steps=200, field="gaussian"):
    rng = np.random.default_rng(seed)

    # 1) Entropy + info
    entropy = generate_entropy_field(shape=(120, 160), method=field,
                                     mean=0.4, std=0.12, seed=seed)
    info = 1.0 - entropy

    # 2) Ψ_eff (normalized 0..1)
    psi = compute_psi_eff(info_density=info, entropy_density=entropy, mode="log_ratio")

    # 3) Agents
    cfg = AgentConfig(dt=1.0, alpha=0.35, beta_momentum=0.85,
                      noise_std=0.005, boundary="reflect",
                      stagnation_tol=2e-3, stagnation_steps=10,
                      collapse_q=0.07, curvature_thresh=0.18)

    agents = []
    for _ in range(n_agents):
        start = np.array([rng.uniform(0, psi.shape[0]-1),
                          rng.uniform(0, psi.shape[1]-1)])
        agents.append(SymbolicAgent(start, psi, cfg))

    # 4) Sim loop
    for _ in range(steps):
        for ag in agents:
            ag.step(rng)

    # 5) Viz
    plt.figure(figsize=(10, 6))
    plt.imshow(psi, cmap="viridis", origin="lower")
    for i, ag in enumerate(agents):
        xs, ys = zip(*ag.history)
        color = "red" if ag.collapsed else "orange"
        plt.plot(ys, xs, lw=1.8, c=color, label=f"agent {i} {'(collapsed)' if ag.collapsed else ''}")
        plt.scatter([ys[0]],[xs[0]], s=22, c="white")
        plt.scatter([ys[-1]],[xs[-1]], s=22, c="black")
    plt.title("Symbolic Agents in Ψ_eff (log-ratio; curvature-aware collapse)")
    cbar = plt.colorbar(); cbar.set_label("Ψ_eff (0..1)")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.show()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--field", type=str, default="gaussian",
                   choices=["gaussian","uniform","gradient","hillvalley"])
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(seed=args.seed, n_agents=args.agents, steps=args.steps, field=args.field)
