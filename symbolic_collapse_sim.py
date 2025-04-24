"""
symbolic_collapse_sim.py
Main runner script for simulating symbolic agent collapse in a Ψeff field
"""

import numpy as np
import matplotlib.pyplot as plt
from core.entropy_field import generate_entropy_field
from core.psi_eff import compute_psi_eff
from core.agents import SymbolicAgent

# 1. Generate entropy field
entropy = generate_entropy_field(shape=(100, 100), method="gaussian", mean=0.4, std=0.1, seed=42)

# 2. Generate synthetic info density (opposite of entropy, here)
info = 1.0 - entropy

# 3. Compute Ψeff
psi_eff = compute_psi_eff(info_density=info, entropy_density=entropy)

# 4. Initialize symbolic agent
agent = SymbolicAgent(position=(10, 10))

# 5. Run simulation steps
for _ in range(100):
    agent.update(psi_eff)
    if agent.collapsed:
        break

# 6. Visualization
x, y = zip(*agent.history)
plt.figure(figsize=(8, 6))
plt.imshow(psi_eff, cmap="viridis", origin="lower")
plt.plot(y, x, color="red", linewidth=2, label="Agent Path")
plt.scatter([y[0]], [x[0]], color="white", label="Start")
plt.scatter([y[-1]], [x[-1]], color="black", label="End")
plt.title("Symbolic Agent Trajectory in Ψeff Field")
plt.colorbar(label="Ψeff")
plt.legend()
plt.show()
