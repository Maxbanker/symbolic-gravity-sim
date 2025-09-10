# tests/test_agents.py
import numpy as np
import pytest
from symbolic_gravity_sim.core.agents import AgentConfig, SymbolicAgent
from symbolic_gravity_sim.core.psi_eff import compute_psi_eff
from symbolic_gravity_sim.core.entropy_field import generate_entropy_field
from symbolic_gravity_sim.core.metrics import curvature

def test_agent_collapse_low_psi():
    psi = np.zeros((120, 160))  # Force low Ψ_eff
    curv = np.ones((120, 160))
    cfg = AgentConfig(collapse_q=0.99)  # High threshold to trigger collapse
    agent = SymbolicAgent(psi, curv, start_xy=(10, 10), cfg=cfg)
    agent.prime_thresholds()
    agent.step()
    assert not agent.alive
    assert agent.collapse_reason == "low_psi"

def test_agent_stagnation():
    psi = np.ones((120, 160))
    curv = np.zeros((120, 160))
    cfg = AgentConfig(stagnation_window=5, stagnation_tol=0.01, stagnation_std=0.01, max_age=5)
    agent = SymbolicAgent(psi, curv, start_xy=(10, 10), cfg=cfg)
    agent.prime_thresholds()
    for _ in range(6):
        agent.step()
    assert not agent.alive
    assert agent.collapse_reason == "stagnation"