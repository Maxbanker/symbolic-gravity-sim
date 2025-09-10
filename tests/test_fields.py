# tests/test_fields.py
import numpy as np
import pytest
from symbolic_gravity_sim.core.entropy_field import generate_entropy_field
from symbolic_gravity_sim.core.psi_eff import compute_psi_eff

def test_entropy_field_range():
    shape = (120, 160)
    for method in ["hillvalley", "gaussian", "uniform", "gradient"]:
        field = generate_entropy_field(shape=shape, method=method, seed=42)
        assert field.shape == shape
        assert np.all(field >= 0.0) and np.all(field <= 1.0)

def test_psi_eff_normalization():
    info = np.random.uniform(0, 1, (120, 160))
    entropy = np.random.uniform(0, 1, (120, 160))
    psi = compute_psi_eff(info, entropy, mode="log_ratio", p1=1.0, p99=99.0)
    assert np.all(psi >= 0.0) and np.all(psi <= 1.0)