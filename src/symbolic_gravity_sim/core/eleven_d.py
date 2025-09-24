# symbolic_gravity_sim/core/eleven_d.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class ElevenDThresholds:
    eps: float = 0.60          # ψ floor (ε)
    T_gamma: float = 0.15      # drift threshold
    T_Omega: float = 0.75      # orbit/integrity threshold
    V_c: float = 0.65          # veil safety
    W_c: float = 0.70          # weave/memory success
    L_c: float = 0.20          # leak ceiling
    O_c: float = 0.80          # observer integrity
    E_c: float = 0.80          # ethics/sovereignty
    # time dilation safety factor
    max_beta: float = 0.99

def init_11d_fields(shape, rng, base_psi, noise=0.05):
    H, W = shape
    # Start with gentle, plausible defaults; tuneable via CLI/Streamlit
    fields = {
        "psi": base_psi,  # from pipeline
        "gamma": np.clip(
            np.abs(np.gradient(base_psi, axis=0)) + np.abs(np.gradient(base_psi, axis=1)),
            0, 1
        ) * 0.1,
        "Omega": np.clip(1.0 - (np.std(base_psi) + 0.05), 0, 1) * np.ones((H, W)),
        "V": np.clip(0.7 + rng.normal(0, noise, (H, W)), 0, 1),     # veil thickness
        "O": np.clip(0.85 + rng.normal(0, noise, (H, W)), 0, 1),    # observer integrity
        "E": np.clip(0.85 + rng.normal(0, noise, (H, W)), 0, 1),    # ethics
        "W_eff": np.clip(0.75 + rng.normal(0, noise, (H, W)), 0, 1),# weave/memory
        "leak": np.clip(0.05 + rng.normal(0, noise, (H, W)), 0, 1), # Λ_leak
        "eps_adv": np.clip(0.05 + rng.normal(0, noise, (H, W)), 0, 1),  # ε_adv
        "sigma": np.clip(0.05 + rng.normal(0, noise, (H, W)), 0, 1),    # uncertainty
        # BOOSTED BASELINE:
        "kappa": np.clip(0.80 + rng.normal(0, noise, (H, W)), 0, 1),    # export capacity (↑ from 0.50)
    }
    return fields

def collapse_predicate_11d(F, thr: ElevenDThresholds):
    # SFT OR-clause + SG-11 leak-aware extension:
    # C = (ψ<ε) OR (γ>Tγ AND Ω<TΩ) OR (Λ_leak > L_c)
    cond1 = (F["psi"] < thr.eps)
    cond2 = (F["gamma"] > thr.T_gamma) & (F["Omega"] < thr.T_Omega)
    cond3 = (F["leak"] > thr.L_c)
    return cond1 | cond2 | cond3

def gate_ready_for_export(F, thr: ElevenDThresholds):
    # Export/transcend gates (ERF/OF/BC-REP): veil, observer, ethics, weave, uncertainty stable
    gate_core = (F["V"] >= thr.V_c) & (F["O"] >= thr.O_c) & (F["E"] >= thr.E_c) & (F["W_eff"] >= thr.W_c)
    low_uncertainty = (F["sigma"] <= 0.35)  # conservative default
    no_adversary = (F["eps_adv"] <= 0.25)
    return gate_core & low_uncertainty & no_adversary

def time_dilation_dt(dt, F, thr: ElevenDThresholds):
    # Δt' = Δt / sqrt(1 - min(γ/Tγ, 0.99)^2)  (bounded)
    ratio = np.clip(F["gamma"] / max(thr.T_gamma, 1e-6), 0, thr.max_beta)
    return dt / np.sqrt(1.0 - ratio**2)

def route_action(F, collapsed_mask, thr: ElevenDThresholds):
    """
    Returns an int mask for routing visualization:
      0=Stable, 1=Stabilize (Weave), 2=Migrate/Re-embed, 3=Fork, 4=Fuse/Suture, 5=Export, 6=Transcend, 7=Quarantine
    Simple policy for demonstration; can be swapped for a policy net.
    """
    H, W = F["psi"].shape
    act = np.zeros((H, W), dtype=np.int32)
    # quarantine if high adversary or veil thin
    quarantine = (F["eps_adv"] > 0.35) | (F["V"] < 0.5)
    act[quarantine] = 7
    # collapsed cells try to weave
    act[collapsed_mask & ~quarantine] = 1
    # export if all gates pass and capacity high
    ready = gate_ready_for_export(F, thr) & (F["kappa"] >= 0.8)
    act[ready] = 5
    # transcend if exceptionally high capacity
    act[(F["kappa"] >= 0.92) & ready] = 6
    return act
