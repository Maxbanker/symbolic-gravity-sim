# 🌌 Symbolic Gravity Simulator (SG-Sim v2.2)

A Python simulator for **Symbolic Gravity**, where attraction emerges not from mass-energy but from recursive **Ψ_eff fields** and entropy-translation dynamics.  
Implements the latest frameworks: **ERF v3.0**, **SFT v4.0**, **FCWF v2.0**, **OFT v3.0**, and **Hyperverse v6.0**.

---

## ✨ Features
- **Entropy Field Generation**  
  Gaussian, uniform, gradient, and hill-valley landscapes (`core/entropy_field.py`).

- **Ψ_eff Field Computation**  
  Log-ratio, ratio, or difference modes with percentile normalization and optional focus weighting (`core/psi_eff.py`).

- **Agents** (`core/agents.py`)  
  Momentum-driven symbolic agents with:
  - Continuous bilinear field sampling  
  - Gradient-based motion with curvature-adaptive orbit bias  
  - Noise injection & momentum  
  - Non-overlapping viable spawns (quantile + min-distance)  
  - Composite collapse detection:  
    - Low-Ψ collapse (`--collapse_q`)  
    - High-curvature collapse with hysteresis & consecutive hits  
    - Stagnation collapse (windowed mean & variance, `--max_age`)  

- **Metrics** (`core/metrics.py`)  
  Curvature proxy from Laplacian + gradient.

- **Simulator Runner** (`symbolic_collapse_sim.py`)  
  - Multi-agent simulation with configurable seeds  
  - Time-colored trajectories & optional curvature overlay  
  - CLI flags for thresholds, spawn rules, ψ normalization, artifacts  
  - PNG plots + JSON summaries per run  

---

## 🚀 Quick Start

### Requirements
- Python 3.9+  
- NumPy ≥ 2.0  
- Matplotlib  

### Install
```bash
git clone https://github.com/Maxbanker/symbolic-gravity-sim.git
cd symbolic-gravity-sim
pip install -r requirements.txt
````

### Run Simulation

```bash
python symbolic_collapse_sim.py --agents 4 --steps 300 --field hillvalley --seed 42 \
  --overlay_curvature --save run1.png --no-show
```

### CLI Options

* `--agents`: number of agents (default 4)
* `--steps`: timesteps (default 300)
* `--field`: entropy field (`gaussian`, `uniform`, `gradient`, `hillvalley`)
* `--collapse_q`: low-Ψ quantile threshold (default 0.15)
* `--curvature_q`: high-curvature quantile threshold (default 0.92)
* `--curv_consecutive`: hits required before collapse (default 4)
* `--curv_hysteresis`: hysteresis factor (default 0.85)
* `--stagnation_tol`: mean step size threshold
* `--stagnation_std`: variance threshold
* `--max_age`: collapse after this many steps if stagnant
* `--start_qpsi`, `--start_qcurv`, `--start_min_dist`: spawn policy
* `--overlay_curvature`: overlay curvature heatmap
* `--overlay_alpha`: curvature overlay opacity
* `--save`: filename to save PNG (+ JSON summary)
* `--no-show`: disable interactive window

---

## 📊 Example Output

* Contour map of normalized **Ψ\_eff**
* Agent trajectories, colored by time
* Start points = white, end points = black
* Collapse reasons annotated (`low_psi`, `high_curvature`, `stagnation`, `active`)
* Optional curvature heatmap overlay

Artifacts:

* **`runX.png`** — visualization
* **`runX.json`** — structured agent summary

```json
{
  "agents": [
    {"id":0, "steps":300, "status":"active", "start":[20,60], "end":[45,80]}
  ]
}
```

---

## 🔬 Research Context

This simulator operationalizes **Symbolic Gravity v2.2**:

* **ERF v3.0** — Collapse as translation
* **SFT v4.0** — Symbolic curvature tensors
* **FCWF v2.0** — Fractal coherence threads
* **OFT v3.0** — Observer as invariant export operator
* **Hyperverse v6.0** — Collapse as translation-gate continuum

For theory details, see:
`/core/Gravity Explainer`

---

## 📌 Roadmap

* [ ] Expose `noise_std` and entropy bump density as CLI flags
* [ ] Add TensorFlow/JAX backends
* [ ] Export flux/curvature field visualizations
* [ ] Multi-agent entanglement experiments
* [ ] Conservation law audits (ΔS + ΔE = 0)

---

## 📜 License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to use and adapt with attribution.

---

## 👤 Author

**Steven Lanier-Egu**
Egu Technologies
[admin@egutechnologies.com](mailto:admin@egutechnologies.com)

```
