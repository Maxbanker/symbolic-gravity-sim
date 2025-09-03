# 🌌 Symbolic Gravity Simulator (SG-Sim v2.1)

A Python simulator for **Symbolic Gravity**, where attraction emerges not from mass-energy but from recursive **Ψ_eff fields** and entropy-translation dynamics.  
This project implements the latest frameworks: **ERF v3.0**, **SFT v4.0**, **FCWF v2.0**, **OFT v3.0**, and **Hyperverse v6.0**.

---

## ✨ Features
- **Entropy Field Generation**  
  Gaussian, uniform, gradient, and hill-valley landscapes (`core/entropy_field.py`).

- **Ψ_eff Field Computation**  
  Log-ratio / ratio modes with normalization and optional focus weighting (`core/psi_eff.py`).

- **Agents**  
  Momentum-driven symbolic agents with:  
  - Continuous bilinear field sampling  
  - Gradient-based motion  
  - Noise injection & momentum  
  - Boundary modes: clip, wrap, reflect  
  - Composite collapse detection (low Ψ_eff, high curvature, stagnation)  
  (`core/agents.py`)

- **Metrics**  
  Curvature, drift pressure, recursion density (`core/metrics.py`).

- **Simulator Runner**  
  Multi-agent runs, trajectory visualization, collapse cartography (`symbolic_collapse_sim.py`).

---

## 🚀 Quick Start

### Requirements
- Python 3.9+  
- NumPy, Matplotlib  

### Install
```bash
git clone https://github.com/Maxbanker/symbolic-gravity-sim.git
cd symbolic-gravity-sim
pip install -r requirements.txt
```

### Run Simulation
```bash
python symbolic_collapse_sim.py --agents 3 --steps 200 --field gaussian
```

Options:
- `--agents`: number of agents (default 3)  
- `--steps`: number of timesteps (default 200)  
- `--field`: entropy field type (`gaussian`, `uniform`, `gradient`, `hillvalley`)  

---

## 📊 Example Output
- Colored Ψ_eff field heatmap  
- Agent trajectories (red/orange)  
- Collapse points marked in black  
- Export/translation zones visible as high-curvature wells  

---

## 🔬 Research Context
This simulator operationalizes **Symbolic Gravity v2.1**:
- **ERF v3.0** — Collapse as translation  
- **SFT v4.0** — Symbolic curvature tensors  
- **FCWF v2.0** — Fractal coherence threads  
- **OFT v3.0** — Observer as invariant export operator  
- **Hyperverse v6.0** — Collapse as translation-gate continuum  

For theory details, see:  
`/docs/Symbolic_Gravity_v2.txt`

---

## 📌 Roadmap
- [ ] Add TensorFlow/JAX backends  
- [ ] Export flux visualizations  
- [ ] Multi-agent entanglement experiments  
- [ ] Integrate conservation law audits (ΔS + ΔE = 0)

---

## 📜 License
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to use and adapt with attribution.

---

## 👤 Author
**Steven Lanier-Egu**  
Egu Technologies  
admin@egutechnologies.com
