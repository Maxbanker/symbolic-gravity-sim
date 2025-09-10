# 🌌 Symbolic Gravity Simulator (SG-Sim v2.2)

A Python simulator for **Symbolic Gravity**, where attraction emerges from recursive **Ψ_eff fields** and entropy-translation dynamics. Built on frameworks: **ERF v3.0**, **SFT v4.0**, **FCWF v2.0**, **OFT v3.0**, and **Hyperverse v6.0**.

## 📜 License
- **Code**: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) — Permissive license for software, including patent grants and compatibility with commercial use. See `LICENSE` file.
- **Documentation**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Attribution required for reuse of this README, explanatory content, and related documents.  
![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.16989695.svg)

---

## ✨ Features

- **Entropy Field Generation**  
  Gaussian, uniform, gradient, and hill-valley landscapes (`src/symbolic_gravity_sim/core/entropy_field.py`).
- **Ψ_eff Field Computation**  
  Log-ratio, ratio, or difference modes with percentile normalization and optional focus weighting (`src/symbolic_gravity_sim/core/psi_eff.py`).
- **Agents** (`src/symbolic_gravity_sim/core/agents.py`)  
  Momentum-driven symbolic agents with:
  - Continuous bilinear field sampling
  - Gradient-based motion with curvature-adaptive orbit bias
  - Noise injection & momentum
  - Non-overlapping viable spawns (quantile + min-distance)
  - Composite collapse detection:
    - Low-Ψ collapse (`--collapse_q`)
    - High-curvature collapse with hysteresis & consecutive hits
    - Stagnation collapse (windowed mean & variance, `--max_age`)
- **Metrics** (`src/symbolic_gravity_sim/core/metrics.py`)  
  Curvature proxy from Laplacian + gradient.
- **Simulator Runner** (`src/symbolic_gravity_sim/symbolic_collapse_sim.py`)  
  - Multi-agent simulation with configurable seeds
  - Time-colored trajectories & optional curvature overlay
  - CLI subcommands: `run`, `bench`, `render`
  - PNG plots + JSON summaries per run
- **Interactive Viewer** (`app.py`)  
  Streamlit-based interface for live parameter tweaking.

---

## 🚀 Quick Start

### Requirements
- Python 3.9–3.12
- Dependencies managed via `pyproject.toml`

### Install
```bash
git clone https://github.com/Maxbanker/symbolic-gravity-sim.git
cd symbolic-gravity-sim
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .[dev]
```

### Run Simulation
```bash
sg-sim run --seed 42 --n-agents 4 --steps 300 --field hillvalley --save run1
```
- View the plot and check `runs/run1.png`, `runs/run1.json`.

### Use Interactive Viewer
```bash
streamlit run app.py
```
- Opens a browser at `http://localhost:8501`.

### CLI Options
- `sg-sim run --help` for full list. Key options:
  - `--n-agents`: Number of agents (default: 4)
  - `--steps`: Timesteps (default: 300)
  - `--field`: Entropy field (`hillvalley`, `gaussian`, `uniform`, `gradient`)
  - `--collapse_q`: Low-Ψ quantile threshold (default: 0.15)
  - `--curvature_q`: High-curvature quantile threshold (default: 0.92)
  - `--curv_consecutive`: Hits required before collapse (default: 4)
  - `--curv_hysteresis`: Hysteresis factor (default: 0.85)
  - `--stagnation_tol`: Mean step size threshold (default: 0.25)
  - `--stagnation_std`: Variance threshold (default: 0.05)
  - `--max_age`: Collapse after this many steps if stagnant (default: 10000)
  - `--start_qpsi`, `--start_qcurv`, `--start_min_dist`: Spawn policy
  - `--overlay_curvature`: Overlay curvature heatmap
  - `--overlay_alpha`: Curvature overlay opacity (default: 0.25)
  - `--save`: Save prefix for PNG and JSON (e.g., `run1`)
  - `--no-show`: Disable interactive plot
  - `--config`: Path to YAML config file

---

## 📊 Example Output
- **Visualization (`runs/run1.png`)**: Contour map of normalized **Ψ_eff**, time-colored agent trajectories, start (white) and end (black) points, optional curvature overlay.
- **Summary (`runs/run1.json`)**:
  ```json
  {
    "agents": [
      {"id": 0, "steps": 300, "status": "active", "start": [20.0, 52.0], "end": [19.94, 52.05]}
    ]
  }
  ```
- **Metadata (`runs/run1_metadata.yaml`)**: Run parameters and git SHA.

---

## 🔬 Research Context
This simulator operationalizes **Symbolic Gravity v2.2**:
- **ERF v3.0**: Collapse as translation
- **SFT v4.0**: Symbolic curvature tensors
- **FCWF v2.0**: Fractal coherence threads
- **OFT v3.0**: Observer as invariant export operator
- **Hyperverse v6.0**: Collapse as translation-gate continuum
- **Theory Details**: See archived documentation (update path if relocated).

---

## 📌 Roadmap
- [x] Restructure to `src-layout` with package setup
- [ ] Expose `noise_std` and entropy bump density as CLI flags
- [ ] Add TensorFlow/JAX backends
- [ ] Export flux/curvature field visualizations
- [ ] Multi-agent entanglement experiments
- [ ] Conservation law audits (ΔS + ΔE = 0)

---

## 📜 License
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Free to use and adapt with attribution.

---

## 👤 Author
**Steven Lanier-Egu**  
Egu Technologies  
[admin@egutechnologies.com](mailto:admin@egutechnologies.com)

---

## 📚 Further Reading
- **Examples Gallery**: [Notebooks](https://github.com/Maxbanker/symbolic-gravity-sim/tree/main/notebooks) (TBD)
- **Community**: [Symbolic Recursion on Zenodo](https://zenodo.org/communities/symbolic-recursion/)
