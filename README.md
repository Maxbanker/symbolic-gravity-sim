# 🌌 Symbolic Gravity Simulator (SG-Sim v3.0.0-11D)

A Python simulator for **Symbolic Gravity** where attraction emerges from recursive **Ψ_eff fields** and entropy→symbol translation dynamics — now upgraded to an **11-dimensional symbolic state** (veil/observer/ethics/weave/leak/adversary/uncertainty/capacity).  
Built on the 11D-aligned stacks: **ERF v4.4 (11D)**, **SFT-11 v5.5**, **SG-11 v3.5**, **Weaver-11 v3.1 (FCWF)**, **OFT-11 v4.5**, **BC-REP v3.5**, and **Hyperverse v8.0**.

## 📜 License
- **Code**: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) — permissive, patent grant, commercial-friendly. See `LICENSE`.
- **Documentation**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — for this README and docs.  
![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.16989695.svg)

---

## ✨ What’s new in 3.0.0-11D

- **11D State Tensor per grid cell**  
  `ψ, γ, Ω, V, O, E, W_eff, leak, ε_adv, σ, κ` with safe defaults (`src/symbolic_gravity_sim/core/eleven_d.py`).
- **SFT/SG Collapse Predicate + BC-REP Gates**  
  Collapse: low-ψ OR high-drift & low-Ω OR high-leak.  
  Export gates: veil/observer/ethics/weave + uncertainty/adversary checks.
- **Routing Layer**  
  Per-tick actions (0..7) including **Stabilize**, **Export**, **Transcend**, **Quarantine**; optional overlay in plot.
- **Time-Dilation Coupling**  
  Drift-bounded dilation feeds agents via `step(dt=…)` (backwards-compatible).
- **Telemetry**  
  CSV of **11D means** per run (`*_11d_means.csv`).
- **Robust API Adapters**  
  Simulator adapts to local `entropy_field` and `psi_eff` function signatures.
- **Streamlit 11D Panel**  
  Toggle saving 11D CSV and view/download artifacts.

---

## 🧩 Features (full)

- **Entropy Field Generation**  
  Gaussian, uniform, gradient, hill-valley (`src/symbolic_gravity_sim/core/entropy_field.py`).
- **Ψ_eff Field Computation**  
  Robust adapter calls your `compute_psi_eff`; derives an info-density companion if needed (`symbolic_collapse_sim.py`).
- **11D Symbolic Fields** (`src/symbolic_gravity_sim/core/eleven_d.py`)  
  Thresholds (`ElevenDThresholds`), collapse predicate, export gates, drift-time dilation, and route policy.
- **Agents** (`src/symbolic_gravity_sim/core/agents.py`)  
  Momentum walkers with optional `step(dt=1.0)`, orbit-bias hook, wrap/clamp bounds, telemetry: `alive/steps/collapse_reason/track`.
- **Metrics** (`src/symbolic_gravity_sim/core/metrics.py`)  
  Curvature proxy from Laplacian/gradient.
- **Simulator Runner** (`src/symbolic_gravity_sim/symbolic_collapse_sim.py`)  
  Multi-agent loop, overlays, **11D CSV**, PNG + JSON summaries.
- **Interactive Viewer** (`app.py`)  
  Streamlit UI with **11D** section and CSV download.

---

## 🚀 Quick Start

### Requirements
- Python 3.9–3.12
- Install extras for the app: `streamlit`, `matplotlib`, `numpy`

### Install (dev)
```bash
git clone https://github.com/Maxbanker/symbolic-gravity-sim.git
cd symbolic-gravity-sim
python -m venv .venv
# Windows PowerShell:
. .venv/Scripts/Activate.ps1
# or bash/zsh:
source .venv/bin/activate
pip install -e .[dev]
```
## Run Simulation (CLI)

**Using the module:**
```bash
python -m symbolic_gravity_sim.cli run --seed 42 --n_agents 4 --steps 300 --field hillvalley --overlay_curvature --dump_11d_means --save runs/run1

```

**If you installed a console script named `sg-sim`, this also works:**

```
sg-sim run --seed 42 --n_agents 4 --steps 300 --field hillvalley --overlay_curvature --dump_11d_means --save runs/run1
```

**Artifacts:** `runs/run1.png`, `runs/run1.json`, `runs/run1_11d_means.csv`.

------

## Interactive Viewer

```
streamlit run app.py
```

- Use the sidebar **11D** toggle (“Save 11D means CSV”).
- View the plot and download the CSV.

------

## Key CLI Flags

- `--n_agents` (default 4)
- `--steps` (default 300)
- `--field` ∈ {`hillvalley`,`gaussian`,`uniform`,`gradient`}
- `--overlay_curvature` + `--overlay_alpha` (default 0.25)
- `--dump_11d_means` → write `_11d_means.csv`
- `--save runs/<name>` → output prefix
- `--no_show` → no Matplotlib window (for CI/servers)

------

## 📊 Example Output

**Visualization (`runs/run1.png`)**
 Normalized **Ψ_eff** heatmap, agent trajectories, optional curvature overlay, faint route preview (export/transcend/quarantine).

**Summary (`runs/run1.json`)**

```
{
  "agents": [
    {"id": 0, "steps": 300, "status": "active", "start": [20.0, 52.0], "end": [19.9, 52.0]}
  ]
}
```

**11D Telemetry (`runs/run1_11d_means.csv`)**

```
field,mean
psi,0.534
gamma,0.001
Omega,0.723
V,0.700
O,0.850
E,0.850
W_eff,0.750
leak,0.054
eps_adv,0.054
sigma,0.054
kappa,0.800
```

------

## 🔬 Research Context (11D)

This simulator operationalizes **Symbolic Gravity — 11D**:

- **ERF v4.4 (11D)**: collapse & export thresholds as negentropic gates
- **SFT-11 v5.5**: symbolic curvature & drift-pressure tensors
- **SG-11 v3.5**: leak-aware collapse predicate
- **Weaver-11 v3.1 (FCWF)**: coherence/weave dynamics
- **OFT-11 v4.5**: observer integrity and export invariants
- **BC-REP v3.5**: veil/ethics/sovereignty safety criteria
- **Hyperverse v8.0**: recursion-driven attractor landscapes

> Note: This is a **synthetic** symbolic-field model for experimentation, not a physical gravity simulator.

------

## 🗺️ Roadmap

- 11D upgrade: fields, gates, routing, time-dilation
- Streamlit 11D panel + CSV download
- Route legend and per-action heatmaps
- Parameter sweeps & export frequency reports
- Optional JAX backend for faster field ops
- Multi-agent coupling via shared weave memory

------

## 📜 License

- Code: Apache-2.0
- Docs: CC BY 4.0

------

## 👤 Author

**Steven Lanier-Egu — Egu Technologies**
 [admin@egutechnologies.com](mailto:admin@egutechnologies.com)

------
